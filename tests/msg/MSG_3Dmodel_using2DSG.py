import meshio
import numpy as np
import fea_traditional as test
from helper import *
import pyvista
import jax.numpy as jnp
import jax
import time
np.set_printoptions(precision=5) 
start = time.time()
#mesh = meshio.read(get_mesh("microscale_2D_r0.vtk")) 2D_mat1_2elem_tri
mesh = meshio.read("UDcomp_2D.msh")
#mesh = meshio.read("2D_mat1_2elem_tri.msh")
points = np.array(mesh.points, dtype=np.float32)[:, 0:2]
cells = np.array(mesh.cells[0].data, dtype=np.uint64)
#print("# DoFs = ", 3 * points.shape[0]) 

# 2. Convert meshio object to a PyVista mesh
#pv_mesh = pyvista.from_meshio(mesh)
#pv_mesh.cell_data["Subdomains"] = mesh.cell_data["gmsh:physical"][0]
#plotter = pyvista.Plotter()
#plotter.add_mesh(pv_mesh, scalars="Subdomains",show_edges=True)
#plotter.add_axes()
#plotter.view_xy() 
#plotter.show()

fe_type = FiniteElementType(
    cell_type=CellType.triangle,
    family=ElementFamily.P,
    basis_degree=1,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=6,
)

xi_qp, W_q = get_quadrature(fe_type=fe_type)
phi_qn, dphi_dxi_qnp = eval_basis_and_derivatives(fe_type=fe_type, xi_qp=xi_qp)
Q = get_quadrature(fe_type=fe_type)[0].shape[0] 

V=points.shape[0]
U=3 # num of solution componets
x_end = mesh_to_jax(vertices=points, cells=cells)
assembly_map_b = mesh_to_sparse_assembly_map(n_vertices=V, cells=cells)
# Extract shape constants needed for args
E = x_end.shape[0] # num elements


# Material properties

cell_domain_ids=mesh.cell_data["gmsh:physical"][0]
matrix_cells = cells[cell_domain_ids == 0]
fiber_cells = cells[cell_domain_ids == 1]

@partial(jax.jit, static_argnames=['num_quad_points'])
def get_heterogeneous_properties(cell_domain_ids, num_quad_points):

    E_m   = 4.76e3  
    nu_m  = 0.37
    G_m   = E_m / (2 * (1 + nu_m)) # Isotropic relation
    props_matrix = jnp.array([E_m, E_m, E_m, G_m, G_m, G_m, nu_m, nu_m, nu_m ])
    
    # Fiber
    E1,E2,E3= 276e3, 19.5e3, 19.5e3
    G12,G13,G23= 70e3, 70e3, 5.735e3
    v12,v13,v23 = 0.28, 0.28, 0.70
    props_fiber = jnp.array([E1,E2,E3,G12,G13,G23,v12,v13,v23])

    mask = cell_domain_ids[:, None, None] 
    
    # 4. Use jnp.where to select properties
    #    If mask == 1 (Fiber), pick props_fiber
    #    If mask == 0 (Matrix), pick props_matrix
    #    Result is automatically broadcasted to (NumElements, Q, 9)
    
    mat_params = jnp.where(
        mask == 1, 
        props_fiber, 
        props_matrix,
    )
    
    # Ensure shape is explicitly (NumElements, Q, 9) if broadcast doesn't fully expand Q
    # We essentially repeat the property vector Q times for each element
    mat_params = jnp.tile(mat_params, (1, num_quad_points, 1))

    return mat_params

material_params_eqm = get_heterogeneous_properties(cell_domain_ids, Q)


@jax.jit
def elastic_orthotropic(
    material_params_qm: jnp.ndarray
):
   # zero = jnp.zeros(shape=(material_params_qm.shape[0:1]))
    #E1,E2,E3,G12,G13,G23,v12,v13,v23
    E_1 = material_params_qm[..., 0]
    E_2 = material_params_qm[..., 1]
    E_3 = material_params_qm[..., 2]
    G_12 = material_params_qm[..., 3]
    G_13 = material_params_qm[..., 4]
    G_23 = material_params_qm[..., 5]
    nu_12 = material_params_qm[..., 6]
    nu_13 = material_params_qm[..., 7]
    nu_23 = material_params_qm[..., 8]


    zero = jnp.zeros_like(E_1)
    #    stack(..., axis=-1) creates a vector of size (..., 6)
    row0 = jnp.stack([1.0/E_1,    -nu_12/E_1, -nu_13/E_1, zero, zero, zero], axis=-1)
    row1 = jnp.stack([-nu_12/E_1, 1.0/E_2,    -nu_23/E_2, zero, zero, zero], axis=-1)
    row2 = jnp.stack([-nu_13/E_1, -nu_23/E_2, 1.0/E_3,    zero, zero, zero], axis=-1)
    row3 = jnp.stack([zero, zero, zero, 1.0/G_23, zero, zero], axis=-1)
    row4 = jnp.stack([zero, zero, zero, zero, 1.0/G_13, zero], axis=-1)
    row5 = jnp.stack([zero, zero, zero, zero, zero, 1.0/G_12], axis=-1)

    S_qss = jnp.stack([row0, row1, row2, row3, row4, row5], axis=-2)

    return jnp.linalg.inv(S_qss)


# Residual 3D model (2D SG)

def _element_residual_single_case(
    u_nd: jnp.ndarray,
    x_nd: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    phi_qn: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params_qm: jnp.ndarray,
    epsilon_bar: jnp.ndarray = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
):

    J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
    G_qpd = jnp.linalg.inv(J_qdp)
    det_J_q = jnp.linalg.det(J_qdp)
    # Physical derivatives dphi/dx (Actually dphi/dy and dphi/dz)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)

    d_dy = dphi_dx_qnd[..., 0] 
    d_dz = dphi_dx_qnd[..., 1]
    zeros = jnp.zeros_like(d_dy)

    # Build B-matrix rows matching Voigt notation: [xx, yy, zz, yz, xz, xy]
    # Each row component has shape (Q, NumNodes, 3) corresponding to DOFs [u, v, w]
    q=d_dy.shape[0]
    # Row 1: xx (d/dx) -> All zeros for 2D cross-section
    row1 = jnp.stack([zeros, zeros, zeros], axis=2).reshape(q,-1)
    
    # Row 2: yy (d/dy acting on v)
    row2 = jnp.stack([zeros, d_dy, zeros], axis=2).reshape(q,-1)
    
    # Row 3: zz (d/dz acting on w)
    row3 = jnp.stack([zeros, zeros, d_dz], axis=2).reshape(q,-1)
    
    # Row 4: yz (d/dz on v + d/dy on w)
    row4 = jnp.stack([zeros, d_dz, d_dy], axis=2).reshape(q,-1)
    
    # Row 5: xz (d/dz on u + d/dx on w) -> d/dx is 0
    row5 = jnp.stack([d_dz, zeros, zeros], axis=2).reshape(q,-1)
    
    # Row 6: xy (d/dy on u + d/dx on v) -> d/dx is 0
    row6 = jnp.stack([d_dy, zeros, zeros], axis=2).reshape(q,-1)

    # Stack to get B matrix: (Q, 6, NumNodes* 3)
    B_qse = jnp.stack([row1, row2, row3, row4, row5, row6], axis=0).transpose(1,0,2) # q_s_9

    # 3. Compute Strain and Stress
    # ---------------------------------------------------------
    # Strain = B * u
    # Contract B (Q, 6, N, 3) with u (N, 3) -> Strain (Q, 6)
    u_e=u_nd.flatten()

    eps_qs = jnp.einsum("qse, e -> qs", B_qse, u_e) 

   # stress_qs = elastic_orthotropic(eps_qs, material_params_qm)
    C_qss =  elastic_orthotropic(material_params_qm)
    
   # stress_qs= jnp.einsum("qss, qs -> qs", C_qss, eps_qs)

    
    stress_qs = jnp.einsum("qij, qj -> qi", C_qss, eps_qs+epsilon_bar)
    
    det_JxW_q = det_J_q * W_q
    
    # Force = sum_q ( detJw * sum_s ( B_qsni * stress_qs ) )
    R_e = jnp.einsum("qes, qs, q -> e", B_qse.transpose(0,2,1), stress_qs, det_JxW_q)
    f_vol_scalar = jnp.einsum("qn, q -> n", phi_qn, det_JxW_q)
    f_vol = jnp.stack([f_vol_scalar, f_vol_scalar, f_vol_scalar], axis=1)
    
    return R_e.reshape(u_nd.shape), f_vol


@jax.jit
def calculate_residual_batch_element_kernel_mixed(
    x_mixed_g: jnp.ndarray,      
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    phi_qn: jnp.ndarray,          
    W_q: jnp.ndarray,
    material_params_eqm: jnp.ndarray,
    assembly_map_b: object
):
    """
    Calculates the mixed global residual (Momentum + Volume Constraint).
    Input x_mixed_g has shape (N_dof + 3).
    """

    # --- 1. SPLIT GLOBAL VECTOR ---
    # u_g_flat: (N_dof,)
    # lamb_vec: (3,) -> [lam_x, lam_y, lam_z]
    u_g_flat = x_mixed_g[:-3]
    lamb_vec = x_mixed_g[-3:] 

    E = x_end.shape[0]
    u_end = transform_global_unraveled_to_element_node(assembly_map_b, u_g_flat, E)

    # --- 3. VMAP KERNEL (Returns Tuple) ---
    # We define the vmap to return BOTH Elastic Residual and Volume Vector
    @jax.vmap
    def residual_kernel_mixed(u_nd, x_nd, mat_params_qm):
        # Calls the 'mixed' single element kernel defined in the previous step
        return _element_residual_single_case(
            u_nd, x_nd, dphi_dxi_qnp, phi_qn, W_q, mat_params_qm
        )

    # R_end_elastic: (NumElements, NumNodes, 3)
    # Vol_end:       (NumElements, NumNodes, 3)
    R_end_elastic, Vol_end = residual_kernel_mixed(u_end, x_end, material_params_eqm)

    # We must assemble both vectors separately.
    
    # A. Global Elastic Force
    R_elastic_g = transform_element_node_to_global_unraveled_sum(
        assembly_map=assembly_map_b, 
        v_en=R_end_elastic
    )

    # B. Global Volume Vector (Integral of shape functions)
    Vol_vec_g = transform_element_node_to_global_unraveled_sum(
        assembly_map=assembly_map_b, 
        v_en=Vol_end
    )

    # --- 5. COMBINE FOR FINAL RESIDUAL ---
    
    # A. Elastic Residual: R_u = F_elastic + lambda * Vol_vector
    # We reshape to (Nodes, 3) to broadcast lambda [lx, ly, lz] correctly
    # Then flatten back to match global vector shape.
    R_elastic_reshaped = R_elastic_g.reshape(-1, 3)
    Vol_vec_reshaped = Vol_vec_g.reshape(-1, 3)
    
    # Apply Lagrange Multiplier force
    R_u_total = R_elastic_reshaped + (Vol_vec_reshaped * lamb_vec[None, :])
    R_u_flat = R_u_total.flatten() 

    # B. Constraint Residual: R_lambda = Integral(u) - 0
    # Numerically: dot product of u_nodal and Volume_vector

    # Result is (3,) -> [Total_Ux_Vol, Total_Uy_Vol, Total_Uz_Vol]
    R_lambda = jnp.sum(u_end * Vol_end, axis=(0, 1))

    return jnp.concatenate([R_u_flat, R_lambda])


jac_fn = jax.jacfwd(calculate_residual_batch_element_kernel_mixed)
u_mixed_0_g=jnp.zeros(shape=(V * U +3)) 
K = jac_fn(u_mixed_0_g, x_end, dphi_dxi_qnp, phi_qn, W_q, material_params_eqm, assembly_map_b)

@jax.jit
def calculate_residual_batch_element_kernel_vmap(
    u_end: jnp.ndarray,
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    phi_qn: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params_eqm: jnp.ndarray,
    assembly_map_b: object
):
    """
    Calculates the global residual matrix (GlobalNDOF, 6) for 6 unit strain cases.
    """
    
    # 1. Define the 6 unit strain cases (Voigt)
    unit_strains_66 = jnp.eye(6)

    # 2. Define function to process ONE element for ALL 6 unit strains
    def element_process_all_cases(u_nd, x_nd, mat_params_qm):
        
        # Inner VMAP: Iterate over the 6 unit strain vectors (row-by-row)
        # We vary ONLY epsilon_bar (arg 6)
        run_6_cases = jax.vmap(
            _element_residual_single_case, 
            in_axes=(None, None, None, None, None, None, 1)
        )

        # Output shape: (6, NumNodes, 3)
        R_6_nodes_3 = run_6_cases(
            u_nd, x_nd, dphi_dxi_qnp, phi_qn, W_q, mat_params_qm, unit_strains_66
        )[0]
        
        return R_6_nodes_3

    # 3. Outer VMAP: Iterate over the batch of elements
    # We vary u_end, x_end, and material_params_eqm over dim 0
    batch_processor = jax.vmap(
        element_process_all_cases, 
        in_axes=(0, 0, 0)
    )

    # 4. Execute
    # Output shape: (NumElements, ndof, 6)

    R_end_cases = batch_processor(u_end, x_end, material_params_eqm) # (E,6,N,3)
    R_cases_first = jnp.transpose(R_end_cases, (1, 0, 2, 3))

    # C. Define Assembly for ONE case
    def assemble_one_case(R_one_case_EN3):
        # Returns Global Vector (GlobalNDOF_unraveled, 3) 
        return transform_element_node_to_global_unraveled_sum(
            assembly_map=assembly_map_b, 
            v_en=R_one_case_EN3
        )
    
    # D. Vmap Assembly over the 6 cases
    # Output: (6, NumVertices, 3)
    R_global_cases = jax.vmap(assemble_one_case)(R_cases_first)
    
    # E. Flatten and Transpose to Final Matrix (GlobalNDOF, 6)
    # (6, V, 3) -> (6, V*3) -> (V*3, 6)
    R_elastic_matrix = R_global_cases.reshape(6, -1).T
    constraint_rhs = jnp.zeros((3, 6))
    
    # 2. Stack them at the bottom
    R_mixed_matrix = jnp.concatenate([R_elastic_matrix, constraint_rhs], axis=0)
    
    return R_mixed_matrix


E = x_end.shape[0]
u_0_g=jnp.zeros(shape=(V * U)) 
u_end = transform_global_unraveled_to_element_node(assembly_map_b, u_0_g, E)
Dhe = -calculate_residual_batch_element_kernel_vmap(u_end, x_end, dphi_dxi_qnp, phi_qn, W_q, material_params_eqm, assembly_map_b)


# Periodic mapping
def create_fem_periodic_map(points, left_idx, right_idx, bottom_idx, top_idx, tol=1e-5):
    """
    Creates a mapping array `dof_map` where:
    dof_map[slave_node_index] == master_node_index
    """
    num_nodes = len(points)
    # Initialize: every node maps to itself
    dof_map = np.arange(num_nodes)
    
    # Calculate Box Dimensions
    Lx = max_xy[0] - min_xy[0]
    Ly = max_xy[1] - min_xy[1]
    
    # Helper to map Slave set -> Master set based on geometric proximity
    def map_boundary(slaves, masters, shift_vec):
        slave_pts = points[slaves]
        master_pts = points[masters]
        
        # For each slave, find the master that is located at (slave_pos - shift_vec)
        # Using brute force broadcasting (OK for typical mesh sizes, use KDTree for massive ones)
        target_pos = slave_pts - shift_vec
        dists = np.linalg.norm(target_pos[:, None, :] - master_pts[None, :, :], axis=2)
        
        # Find closest master index
        nearest_idx = np.argmin(dists, axis=1)
        min_dists = np.min(dists, axis=1)
        
        if not np.all(min_dists < tol):
            raise ValueError("Geometric mismatch on periodic boundary")
            
        # Update the global map
        dof_map[slaves] = masters[nearest_idx]

    # 1. Map Right -> Left (Shift x by -Lx)
    map_boundary(right_idx, left_idx, np.array([Lx, 0.0]))
    
    # 2. Map Top -> Bottom (Shift y by -Ly)
    map_boundary(top_idx, bottom_idx, np.array([0.0, Ly]))
    
    # 3. Resolve Corners (Chain resolution)
    # If Top-Right maps to Top-Left, and Top-Left maps to Bottom-Left,
    # we need Top-Right to point directly to Bottom-Left.
    # Running this twice resolves the corner transitive dependency.
    for _ in range(2):
        dof_map = dof_map[dof_map]
        
    return jnp.array(dof_map)


def create_dof_map_expanded(node_periodic_map, ndof_per_node=3):

    num_nodes = len(node_periodic_map)
    
    # Get the master node index for every slave node
    master_nodes = node_periodic_map  
    
    dof_offsets = jnp.arange(ndof_per_node)
    
    # Full DOF map
    # (N, 1) * 3 + (1, 3) -> (N, 3) 
    master_dof_indices = master_nodes[:, None] * ndof_per_node + dof_offsets[None, :]
    
    # Flatten to (N_total_dofs, )
    return master_dof_indices.flatten()

# Define node sets
min_xy = np.min(points, axis=0)
max_xy = np.max(points, axis=0)

left_points = np.isclose(points[:, 0], min_xy[0], atol=1e-16).nonzero()[0]
right_points = np.isclose(points[:, 0], max_xy[0], atol=1e-16).nonzero()[0]
bottom_points = np.isclose(points[:, 1], min_xy[1], atol=1e-16).nonzero()[0]
top_points = np.isclose(points[:, 1], max_xy[1], atol=1e-16).nonzero()[0]

periodic_map=create_fem_periodic_map(points, left_points, right_points, bottom_points, top_points)
dof_map_disp = create_dof_map_expanded(periodic_map) 

N_total = K.shape[0]                  
N_disp  = len(dof_map_disp)           
lambda_indices = jnp.arange(N_disp, N_total) 
dof_map_full = jnp.concatenate([dof_map_disp, lambda_indices])

def apply_pbc_reduction(K_full, R_full, dof_map_full):
    """
    Condenses K (N, N) -> K_red (M, M)
    Condenses R (N,)   -> R_red (M,)
    Where M is the number of unique master DOFs.
    """
    
    # 1. Identify Unique Master DOFs
    unique_dofs = jnp.unique(dof_map_full)
    num_reduced = len(unique_dofs)
    
    # 2. Reduce Residual (R)

    R_accum = jnp.zeros_like(R_full).at[dof_map_full].add(R_full)
    R_reduced = R_accum[unique_dofs]
    
    # 3. Reduce Stiffness Matrix (K)
    # K_red = L.T @ K_full @ L
    # Logic: K_reduced[I, J] = Sum(K_full[i, j]) where map[i]=I and map[j]=J
    
    # Step A: Sum Columns (Condense j -> J)
    # We want to add column j into column map[j]
    K_col_reduced = jnp.zeros_like(K_full).at[:, dof_map_full].add(K_full)
    
    # Step B: Sum Rows (Condense i -> I)
    # We want to add row i into row map[i]
    # Note: We operate on K_col_reduced from Step A
    K_total_reduced_fullsize = jnp.zeros_like(K_col_reduced).at[dof_map_full, :].add(K_col_reduced)
    
    # Step C: Extract only the unique master rows/cols
    K_reduced = K_total_reduced_fullsize[jnp.ix_(unique_dofs, unique_dofs)]
    
    return K_reduced, R_reduced, unique_dofs
K_reduced, R_reduced, unique_dofs= apply_pbc_reduction(K, Dhe, dof_map_full)

V0_reduced = jnp.linalg.solve(K_reduced, R_reduced)

# 4. Compute D1
D1 = V0_reduced.T @ (-R_reduced)

def _element_D_bar(
    x_nd: jnp.ndarray,               # Nodal coords (NumNodes, 2 or 3)
    dphi_dxi_qnp: jnp.ndarray,       # Shape gradients (Q, NumNodes, Dim)
    W_q: jnp.ndarray,                # Quadrature weights (Q,)
    material_params_qm: jnp.ndarray  # Material params per quad point
):

    J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
    det_J_q = jnp.linalg.det(J_qdp)
    det_JxW_q = det_J_q * W_q
    C_qss = elastic_orthotropic(material_params_qm)
    D_bar_element = jnp.einsum("qij, q -> ij", C_qss, det_JxW_q)
    
    return D_bar_element
    
def compute_global_D_bar(
    x_end: jnp.ndarray,             # (NumCells, NumNodes, Dim) - Element coordinates
    dphi_dxi_qnp: jnp.ndarray,      # (Q, NumNodes, Dim) - Ref shape derivatives
    W_q: jnp.ndarray,               # (Q,) - Quadrature weights
    material_params_eqm: jnp.ndarray, # (NumCells, Q, NumParams)
):

    batch_D_bar_fn = jax.vmap(
        _element_D_bar, 
        in_axes=(0, None, None, 0) # Map x_end and materials, keep dphi and W constant
    )
    D_bar_elements = batch_D_bar_fn(x_end, dphi_dxi_qnp, W_q, material_params_eqm)
    D_bar_global = jnp.sum(D_bar_elements, axis=0)
    
    return D_bar_global

D_bar = compute_global_D_bar(x_end, dphi_dxi_qnp, W_q, material_params_eqm)
D_eff= D_bar+D1 # Effective Stiffness Matrix
print(D_eff)

print('Time taken', time.time()- start)

Com=np.linalg.inv(D_eff)  
E1,E2,E3=1/Com[0,0], 1/Com[1,1], 1/Com[2,2]
v12,v13,v23=-Com[0,1]/Com[0,0], -Com[0,2]/Com[0,0], -Com[1,2]/Com[1,1]
G23,G13,G12=1/Com[3,3], 1/Com[4,4], 1/Com[5,5]


print(E1,E2,E3,G12,G13,G23,v12,v13,v23)
