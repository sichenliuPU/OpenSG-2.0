"""Simple-cubic C3D20 verification support, separate from solver workflows."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .beam import FloatArray, IntArray


_CORNER_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)

# Abaqus C3D20 node order in natural coordinates.
_NATURAL_NODES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    [0, -1, -1], [1, 0, -1], [0, 1, -1], [-1, 0, -1],
    [0, -1, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1],
    [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
], dtype=float)


@dataclass(frozen=True)
class C3D20Mesh:
    nodes: FloatArray
    elements: IntArray


@dataclass(frozen=True)
class C3D20Recovery:
    mesh: C3D20Mesh
    displacement_map: FloatArray
    junction_stiffness: FloatArray
    element_volumes: FloatArray


def _isotropic_stiffness(young: float, poisson: float) -> FloatArray:
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    matrix = np.zeros((6, 6), dtype=float)
    matrix[:3, :3] = lam
    matrix[np.arange(3), np.arange(3)] += 2.0 * mu
    matrix[3:, 3:] = np.eye(3) * mu
    return matrix


def c3d20_shape_derivatives(xi: float, eta: float, zeta: float) -> FloatArray:
    """Return natural-coordinate derivatives in Abaqus C3D20 node order."""

    derivatives = np.empty((20, 3), dtype=float)
    for node, (sx, sy, sz) in enumerate(_CORNER_SIGNS):
        ax, ay, az = 1.0 + sx * xi, 1.0 + sy * eta, 1.0 + sz * zeta
        linear = sx * xi + sy * eta + sz * zeta - 2.0
        derivatives[node] = 0.125 * np.array([
            sx * ay * az * (linear + ax),
            sy * ax * az * (linear + ay),
            sz * ax * ay * (linear + az),
        ])
    for edge, (sy, sz) in zip(
        [8, 10, 12, 14],
        [(-1, -1), (1, -1), (-1, 1), (1, 1)],
        strict=True,
    ):
        derivatives[edge] = 0.25 * np.array([
            -2.0 * xi * (1.0 + sy * eta) * (1.0 + sz * zeta),
            sy * (1.0 - xi**2) * (1.0 + sz * zeta),
            sz * (1.0 - xi**2) * (1.0 + sy * eta),
        ])
    for edge, (sx, sz) in zip(
        [9, 11, 13, 15],
        [(1, -1), (-1, -1), (1, 1), (-1, 1)],
        strict=True,
    ):
        derivatives[edge] = 0.25 * np.array([
            sx * (1.0 - eta**2) * (1.0 + sz * zeta),
            -2.0 * eta * (1.0 + sx * xi) * (1.0 + sz * zeta),
            sz * (1.0 - eta**2) * (1.0 + sx * xi),
        ])
    for edge, (sx, sy) in enumerate(
        [(-1, -1), (1, -1), (1, 1), (-1, 1)], start=16
    ):
        derivatives[edge] = 0.25 * np.array([
            sx * (1.0 - zeta**2) * (1.0 + sy * eta),
            sy * (1.0 - zeta**2) * (1.0 + sx * xi),
            -2.0 * zeta * (1.0 + sx * xi) * (1.0 + sy * eta),
        ])
    return derivatives


def _strain_matrix(
    coordinates: FloatArray, natural: FloatArray
) -> tuple[FloatArray, float]:
    derivatives = c3d20_shape_derivatives(*natural)
    jacobian = coordinates.T @ derivatives
    determinant = float(np.linalg.det(jacobian))
    if determinant <= 0.0:
        raise ValueError("Invalid C3D20 element Jacobian.")
    global_derivatives = derivatives @ np.linalg.inv(jacobian)
    matrix = np.zeros((6, 60), dtype=float)
    for node, (dx, dy, dz) in enumerate(global_derivatives):
        columns = slice(3 * node, 3 * node + 3)
        matrix[:, columns] = np.array([
            [dx, 0.0, 0.0],
            [0.0, dy, 0.0],
            [0.0, 0.0, dz],
            [0.0, dz, dy],
            [dz, 0.0, dx],
            [dy, dx, 0.0],
        ])
    return matrix, determinant


def build_simple_cubic_c3d20_mesh(
    side: float, stub_length: float, elements_per_side: int = 4
) -> C3D20Mesh:
    """Build three conforming orthogonal square prisms from C3D20 elements."""

    if elements_per_side < 2 or elements_per_side % 2:
        raise ValueError("elements_per_side must be an even integer of at least two.")
    element_size = side / elements_per_side
    stub_elements = int(round(stub_length / element_size))
    if not np.isclose(stub_elements * element_size, stub_length):
        raise ValueError("stub_length must be an integer multiple of element size.")
    total = elements_per_side + 2 * stub_elements
    extent = side / 2.0 + stub_length
    lattice_coordinates = np.linspace(-extent, extent, 2 * total + 1)

    element_lattice_nodes = []
    used = set()
    for i, j, k in itertools.product(range(total), repeat=3):
        center = -extent + element_size * (np.array([i, j, k]) + 0.5)
        if np.count_nonzero(np.abs(center) < side / 2.0 + 1.0e-12) < 2:
            continue
        i0, j0, k0 = 2 * i, 2 * j, 2 * k
        local = [
            (i0, j0, k0), (i0 + 2, j0, k0),
            (i0 + 2, j0 + 2, k0), (i0, j0 + 2, k0),
            (i0, j0, k0 + 2), (i0 + 2, j0, k0 + 2),
            (i0 + 2, j0 + 2, k0 + 2), (i0, j0 + 2, k0 + 2),
            (i0 + 1, j0, k0), (i0 + 2, j0 + 1, k0),
            (i0 + 1, j0 + 2, k0), (i0, j0 + 1, k0),
            (i0 + 1, j0, k0 + 2), (i0 + 2, j0 + 1, k0 + 2),
            (i0 + 1, j0 + 2, k0 + 2), (i0, j0 + 1, k0 + 2),
            (i0, j0, k0 + 1), (i0 + 2, j0, k0 + 1),
            (i0 + 2, j0 + 2, k0 + 1), (i0, j0 + 2, k0 + 1),
        ]
        element_lattice_nodes.append(local)
        used.update(local)
    lattice_nodes = sorted(used)
    lookup = {node: index for index, node in enumerate(lattice_nodes)}
    nodes = np.asarray([
        [lattice_coordinates[i], lattice_coordinates[j], lattice_coordinates[k]]
        for i, j, k in lattice_nodes
    ])
    elements = np.asarray([
        [lookup[node] for node in element] for element in element_lattice_nodes
    ], dtype=np.int64)
    return C3D20Mesh(nodes, elements)


def _quad8_interpolation(u: float, v: float) -> tuple[FloatArray, FloatArray]:
    signs = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    values = np.empty(8)
    derivatives = np.empty((8, 2))
    for node, (su, sv) in enumerate(signs):
        a, b = 1.0 + su * u, 1.0 + sv * v
        linear = su * u + sv * v - 1.0
        values[node] = 0.25 * a * b * linear
        derivatives[node] = 0.25 * np.array([
            su * b * (linear + a), sv * a * (linear + b)
        ])
    values[4:] = [
        0.5 * (1.0 - u * u) * (1.0 - v),
        0.5 * (1.0 + u) * (1.0 - v * v),
        0.5 * (1.0 - u * u) * (1.0 + v),
        0.5 * (1.0 - u) * (1.0 - v * v),
    ]
    derivatives[4:] = [
        [-u * (1.0 - v), -0.5 * (1.0 - u * u)],
        [0.5 * (1.0 - v * v), -(1.0 + u) * v],
        [-u * (1.0 + v), 0.5 * (1.0 - u * u)],
        [-0.5 * (1.0 - v * v), -(1.0 - u) * v],
    ]
    return values, derivatives


def _interface_faces(mesh: C3D20Mesh, branch: int) -> IntArray:
    axis = branch // 2
    sign = 1 if branch % 2 == 0 else -1
    local_faces = {
        (0, -1): [0, 4, 7, 3, 16, 15, 19, 11],
        (0, 1): [1, 2, 6, 5, 9, 18, 13, 17],
        (1, -1): [0, 1, 5, 4, 8, 17, 12, 16],
        (1, 1): [3, 7, 6, 2, 19, 14, 18, 10],
        (2, -1): [0, 3, 2, 1, 11, 10, 9, 8],
        (2, 1): [4, 5, 6, 7, 12, 13, 14, 15],
    }
    extent = float(np.max(np.abs(mesh.nodes[:, axis])))
    tolerance = max(1.0, extent) * 1.0e-11
    faces = []
    local = local_faces[(axis, sign)]
    for element in mesh.elements:
        face = element[local]
        if np.all(np.abs(mesh.nodes[face, axis] - sign * extent) <= tolerance):
            faces.append(face)
    if not faces:
        raise ValueError(f"No C3D20 faces found for connection {branch + 1}.")
    return np.asarray(faces, dtype=np.int64)


def _interface_extraction(
    mesh: C3D20Mesh,
    frames: FloatArray,
    side: float,
    stub_length: float,
) -> FloatArray:
    extraction = np.zeros((36, 3 * len(mesh.nodes)))
    gauss, weights = np.polynomial.legendre.leggauss(3)
    distance = side / 2.0 + stub_length
    for branch, frame in enumerate(frames):
        faces = _interface_faces(mesh, branch)
        face_nodes = np.unique(faces.ravel())
        lookup = {int(node): index for index, node in enumerate(face_nodes)}
        scalar_mass = np.zeros((len(face_nodes), len(face_nodes)))
        for face in faces:
            coordinates = mesh.nodes[face]
            indices = np.asarray([lookup[int(node)] for node in face])
            for i, j in itertools.product(range(3), repeat=2):
                shape, derivatives = _quad8_interpolation(gauss[i], gauss[j])
                tangents = coordinates.T @ derivatives
                jacobian = np.linalg.norm(np.cross(tangents[:, 0], tangents[:, 1]))
                scalar_mass[np.ix_(indices, indices)] += (
                    np.outer(shape, shape) * jacobian * weights[i] * weights[j]
                )
        center = distance * frame[0]
        modes = np.zeros((3 * len(face_nodes), 6))
        for local_node, global_node in enumerate(face_nodes):
            local_point = frame @ (mesh.nodes[global_node] - center)
            skew = np.array([
                [0.0, -local_point[2], local_point[1]],
                [local_point[2], 0.0, -local_point[0]],
                [-local_point[1], local_point[0], 0.0],
            ])
            modes[3 * local_node:3 * local_node + 3] = (
                frame.T @ np.hstack((np.eye(3), -skew))
            )
        mass = np.kron(scalar_mass, np.eye(3))
        weighted = modes.T @ mass
        local_extraction = np.linalg.solve(weighted @ modes, weighted)
        columns = np.concatenate([
            np.arange(3 * node, 3 * node + 3) for node in face_nodes
        ])
        extraction[6 * branch:6 * branch + 6, columns] = local_extraction
        error = np.linalg.norm(local_extraction @ modes - np.eye(6), ord=np.inf)
        if error > 1.0e-10:
            raise ValueError(f"Invalid C3D20 interface rigid-mode error: {error:g}.")
    return extraction


def build_simple_cubic_c3d20_recovery(
    frames: FloatArray,
    side: float,
    stub_length: float,
    young: float,
    poisson: float,
    elements_per_side: int = 4,
) -> C3D20Recovery:
    """Build the junction displacement recovery operator on a C3D20 mesh."""

    mesh = build_simple_cubic_c3d20_mesh(side, stub_length, elements_per_side)
    material = _isotropic_stiffness(young, poisson)
    gauss, weights = np.polynomial.legendre.leggauss(3)
    row_parts, column_parts, value_parts = [], [], []
    volumes = np.zeros(len(mesh.elements))
    for element_index, element in enumerate(mesh.elements):
        coordinates = mesh.nodes[element]
        element_stiffness = np.zeros((60, 60), dtype=float)
        for ix, iy, iz in itertools.product(range(3), repeat=3):
            natural = np.array([gauss[ix], gauss[iy], gauss[iz]])
            strain, determinant = _strain_matrix(coordinates, natural)
            factor = weights[ix] * weights[iy] * weights[iz] * determinant
            element_stiffness += strain.T @ material @ strain * factor
            volumes[element_index] += factor
        dofs = np.concatenate([
            np.arange(3 * node, 3 * node + 3) for node in element
        ])
        row_parts.append(np.repeat(dofs, 60))
        column_parts.append(np.tile(dofs, 60))
        value_parts.append(element_stiffness.ravel())
    number_of_dofs = 3 * len(mesh.nodes)
    stiffness = sparse.coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(number_of_dofs, number_of_dofs),
    ).tocsr()
    extraction = _interface_extraction(mesh, np.asarray(frames), side, stub_length)
    system = sparse.bmat([
        [stiffness / young, -sparse.csr_matrix(extraction.T)],
        [sparse.csr_matrix(extraction), None],
    ], format="csc")
    right_hand_side = np.vstack((
        np.zeros((number_of_dofs, 36)), np.eye(36)
    ))
    solution = sparse_linalg.splu(system).solve(right_hand_side)
    displacement_map = solution[:number_of_dofs]
    constraint_error = np.linalg.norm(
        extraction @ displacement_map - np.eye(36), ord=np.inf
    )
    if constraint_error > 1.0e-9:
        raise ValueError(f"Invalid C3D20 interface constraint: {constraint_error:g}.")
    junction_stiffness = displacement_map.T @ (stiffness @ displacement_map)
    junction_stiffness = 0.5 * (junction_stiffness + junction_stiffness.T)
    return C3D20Recovery(
        mesh, displacement_map, junction_stiffness, volumes
    )


def recover_c3d20_centerline_stress(
    recovery: C3D20Recovery,
    generalized_displacement: FloatArray,
    young: float,
    poisson: float,
    x_min: float = 0.0,
    x_max: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return stress at actual C3D20 nodes on the positive x centerline."""

    displacement = (
        recovery.displacement_map
        @ np.asarray(generalized_displacement, dtype=float).reshape(36)
    ).reshape(-1, 3)
    nodes = recovery.mesh.nodes
    upper = float(nodes[:, 0].max()) if x_max is None else float(x_max)
    tolerance = max(1.0, np.max(np.abs(nodes))) * 1.0e-10
    selected = np.flatnonzero(
        (nodes[:, 0] >= x_min - tolerance)
        & (nodes[:, 0] <= upper + tolerance)
        & (np.abs(nodes[:, 1]) <= tolerance)
        & (np.abs(nodes[:, 2]) <= tolerance)
    )
    selected = selected[np.argsort(nodes[selected, 0])]
    lookup = {int(node): index for index, node in enumerate(selected)}
    accumulated = np.zeros((len(selected), 6))
    accumulated_volume = np.zeros(len(selected))
    material = _isotropic_stiffness(young, poisson)
    for element_index, element in enumerate(recovery.mesh.elements):
        relevant = [
            (local_node, lookup[int(node)])
            for local_node, node in enumerate(element)
            if int(node) in lookup
        ]
        if not relevant:
            continue
        coordinates = nodes[element]
        element_displacement = displacement[element].reshape(60)
        for local_node, output_index in relevant:
            strain, _ = _strain_matrix(
                coordinates, _NATURAL_NODES[local_node]
            )
            volume = recovery.element_volumes[element_index]
            accumulated[output_index] += (
                volume * material @ (strain @ element_displacement)
            )
            accumulated_volume[output_index] += volume
    if np.any(accumulated_volume == 0.0):
        raise ValueError("A selected C3D20 centerline node has no stress value.")
    return nodes[selected, 0].copy(), accumulated / accumulated_volume[:, None]
