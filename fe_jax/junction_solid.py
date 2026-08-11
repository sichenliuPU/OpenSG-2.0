"""Three-dimensional solid-junction stiffness analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .beam import FloatArray, IntArray
from .junction import (
    JunctionConnectionPoint,
    JunctionStiffness,
    remove_rigid_roundoff,
)


@dataclass(frozen=True)
class SolidJunctionModel:
    """Solid mesh, materials, and interface nodes for one junction type."""

    nodes: FloatArray
    elements: tuple[IntArray, ...]
    material_ids: IntArray
    materials: dict[int, tuple[float, float]]
    connection_points: tuple[JunctionConnectionPoint, ...]
    interface_faces: tuple[IntArray, ...]


@dataclass(frozen=True)
class JunctionSolution:
    """Junction stiffness together with its full solid displacement map."""

    stiffness: JunctionStiffness
    displacement_recovery: FloatArray


_TET10_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def _tetrahedron_shape_derivatives(
    number_of_nodes: int, r: float, s: float, t: float
) -> FloatArray:
    barycentric = np.array([1.0 - r - s - t, r, s, t])
    derivatives = np.array(
        [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    if number_of_nodes == 4:
        return derivatives
    if number_of_nodes != 10:
        raise ValueError(
            f"Solid junction analysis does not support {number_of_nodes}-node elements."
        )
    result = np.empty((10, 3), dtype=float)
    result[:4] = (4.0 * barycentric - 1.0)[:, None] * derivatives
    for row, (i, j) in enumerate(_TET10_EDGES, start=4):
        result[row] = 4.0 * (
            derivatives[i] * barycentric[j] + barycentric[i] * derivatives[j]
        )
    return result


def _material_stiffness(engineering_e: float, poisson_nu: float) -> FloatArray:
    factor = engineering_e / ((1.0 + poisson_nu) * (1.0 - 2.0 * poisson_nu))
    return factor * np.array(
        [
            [1.0 - poisson_nu, poisson_nu, poisson_nu, 0.0, 0.0, 0.0],
            [poisson_nu, 1.0 - poisson_nu, poisson_nu, 0.0, 0.0, 0.0],
            [poisson_nu, poisson_nu, 1.0 - poisson_nu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson_nu), 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson_nu), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson_nu)],
        ]
    )


def _strain_matrix(derivatives: FloatArray) -> FloatArray:
    result = np.zeros((6, 3 * len(derivatives)), dtype=float)
    for node, (dx, dy, dz) in enumerate(derivatives):
        result[:, 3 * node : 3 * node + 3] = np.array(
            [
                [dx, 0.0, 0.0],
                [0.0, dy, 0.0],
                [0.0, 0.0, dz],
                [0.0, dz, dy],
                [dz, 0.0, dx],
                [dy, dx, 0.0],
            ]
        )
    return result


def assemble_solid_stiffness(model: SolidJunctionModel) -> sparse.csc_matrix:
    """Assemble the sparse stiffness of a tetrahedral junction mesh."""

    a = 0.5854101966249685
    b = 0.1381966011250105
    quadratic_quadrature = (
        (b, b, b, 1.0 / 24.0),
        (a, b, b, 1.0 / 24.0),
        (b, a, b, 1.0 / 24.0),
        (b, b, a, 1.0 / 24.0),
    )
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    material_matrices = {
        identifier: _material_stiffness(*properties)
        for identifier, properties in model.materials.items()
    }
    for element_index, element in enumerate(model.elements):
        element = np.asarray(element, dtype=np.int64)
        number_of_nodes = len(element)
        if number_of_nodes == 4:
            quadrature = ((0.25, 0.25, 0.25, 1.0 / 6.0),)
        elif number_of_nodes == 10:
            quadrature = quadratic_quadrature
        else:
            raise ValueError(
                f"Solid element {element_index + 1} has {number_of_nodes} nodes; "
                "supported tetrahedra have 4 or 10 nodes."
            )
        material_id = int(model.material_ids[element_index])
        if material_id not in material_matrices:
            raise ValueError(f"Element references undefined material {material_id}.")
        coordinates = model.nodes[element]
        element_stiffness = np.zeros((3 * number_of_nodes, 3 * number_of_nodes), dtype=float)
        for r, s, t, weight in quadrature:
            natural = _tetrahedron_shape_derivatives(number_of_nodes, r, s, t)
            jacobian = coordinates.T @ natural
            determinant = float(np.linalg.det(jacobian))
            if determinant <= 0.0:
                raise ValueError(
                    f"TET{number_of_nodes} element {element_index + 1} has a "
                    "non-positive Jacobian."
                )
            derivatives = natural @ np.linalg.inv(jacobian)
            strain = _strain_matrix(derivatives)
            element_stiffness += (
                strain.T @ material_matrices[material_id] @ strain
            ) * determinant * weight
        dofs = np.concatenate(
            [3 * int(node) + np.arange(3, dtype=np.int64) for node in element]
        )
        row_indices, column_indices = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(row_indices.reshape(-1).tolist())
        columns.extend(column_indices.reshape(-1).tolist())
        values.extend(element_stiffness.reshape(-1).tolist())
    size = 3 * len(model.nodes)
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=(size, size)).tocsc()
    return 0.5 * (matrix + matrix.T)


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _tri6_shape_functions(r: float, s: float) -> FloatArray:
    l1, l2, l3 = 1.0 - r - s, r, s
    return np.array(
        [l1 * (2.0 * l1 - 1.0), l2 * (2.0 * l2 - 1.0), l3 * (2.0 * l3 - 1.0),
         4.0 * l1 * l2, 4.0 * l2 * l3, 4.0 * l3 * l1]
    )


def _tri6_shape_derivatives(r: float, s: float) -> FloatArray:
    barycentric = np.array([1.0 - r - s, r, s])
    derivatives = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    result = np.empty((6, 2), dtype=float)
    result[:3] = (4.0 * barycentric - 1.0)[:, None] * derivatives
    for row, (i, j) in enumerate(((0, 1), (1, 2), (2, 0)), start=3):
        result[row] = 4.0 * (
            derivatives[i] * barycentric[j] + barycentric[i] * derivatives[j]
        )
    return result


def _triangle_interpolation(
    number_of_nodes: int, r: float, s: float
) -> tuple[FloatArray, FloatArray]:
    if number_of_nodes == 3:
        return (
            np.array([1.0 - r - s, r, s]),
            np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
        )
    if number_of_nodes == 6:
        return _tri6_shape_functions(r, s), _tri6_shape_derivatives(r, s)
    raise ValueError(
        f"Solid junction interfaces do not support {number_of_nodes}-node faces."
    )


def _face_mass_matrix(nodes: FloatArray, faces: IntArray) -> tuple[IntArray, FloatArray]:
    """Return interface nodes and the consistent triangular surface mass."""

    face_nodes = np.unique(np.asarray(faces, dtype=np.int64).reshape(-1))
    local_index = {int(node): index for index, node in enumerate(face_nodes)}
    mass = np.zeros((len(face_nodes), len(face_nodes)), dtype=float)
    number_of_nodes = faces.shape[1]
    if number_of_nodes == 3:
        quadrature = (
            (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        )
    elif number_of_nodes == 6:
        quadrature = [(1.0 / 3.0, 1.0 / 3.0, 0.225 / 2.0)]
        for a, b, weight in (
            (0.470142064105115, 0.059715871789770, 0.132394152788506 / 2.0),
            (0.101286507323456, 0.797426985353087, 0.125939180544827 / 2.0),
        ):
            quadrature.extend(((a, a, weight), (a, b, weight), (b, a, weight)))
    else:
        raise ValueError(
            "Solid junction interfaces require TRI3 or TRI6 faces; "
            f"received {number_of_nodes} nodes."
        )
    for face in faces:
        coordinates = nodes[face]
        indices = np.asarray([local_index[int(node)] for node in face], dtype=np.int64)
        for r, s, weight in quadrature:
            shape, derivatives = _triangle_interpolation(number_of_nodes, r, s)
            tangents = coordinates.T @ derivatives
            jacobian = np.linalg.norm(np.cross(tangents[:, 0], tangents[:, 1]))
            mass[np.ix_(indices, indices)] += (
                np.outer(shape, shape) * jacobian * weight
            )
    return face_nodes, 0.5 * (mass + mass.T)


def interface_extraction(model: SolidJunctionModel) -> FloatArray:
    """Map solid-interface displacements to connection-point variables."""

    number_of_solid_dofs = 3 * len(model.nodes)
    extraction = np.zeros(
        (6 * len(model.connection_points), number_of_solid_dofs), dtype=float
    )
    for index, (connection_point, faces) in enumerate(
        zip(model.connection_points, model.interface_faces, strict=True)
    ):
        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] not in (3, 6):
            raise ValueError(
                f"Connection point {connection_point.identifier} requires a TRI3 or "
                "TRI6 interface."
            )
        node_ids, scalar_mass = _face_mass_matrix(model.nodes, faces)
        modes = np.zeros((3 * len(node_ids), 6), dtype=float)
        for local_node, node in enumerate(node_ids):
            offset = model.nodes[node] - connection_point.origin
            modes[3 * local_node : 3 * local_node + 3, :3] = connection_point.frame.T
            modes[3 * local_node : 3 * local_node + 3, 3:] = (
                -_skew(offset) @ connection_point.frame.T
            )
        mass = np.kron(scalar_mass, np.eye(3))
        weighted_modes = modes.T @ mass
        local_extraction = np.linalg.solve(weighted_modes @ modes, weighted_modes)
        columns = np.concatenate(
            [3 * int(node) + np.arange(3, dtype=np.int64) for node in node_ids]
        )
        extraction[6 * index : 6 * index + 6, columns] = local_extraction
        error = np.linalg.norm(local_extraction @ modes - np.eye(6), ord=np.inf)
        if error > 1.0e-9:
            raise ValueError(
                f"Connection point {connection_point.identifier} interface extraction "
                f"error is {error:g}."
            )
    return extraction


def analyze_junction(model: SolidJunctionModel) -> JunctionSolution:
    """Calculate junction stiffness and the solid displacement recovery map."""

    stiffness = assemble_solid_stiffness(model)
    extraction = interface_extraction(model)
    number_of_solid_dofs = stiffness.shape[0]
    number_of_connection_dofs = extraction.shape[0]
    system = sparse.bmat(
        [[stiffness, -sparse.csr_matrix(extraction.T)],
         [sparse.csr_matrix(extraction), None]],
        format="csc",
    )
    right_hand_side = np.vstack(
        (np.zeros((number_of_solid_dofs, number_of_connection_dofs)),
         np.eye(number_of_connection_dofs))
    )
    solution = splu(system).solve(right_hand_side)
    matrix = solution[number_of_solid_dofs:]
    data = JunctionStiffness(model.connection_points, 0.5 * (matrix + matrix.T))
    return JunctionSolution(
        stiffness=remove_rigid_roundoff(data),
        displacement_recovery=solution[:number_of_solid_dofs],
    )


def calculate_junction_stiffness(model: SolidJunctionModel) -> JunctionStiffness:
    """Calculate generalized junction stiffness from a 3D solid model."""

    return analyze_junction(model).stiffness
