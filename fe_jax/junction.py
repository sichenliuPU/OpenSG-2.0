"""Hybrid beam--junction data and assembly operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .beam import (
    FloatArray,
    HomogenizationTerms,
    IntArray,
    macro_displacement_matrix,
)


@dataclass(frozen=True)
class JunctionConnectionPoint:
    """Position and local frame of one junction connection."""

    identifier: int
    origin: FloatArray
    frame: FloatArray


@dataclass(frozen=True)
class JunctionStiffness:
    """Reusable stiffness and connection metadata for one junction type."""

    connection_points: tuple[JunctionConnectionPoint, ...]
    matrix: FloatArray


@dataclass(frozen=True)
class JunctionType:
    """Source file associated with a junction type."""

    identifier: int
    number_of_connection_points: int
    source: Path


@dataclass(frozen=True)
class JunctionInstance:
    """Position and orientation of one junction in the structure gene."""

    identifier: int
    junction_type_id: int
    origin: FloatArray
    frame: FloatArray


@dataclass(frozen=True)
class JunctionConnection:
    """Map one junction connection to one beam endpoint."""

    junction_id: int
    connection_point_id: int
    element_id: int
    endpoint: int
    image_shift: FloatArray


def _data_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].split("!", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def validate_junction_stiffness(data: JunctionStiffness) -> None:
    """Validate dimensions, connection numbering, and matrix symmetry."""

    number_of_connection_points = len(data.connection_points)
    expected = 6 * number_of_connection_points
    if data.matrix.shape != (expected, expected):
        raise ValueError(
            f"Junction stiffness must have shape {(expected, expected)}, "
            f"received {data.matrix.shape}."
        )
    identifiers = [point.identifier for point in data.connection_points]
    if identifiers != list(range(1, number_of_connection_points + 1)):
        raise ValueError(
            "Junction connection points must be numbered consecutively from one."
        )
    for point in data.connection_points:
        if point.origin.shape != (3,) or point.frame.shape != (3, 3):
            raise ValueError(
                f"Junction connection point {point.identifier} has invalid geometry metadata."
            )
        if not np.allclose(point.frame @ point.frame.T, np.eye(3), atol=1.0e-9):
            raise ValueError(
                f"Junction connection point {point.identifier} frame is not orthonormal."
            )
        if np.linalg.det(point.frame) <= 0.0:
            raise ValueError(
                f"Junction connection point {point.identifier} frame is not right-handed."
            )
    scale = max(float(np.linalg.norm(data.matrix)), 1.0)
    error = float(np.linalg.norm(data.matrix - data.matrix.T) / scale)
    if error > 1.0e-9:
        raise ValueError(f"Junction stiffness is not symmetric; relative error={error:g}.")
    if np.linalg.norm(data.matrix) > 0.0:
        eigenvalues = np.linalg.eigvalsh(data.matrix)
        tolerance = 1.0e-9 * max(float(np.max(np.abs(eigenvalues))), 1.0)
        if np.min(eigenvalues) < -tolerance:
            raise ValueError("Junction stiffness contains a negative deformational mode.")
        expected_rank = max(expected - 6, 0)
        rank = int(np.count_nonzero(eigenvalues > tolerance))
        if rank != expected_rank:
            raise ValueError(
                f"Junction stiffness rank is {rank}; expected {expected_rank}."
            )
        rigid = rigid_connection_modes(data.connection_points)
        denominator = np.linalg.norm(data.matrix) * max(np.linalg.norm(rigid), 1.0)
        residual = float(np.linalg.norm(data.matrix @ rigid) / denominator)
        if residual > 1.0e-8:
            raise ValueError(f"Junction rigid-mode residual is {residual:g}.")


def read_junction_stiffness(path: str | Path) -> JunctionStiffness:
    """Read an OpenSG junction-stiffness file."""

    path = Path(path)
    lines = _data_lines(path)
    if not lines:
        raise ValueError(f"Junction stiffness file is empty: {path}")
    header = lines[0].split()
    if len(header) != 2 or int(header[0]) != 1:
        raise ValueError(
            "The junction-stiffness header must be: version nconnectionpoint."
        )
    number_of_connection_points = int(header[1])
    if len(lines) < 1 + number_of_connection_points + 6 * number_of_connection_points:
        raise ValueError(f"Junction stiffness file is incomplete: {path}")

    connection_points: list[JunctionConnectionPoint] = []
    cursor = 1
    for expected_identifier in range(1, number_of_connection_points + 1):
        values = lines[cursor].split()
        cursor += 1
        if len(values) != 13:
            raise ValueError("A junction connection-point record requires 13 values.")
        identifier = int(values[0])
        if identifier != expected_identifier:
            raise ValueError(
                "Junction connection points must be ordered consecutively from one."
            )
        numbers = np.asarray(values[1:], dtype=float)
        connection_points.append(
            JunctionConnectionPoint(
                identifier=identifier,
                origin=numbers[:3],
                frame=numbers[3:].reshape(3, 3),
            )
        )

    size = 6 * number_of_connection_points
    matrix = np.asarray(
        [[float(value) for value in lines[cursor + row].split()] for row in range(size)],
        dtype=float,
    )
    data = JunctionStiffness(tuple(connection_points), matrix)
    validate_junction_stiffness(data)
    return JunctionStiffness(data.connection_points, 0.5 * (matrix + matrix.T))


def write_junction_stiffness(path: str | Path, data: JunctionStiffness) -> None:
    """Write a reusable OpenSG junction-stiffness file."""

    validate_junction_stiffness(data)
    path = Path(path)
    lines = [f"1 {len(data.connection_points)}"]
    for point in data.connection_points:
        values = np.concatenate((point.origin, point.frame.reshape(-1)))
        lines.append(
            f"{point.identifier} " + " ".join(f"{value:.16e}" for value in values)
        )
    lines.extend(
        " ".join(f"{value:.16e}" for value in row) for row in data.matrix
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def connection_matrices(
    number_of_dofs: int,
    nodes: FloatArray,
    elements: dict[int, IntArray],
    instance: JunctionInstance,
    connections: list[JunctionConnection],
    stiffness: JunctionStiffness,
) -> tuple[FloatArray, FloatArray]:
    """Build junction maps from beam variables and macroscopic strain."""

    if len(connections) != len(stiffness.connection_points):
        raise ValueError(
            f"Junction {instance.identifier} has {len(connections)} connection records; "
            f"its type requires {len(stiffness.connection_points)} connection points."
        )
    by_identifier = {
        connection.connection_point_id: connection for connection in connections
    }
    expected_connection_points = {
        point.identifier for point in stiffness.connection_points
    }
    if set(by_identifier) != expected_connection_points:
        raise ValueError(
            f"Junction {instance.identifier} has missing or duplicate connection points."
        )

    b_v = np.zeros((6 * len(stiffness.connection_points), number_of_dofs), dtype=float)
    b_epsilon = np.zeros((6 * len(stiffness.connection_points), 6), dtype=float)
    for connection_point in stiffness.connection_points:
        connection = by_identifier[connection_point.identifier]
        node_ids = elements.get(connection.element_id)
        if node_ids is None:
            raise ValueError(
                f"Junction {instance.identifier} references undefined element "
                f"{connection.element_id}."
            )
        if connection.endpoint not in (1, 2):
            raise ValueError("A junction endpoint must be one or two.")
        node = int(node_ids[connection.endpoint - 1])
        rows = slice(
            6 * (connection_point.identifier - 1),
            6 * connection_point.identifier,
        )
        columns = slice(6 * node, 6 * node + 6)
        global_connection_frame = connection_point.frame @ instance.frame
        b_v[rows.start : rows.start + 3, columns.start : columns.start + 3] = (
            global_connection_frame
        )
        b_v[rows.start + 3 : rows.stop, columns.start + 3 : columns.stop] = (
            global_connection_frame
        )

        unwrapped_position = nodes[node] + connection.image_shift
        # The macroscopic part of the revised SG-coordinate connection field
        # is A_epsilon * epsilon.  Express its translational rows in the
        # connection-local frame; the three rotational rows remain zero.
        b_epsilon[rows.start : rows.start + 3] = (
            global_connection_frame
            @ macro_displacement_matrix(unwrapped_position)[:3]
        )


        expected_position = instance.origin + instance.frame.T @ connection_point.origin
        if not np.allclose(unwrapped_position, expected_position, rtol=1.0e-8, atol=1.0e-9):
            raise ValueError(
                f"Junction {instance.identifier}, connection "
                f"{connection_point.identifier} does not coincide with its beam "
                "endpoint after applying the shift."
            )
    return b_v, b_epsilon


def add_junction_terms(
    terms: HomogenizationTerms,
    b_v: FloatArray,
    b_epsilon: FloatArray,
    stiffness: JunctionStiffness,
) -> None:
    """Add one complete physical junction to homogenization arrays."""

    matrix = stiffness.matrix
    junction_e=b_v.T @ matrix @ b_v
    if sparse.issparse(terms.e):
        terms.e += sparse.csr_matrix(junction_e)
    else:
        terms.e += junction_e
    terms.d_h_epsilon += b_v.T @ matrix @ b_epsilon
    terms.d_epsilon_epsilon += b_epsilon.T @ matrix @ b_epsilon
    terms.e = 0.5 * (terms.e + terms.e.T)
    terms.d_epsilon_epsilon[:] = 0.5 * (
        terms.d_epsilon_epsilon + terms.d_epsilon_epsilon.T
    )


def rigid_connection_modes(
    connection_points: tuple[JunctionConnectionPoint, ...]
) -> FloatArray:
    """Return connection-point coordinates of six global rigid motions."""

    result = np.zeros((6 * len(connection_points), 6), dtype=float)
    identity = np.eye(3)
    for index, connection in enumerate(connection_points):
        rows = slice(6 * index, 6 * index + 6)
        result[rows.start : rows.start + 3, :3] = connection.frame
        for axis in range(3):
            omega = identity[axis]
            result[rows.start : rows.start + 3, 3 + axis] = (
                connection.frame @ np.cross(omega, connection.origin)
            )
            result[rows.start + 3 : rows.stop, 3 + axis] = connection.frame @ omega
    return result


def remove_rigid_roundoff(data: JunctionStiffness) -> JunctionStiffness:
    """Project numerical roundoff out of the six analytical rigid modes."""

    rigid = rigid_connection_modes(data.connection_points)
    projector = np.eye(data.matrix.shape[0]) - rigid @ np.linalg.pinv(rigid)
    matrix = projector.T @ data.matrix @ projector
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    tolerance = 1.0e-10 * max(float(np.max(np.abs(values))), 1.0)
    values[np.abs(values) < tolerance] = 0.0
    if np.min(values) < -tolerance:
        raise ValueError("The junction stiffness contains a negative deformational mode.")
    matrix = (vectors * np.maximum(values, 0.0)) @ vectors.T
    return JunctionStiffness(data.connection_points, 0.5 * (matrix + matrix.T))
