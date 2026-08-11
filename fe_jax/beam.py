"""Shared data and operations for three-dimensional beam elements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class BeamTheory(IntEnum):
    """Beam formulations supported by the structure-gene solver."""

    EULER_BERNOULLI = 0
    TIMOSHENKO = 1


@dataclass(frozen=True)
class BeamType:
    """Properties shared by a batch of beam elements.

    Parameters
    ----------
    identifier : one-based beam-type number
    theory : beam formulation
    number_of_nodes : number of interpolation nodes per element
    section_stiffness : generalized section stiffness
    """

    identifier: int
    theory: BeamTheory
    number_of_nodes: int
    section_stiffness: FloatArray


@dataclass(frozen=True)
class BeamElement:
    """A beam element in the structure gene."""

    identifier: int
    node_ids: IntArray
    beam_type_id: int
    frame: FloatArray


@dataclass
class HomogenizationTerms:
    """Global arrays assembled before periodic reduction."""

    e: FloatArray
    d_h_epsilon: FloatArray
    d_epsilon_epsilon: FloatArray
    d_h_lambda: FloatArray
    f_h_lambda: FloatArray


@dataclass(frozen=True)
class BeamElementTerms:
    """Element arrays and indexing required by homogenization."""

    dofs: IntArray
    e: FloatArray
    d_h_epsilon: FloatArray
    d_epsilon_epsilon: FloatArray
    d_h_lambda: FloatArray
    f_h_lambda: FloatArray
    length: float
    frame: FloatArray


def node_dofs(node: int) -> IntArray:
    """Return the six global variable numbers associated with one node."""

    return 6 * node + np.arange(6, dtype=np.int64)


def normalize(vector: FloatArray, tolerance: float = 1.0e-12) -> FloatArray:
    """Return a normalized vector and reject a near-zero input."""

    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        raise ValueError("Cannot normalize a near-zero vector.")
    return np.asarray(vector, dtype=float) / norm


def beam_frame(
    start: FloatArray,
    end: FloatArray,
    transverse: FloatArray,
    tolerance: float = 1.0e-12,
) -> FloatArray:
    """Build an orthonormal beam frame whose rows are local basis vectors."""

    b1 = normalize(np.asarray(end) - np.asarray(start), tolerance)
    b2 = np.asarray(transverse, dtype=float) - np.dot(transverse, b1) * b1
    if np.linalg.norm(b2) <= tolerance:
        raise ValueError("The beam transverse direction is parallel to its axis.")
    b2 = normalize(b2, tolerance)
    b3 = normalize(np.cross(b1, b2), tolerance)
    result = np.vstack((b1, b2, b3))
    result[np.abs(result) < tolerance] = 0.0
    return result


def frame_from_points(points: FloatArray) -> FloatArray:
    """Build a local frame from the three-point OpenSG orientation record."""

    points = np.asarray(points, dtype=float)
    if points.shape != (3, 3):
        raise ValueError("An orientation record must contain three 3D points.")
    return beam_frame(points[0], points[1], points[2] - points[0])


def variable_transformation(frame: FloatArray, number_of_nodes: int) -> FloatArray:
    """Map global nodal translations and rotations to a local beam frame."""

    nodal = linalg.block_diag(frame, frame)
    return linalg.block_diag(*([nodal] * number_of_nodes))


def macro_displacement_matrix(point: FloatArray) -> FloatArray:
    """Map engineering macroscopic strain to displacement at a point."""

    x, y, z = np.asarray(point, dtype=float)
    result = np.zeros((6, 6), dtype=float)
    result[:3] = np.array(
        [
            [x, 0.0, 0.0, 0.0, 0.5 * z, 0.5 * y],
            [0.0, y, 0.0, 0.5 * z, 0.0, 0.5 * x],
            [0.0, 0.0, z, 0.5 * y, 0.5 * x, 0.0],
        ]
    )
    return result


def local_macro_strain_row(frame: FloatArray, i: int, j: int) -> FloatArray:
    """Map engineering macroscopic strain to one local tensor component."""

    return np.array(
        [
            frame[i, 0] * frame[j, 0],
            frame[i, 1] * frame[j, 1],
            frame[i, 2] * frame[j, 2],
            0.5 * (frame[i, 1] * frame[j, 2] + frame[i, 2] * frame[j, 1]),
            0.5 * (frame[i, 0] * frame[j, 2] + frame[i, 2] * frame[j, 0]),
            0.5 * (frame[i, 0] * frame[j, 1] + frame[i, 1] * frame[j, 0]),
        ],
        dtype=float,
    )


def create_homogenization_terms(number_of_nodes: int) -> HomogenizationTerms:
    """Allocate zero-filled global homogenization arrays."""

    number_of_dofs = 6 * number_of_nodes
    return HomogenizationTerms(
        e=np.zeros((number_of_dofs, number_of_dofs), dtype=float),
        d_h_epsilon=np.zeros((number_of_dofs, 6), dtype=float),
        d_epsilon_epsilon=np.zeros((6, 6), dtype=float),
        d_h_lambda=np.zeros((6, number_of_dofs), dtype=float),
        f_h_lambda=np.zeros((6, 6), dtype=float),
    )


def add_element_terms(
    global_terms: HomogenizationTerms,
    element_terms: BeamElementTerms,
) -> None:
    """Add one beam element contribution to global arrays."""

    dofs = element_terms.dofs
    global_terms.e[np.ix_(dofs, dofs)] += element_terms.e
    global_terms.d_h_epsilon[dofs] += element_terms.d_h_epsilon
    global_terms.d_epsilon_epsilon += element_terms.d_epsilon_epsilon
    global_terms.d_h_lambda[:, dofs] += element_terms.d_h_lambda
    global_terms.f_h_lambda += element_terms.f_h_lambda


def periodic_reduction(
    number_of_nodes: int,
    pairs: list[tuple[int, int]],
) -> sparse.csr_matrix:
    """Build the full-to-independent periodic mapping for six variables per node."""

    parent = np.arange(number_of_nodes, dtype=np.int64)

    def root(node: int) -> int:
        current = int(node)
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = int(parent[current])
        return current

    for slave, master in pairs:
        if not 0 <= slave < number_of_nodes or not 0 <= master < number_of_nodes:
            raise ValueError("A periodic node pair references an undefined node.")
        slave_root = root(slave)
        master_root = root(master)
        if slave_root != master_root:
            parent[slave_root] = master_root

    roots = [root(node) for node in range(number_of_nodes)]
    unique = {value: index for index, value in enumerate(dict.fromkeys(roots))}
    rows = np.arange(6 * number_of_nodes, dtype=np.int64)
    columns = np.empty_like(rows)
    for node, representative in enumerate(roots):
        start = 6 * node
        columns[start : start + 6] = 6 * unique[representative] + np.arange(6)
    return sparse.csr_matrix(
        (np.ones_like(rows, dtype=float), (rows, columns)),
        shape=(6 * number_of_nodes, 6 * len(unique)),
    )


def as_float_array(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
    """Convert an input to a finite floating-point array of a required shape."""

    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, received {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value.")
    return result
