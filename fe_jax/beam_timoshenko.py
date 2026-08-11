"""Four-node three-dimensional Timoshenko beam element."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .beam import (
    BeamElementTerms,
    FloatArray,
    IntArray,
    local_macro_strain_row,
    node_dofs,
    variable_transformation,
)


_QUADRATURE_POINTS = np.array([-0.861136, -0.339981, 0.339981, 0.861136])
_QUADRATURE_WEIGHTS = np.array([0.347855, 0.652145, 0.652145, 0.347855])


def shape_functions(xi: float) -> FloatArray:
    """Evaluate cubic Lagrange functions in OpenSG beam-node order."""

    return np.array(
        [
            -9.0 / 16.0 * (xi - 1.0) * (xi + 1.0 / 3.0) * (xi - 1.0 / 3.0),
            9.0 / 16.0 * (xi + 1.0) * (xi + 1.0 / 3.0) * (xi - 1.0 / 3.0),
            27.0 / 16.0 * (xi + 1.0) * (xi - 1.0) * (xi - 1.0 / 3.0),
            -27.0 / 16.0 * (xi + 1.0) * (xi - 1.0) * (xi + 1.0 / 3.0),
        ],
        dtype=float,
    )


def shape_function_derivatives(xi: float) -> FloatArray:
    """Evaluate derivatives of the cubic Lagrange functions."""

    return np.array(
        [
            -27.0 * xi**2 / 16.0 + 9.0 * xi / 8.0 + 1.0 / 16.0,
            27.0 * xi**2 / 16.0 + 9.0 * xi / 8.0 - 1.0 / 16.0,
            81.0 * xi**2 / 16.0 - 9.0 * xi / 8.0 - 27.0 / 16.0,
            -81.0 * xi**2 / 16.0 - 9.0 * xi / 8.0 + 27.0 / 16.0,
        ],
        dtype=float,
    )


def shape_matrix(xi: float) -> FloatArray:
    """Return the local interpolation matrix."""

    identity = np.eye(6)
    return np.hstack([value * identity for value in shape_functions(xi)])


def generalized_strain_matrix(xi: float, length: float) -> FloatArray:
    """Return the six by twenty-four generalized-strain matrix."""

    values = shape_matrix(xi)
    derivatives = (2.0 / length) * shape_function_derivatives(xi)
    derivative_matrix = np.hstack([value * np.eye(6) for value in derivatives])
    result = np.empty((6, 24), dtype=float)
    result[0] = derivative_matrix[0]
    result[1] = derivative_matrix[1] - values[5]
    result[2] = derivative_matrix[2] + values[4]
    result[3] = derivative_matrix[3]
    result[4] = derivative_matrix[4]
    result[5] = derivative_matrix[5]
    return result


def element_terms(
    nodes: FloatArray,
    node_ids: IntArray,
    frame: FloatArray,
    section_stiffness: FloatArray,
) -> BeamElementTerms:
    """Calculate the contribution of one four-node Timoshenko element."""

    node_ids = np.asarray(node_ids, dtype=np.int64)
    if node_ids.shape != (4,):
        raise ValueError("A Timoshenko element requires four node numbers.")
    section_stiffness = np.asarray(section_stiffness, dtype=float)
    if section_stiffness.shape != (6, 6):
        raise ValueError("Timoshenko section stiffness must have shape (6, 6).")

    length = float(np.linalg.norm(nodes[node_ids[1]] - nodes[node_ids[0]]))
    if length <= 0.0:
        raise ValueError("A Timoshenko element has zero length.")

    transformation = variable_transformation(frame, 4)
    nodal_rotation = linalg.block_diag(frame, frame)
    gamma_epsilon = np.vstack(
        (
            local_macro_strain_row(frame, 0, 0),
            local_macro_strain_row(frame, 0, 1),
            local_macro_strain_row(frame, 0, 2),
            np.zeros((3, 6)),
        )
    )
    e = np.zeros((24, 24), dtype=float)
    d_h_epsilon = np.zeros((24, 6), dtype=float)
    d_epsilon_epsilon = np.zeros((6, 6), dtype=float)
    d_h_lambda = np.zeros((6, 24), dtype=float)

    jacobian = 0.5 * length
    for xi, weight in zip(_QUADRATURE_POINTS, _QUADRATURE_WEIGHTS, strict=True):
        interpolation = shape_matrix(float(xi))
        strain = generalized_strain_matrix(float(xi), length)
        b = strain @ transformation
        global_interpolation = nodal_rotation.T @ interpolation @ transformation
        factor = float(weight) * jacobian
        e += (b.T @ section_stiffness @ b) * factor
        d_h_epsilon += (b.T @ section_stiffness @ gamma_epsilon) * factor
        d_epsilon_epsilon += (
            gamma_epsilon.T @ section_stiffness @ gamma_epsilon
        ) * factor
        d_h_lambda += global_interpolation * factor

    dofs = np.concatenate([node_dofs(int(node)) for node in node_ids])
    return BeamElementTerms(
        dofs=dofs,
        e=0.5 * (e + e.T),
        d_h_epsilon=d_h_epsilon,
        d_epsilon_epsilon=0.5 * (d_epsilon_epsilon + d_epsilon_epsilon.T),
        d_h_lambda=d_h_lambda,
        f_h_lambda=np.zeros((6, 6), dtype=float),
        length=length,
        frame=np.asarray(frame, dtype=float),
    )
