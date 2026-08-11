"""Three-dimensional Euler--Bernoulli beam elements."""

from __future__ import annotations

import numpy as np

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


_AXIAL_COEFFICIENTS = (
    np.array([0.0, -0.5, 0.5]),
    np.array([0.0, 0.5, 0.5]),
    np.array([1.0, 0.0, -1.0]),
)
_BENDING_TRANSLATION_COEFFICIENTS = (
    np.array([0.0, 0.0, 1.0, -1.25, -0.5, 0.75]),
    np.array([0.0, 0.0, 1.0, 1.25, -0.5, -0.75]),
    np.array([1.0, 0.0, -2.0, 0.0, 1.0]),
)


def _evaluate(coefficients: FloatArray, xi: float, derivative: int = 0) -> float:
    return float(np.polynomial.polynomial.polyval(
        xi, np.polynomial.polynomial.polyder(coefficients, m=derivative)
    ))


def _bending_rotation_coefficients(length: float) -> tuple[FloatArray, ...]:
    return (
        length / 8.0 * np.array([0.0, 0.0, 1.0, -1.0, -1.0, 1.0]),
        length / 8.0 * np.array([0.0, 0.0, -1.0, -1.0, 1.0, 1.0]),
        length / 2.0 * np.array([0.0, 1.0, 0.0, -2.0, 0.0, 1.0]),
    )


def three_node_shape_matrix(xi: float, length: float) -> FloatArray:
    """Return the four by eighteen interpolation used by the Mathematica model."""

    axial = [_evaluate(coefficients, xi) for coefficients in _AXIAL_COEFFICIENTS]
    transverse = [
        _evaluate(coefficients, xi)
        for coefficients in _BENDING_TRANSLATION_COEFFICIENTS
    ]
    rotations = [
        _evaluate(coefficients, xi)
        for coefficients in _bending_rotation_coefficients(length)
    ]
    result = np.zeros((4, 18), dtype=float)
    for node in range(3):
        start = 6 * node
        result[0, start] = axial[node]
        result[1, start + 1] = transverse[node]
        result[1, start + 5] = rotations[node]
        result[2, start + 2] = transverse[node]
        result[2, start + 4] = -rotations[node]
        result[3, start + 3] = axial[node]
    return result


def three_node_center_matrix(xi: float, length: float) -> FloatArray:
    """Return centerline translations and rotations in local coordinates."""

    result = np.zeros((6, 18), dtype=float)
    axial = [_evaluate(coefficients, xi) for coefficients in _AXIAL_COEFFICIENTS]
    transverse = [
        _evaluate(coefficients, xi)
        for coefficients in _BENDING_TRANSLATION_COEFFICIENTS
    ]
    transverse_derivative = [
        (2.0 / length) * _evaluate(coefficients, xi, 1)
        for coefficients in _BENDING_TRANSLATION_COEFFICIENTS
    ]
    rotation_coefficients = _bending_rotation_coefficients(length)
    rotations = [_evaluate(coefficients, xi) for coefficients in rotation_coefficients]
    rotation_derivative = [
        (2.0 / length) * _evaluate(coefficients, xi, 1)
        for coefficients in rotation_coefficients
    ]
    for node in range(3):
        start = 6 * node
        result[0, start] = axial[node]
        result[1, start + 1] = transverse[node]
        result[1, start + 5] = rotations[node]
        result[2, start + 2] = transverse[node]
        result[2, start + 4] = -rotations[node]
        result[3, start + 3] = axial[node]
        # theta_2=-du_3/dx and theta_3=du_2/dx.
        result[4, start + 2] = -transverse_derivative[node]
        result[4, start + 4] = rotation_derivative[node]
        result[5, start + 1] = transverse_derivative[node]
        result[5, start + 5] = rotation_derivative[node]
    return result


def three_node_generalized_strain_matrix(xi: float, length: float) -> FloatArray:
    """Return the four by eighteen Euler generalized-strain matrix."""

    result = np.zeros((4, 18), dtype=float)
    rotation_coefficients = _bending_rotation_coefficients(length)
    for node in range(3):
        start = 6 * node
        result[0, start] = (
            2.0 / length * _evaluate(_AXIAL_COEFFICIENTS[node], xi, 1)
        )
        result[1, start + 3] = (
            2.0 / length * _evaluate(_AXIAL_COEFFICIENTS[node], xi, 1)
        )
        result[2, start + 2] = -(
            (2.0 / length) ** 2
            * _evaluate(_BENDING_TRANSLATION_COEFFICIENTS[node], xi, 2)
        )
        result[2, start + 4] = (
            (2.0 / length) ** 2 * _evaluate(rotation_coefficients[node], xi, 2)
        )
        result[3, start + 1] = (
            (2.0 / length) ** 2
            * _evaluate(_BENDING_TRANSLATION_COEFFICIENTS[node], xi, 2)
        )
        result[3, start + 5] = (
            (2.0 / length) ** 2 * _evaluate(rotation_coefficients[node], xi, 2)
        )
    return result


def _three_node_macro_map(frame: FloatArray) -> FloatArray:
    b2 = local_macro_strain_row(frame, 1, 0)
    b3 = local_macro_strain_row(frame, 2, 0)
    nodal = np.vstack((np.zeros((4, 6)), b3, -b2))
    return np.vstack((nodal, nodal, nodal))


def local_stiffness(length: float, section_stiffness: FloatArray) -> FloatArray:
    """Return the twelve by twelve local Euler--Bernoulli stiffness."""

    section_stiffness = np.asarray(section_stiffness, dtype=float)
    if section_stiffness.shape != (4, 4):
        raise ValueError("Euler--Bernoulli section stiffness must have shape (4, 4).")
    if not np.allclose(section_stiffness, np.diag(np.diag(section_stiffness))):
        raise ValueError("The closed-form Euler stiffness requires diagonal section stiffness.")
    ea, gj, ei2, ei3 = np.diag(section_stiffness)
    result = np.zeros((12, 12), dtype=float)
    for dof, value in ((0, ea / length), (3, gj / length)):
        result[dof, dof] += value
        result[dof + 6, dof + 6] += value
        result[dof, dof + 6] -= value
        result[dof + 6, dof] -= value

    def add_bending(
        translation_1: int,
        rotation_1: int,
        translation_2: int,
        rotation_2: int,
        stiffness: float,
        sign: float,
    ) -> None:
        block = stiffness / length**3 * np.array(
            [
                [12.0, sign * 6.0 * length, -12.0, sign * 6.0 * length],
                [sign * 6.0 * length, 4.0 * length**2, -sign * 6.0 * length, 2.0 * length**2],
                [-12.0, -sign * 6.0 * length, 12.0, -sign * 6.0 * length],
                [sign * 6.0 * length, 2.0 * length**2, -sign * 6.0 * length, 4.0 * length**2],
            ]
        )
        indices = [translation_1, rotation_1, translation_2, rotation_2]
        result[np.ix_(indices, indices)] += block

    add_bending(1, 5, 7, 11, ei3, 1.0)
    add_bending(2, 4, 8, 10, ei2, -1.0)
    return 0.5 * (result + result.T)


def _three_node_element_terms(
    nodes: FloatArray,
    node_ids: IntArray,
    frame: FloatArray,
    section_stiffness: FloatArray,
) -> BeamElementTerms:
    """Calculate the contribution of one three-node Euler element."""

    section_stiffness = np.asarray(section_stiffness, dtype=float)
    if section_stiffness.shape != (4, 4):
        raise ValueError("Euler--Bernoulli section stiffness must have shape (4, 4).")
    length = float(np.linalg.norm(nodes[node_ids[1]] - nodes[node_ids[0]]))
    if length <= 0.0:
        raise ValueError("An Euler--Bernoulli element has zero length.")
    midpoint = 0.5 * (nodes[node_ids[0]] + nodes[node_ids[1]])
    if not np.allclose(nodes[node_ids[2]], midpoint, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("The third Euler node must lie at the element midpoint.")

    transformation = variable_transformation(frame, 3)
    macro_map = _three_node_macro_map(frame)
    gamma_epsilon = np.vstack(
        (local_macro_strain_row(frame, 0, 0), np.zeros((3, 6)))
    )
    e = np.zeros((18, 18), dtype=float)
    d_h_epsilon = np.zeros((18, 6), dtype=float)
    d_epsilon_epsilon = np.zeros((6, 6), dtype=float)
    d_h_lambda = np.zeros((6, 18), dtype=float)
    f_h_lambda = np.zeros((6, 6), dtype=float)
    jacobian = 0.5 * length
    for xi, weight in zip(_QUADRATURE_POINTS, _QUADRATURE_WEIGHTS, strict=True):
        interpolation = three_node_shape_matrix(float(xi), length)
        strain = three_node_generalized_strain_matrix(float(xi), length)
        b = strain @ transformation
        effective_macro_strain = gamma_epsilon + strain @ macro_map
        factor = float(weight) * jacobian
        e += (b.T @ section_stiffness @ b) * factor
        d_h_epsilon += (b.T @ section_stiffness @ effective_macro_strain) * factor
        d_epsilon_epsilon += (
            effective_macro_strain.T @ section_stiffness @ effective_macro_strain
        ) * factor
        d_h_lambda[:3] += (
            frame.T @ interpolation[:3] @ transformation
        ) * factor
    dofs = np.concatenate([node_dofs(int(node)) for node in node_ids])
    return BeamElementTerms(
        dofs=dofs,
        e=0.5 * (e + e.T),
        d_h_epsilon=d_h_epsilon,
        d_epsilon_epsilon=0.5 * (d_epsilon_epsilon + d_epsilon_epsilon.T),
        d_h_lambda=d_h_lambda,
        f_h_lambda=f_h_lambda,
        length=length,
        frame=np.asarray(frame, dtype=float),
    )


def element_terms(
    nodes: FloatArray,
    node_ids: IntArray,
    frame: FloatArray,
    section_stiffness: FloatArray,
) -> BeamElementTerms:
    """Calculate a three-node Hermite Euler--Bernoulli contribution."""

    node_ids = np.asarray(node_ids, dtype=np.int64)
    if node_ids.shape != (3,):
        raise ValueError("Euler--Bernoulli connectivity requires three internal nodes.")
    return _three_node_element_terms(nodes, node_ids, frame, section_stiffness)
