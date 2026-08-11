"""Beam and hybrid beam--junction structure-gene homogenization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import linalg

from .beam import (
    BeamTheory,
    FloatArray,
    HomogenizationTerms,
    add_element_terms,
    create_homogenization_terms,
    periodic_reduction,
)
from .beam_euler import element_terms as euler_element_terms
from .beam_timoshenko import element_terms as timoshenko_element_terms
from .junction import (
    JunctionStiffness,
    add_junction_terms,
    connection_matrices,
    read_junction_stiffness,
    write_junction_stiffness,
)
from .junction_solid import JunctionSolution, analyze_junction
from .sc_hybrid_input import (
    HybridSupplement,
    StructuralGenomeInput,
    build_beam_discretization,
    read_hybrid_supplement,
    read_solid_junction,
    read_structural_genome,
    validate_hybrid_input,
)


@dataclass(frozen=True)
class HomogenizationResult:
    """Effective properties and retained homogenization operators."""

    effective_stiffness: FloatArray
    effective_compliance: FloatArray
    engineering_constants: dict[str, float]
    has_mechanism: bool
    full_terms: HomogenizationTerms
    periodic_matrix: FloatArray
    reduced_fluctuation: FloatArray
    full_fluctuation: FloatArray
    number_of_full_dofs: int
    number_of_independent_dofs: int
    number_of_junctions: int
    junction_analysis_time: float
    homogenization_time: float
    total_time: float
    nodes: FloatArray
    elements: tuple
    junction_stiffness_by_type: dict[int, JunctionStiffness]
    junction_solution_by_type: dict[int, JunctionSolution]
    junction_assemblies: tuple["JunctionAssembly", ...]


@dataclass(frozen=True)
class JunctionAssembly:
    """Maps one physical junction between global and connection variables."""

    identifier: int
    junction_type_id: int
    b_v: FloatArray
    b_epsilon: FloatArray


def _assemble_beams(
    model: StructuralGenomeInput,
    supplement: HybridSupplement,
) -> tuple[HomogenizationTerms, dict[int, np.ndarray], FloatArray, tuple]:
    nodes, elements = build_beam_discretization(model, supplement)
    terms = create_homogenization_terms(len(nodes))
    connectivity: dict[int, np.ndarray] = {}
    for element in elements:
        beam_type = supplement.beam_types[element.beam_type_id]
        if beam_type.theory == BeamTheory.EULER_BERNOULLI:
            local_terms = euler_element_terms(
                nodes,
                element.node_ids,
                element.frame,
                beam_type.section_stiffness,
            )
        elif beam_type.theory == BeamTheory.TIMOSHENKO:
            local_terms = timoshenko_element_terms(
                nodes,
                element.node_ids,
                element.frame,
                beam_type.section_stiffness,
            )
        else:
            raise ValueError(f"Unsupported beam theory: {beam_type.theory}")
        add_element_terms(terms, local_terms)
        connectivity[element.identifier] = element.node_ids
    return terms, connectivity, nodes, elements


def _load_junction_types(
    model: StructuralGenomeInput,
    supplement: HybridSupplement,
) -> tuple[dict[int, JunctionStiffness], dict[int, JunctionSolution], float]:
    if model.junction_flag == 0:
        if supplement.junction_types or supplement.junction_instances or supplement.junction_connections:
            raise ValueError("junction_flag=0 does not permit hybrid junction records.")
        return {}, {}, 0.0
    if not supplement.junction_types:
        raise ValueError("Hybrid junction modes require at least one junction type.")

    started = perf_counter()
    stiffness_by_type: dict[int, JunctionStiffness] = {}
    solution_by_type: dict[int, JunctionSolution] = {}
    for identifier, junction_type in supplement.junction_types.items():
        if model.junction_flag == 1:
            solid = read_solid_junction(junction_type.source)
            solution = analyze_junction(solid)
            stiffness = solution.stiffness
            solution_by_type[identifier] = solution
            output_path = Path(str(junction_type.source) + ".kj")
            write_junction_stiffness(output_path, stiffness)
        else:
            stiffness = read_junction_stiffness(junction_type.source)
        if (
            len(stiffness.connection_points)
            != junction_type.number_of_connection_points
        ):
            raise ValueError(
                f"Junction type {identifier} declares "
                f"{junction_type.number_of_connection_points} connection points, "
                f"but its source contains {len(stiffness.connection_points)}."
            )
        stiffness_by_type[identifier] = stiffness
    return stiffness_by_type, solution_by_type, perf_counter() - started


def _assemble_junctions(
    model: StructuralGenomeInput,
    supplement: HybridSupplement,
    terms: HomogenizationTerms,
    connectivity: dict[int, np.ndarray],
    nodes: FloatArray,
    stiffness_by_type: dict[int, JunctionStiffness],
) -> tuple[JunctionAssembly, ...]:
    connections_by_junction: dict[int, list] = {}
    endpoint_node_counts: dict[int, int] = {}
    for node_ids in connectivity.values():
        for node in node_ids[:2]:
            endpoint_node_counts[int(node)] = endpoint_node_counts.get(int(node), 0) + 1
    connection_keys: set[tuple[int, int]] = set()
    endpoint_owners: set[tuple[int, int]] = set()
    for connection in supplement.junction_connections:
        key = (connection.junction_id, connection.connection_point_id)
        if key in connection_keys:
            raise ValueError(
                f"Junction {connection.junction_id}, connection "
                f"point {connection.connection_point_id} is listed twice."
            )
        connection_keys.add(key)
        endpoint = (connection.element_id, connection.endpoint)
        if endpoint in endpoint_owners:
            raise ValueError(
                f"Element {connection.element_id}, endpoint {connection.endpoint} belongs "
                "to more than one hybrid junction."
            )
        endpoint_owners.add(endpoint)
        node_ids = connectivity.get(connection.element_id)
        if node_ids is None:
            raise ValueError(f"A junction references undefined element {connection.element_id}.")
        if connection.endpoint not in (1, 2):
            raise ValueError("A junction endpoint must be one or two.")
        connection_node = int(node_ids[connection.endpoint - 1])
        if endpoint_node_counts[connection_node] != 1:
            raise ValueError(
                f"Connection point node {connection_node + 1} is shared by multiple beams; "
                "use a separate beam endpoint node for every junction connection point."
            )
        connections_by_junction.setdefault(connection.junction_id, []).append(connection)

    number_of_dofs = terms.e.shape[0]
    assemblies: list[JunctionAssembly] = []
    for identifier, instance in supplement.junction_instances.items():
        junction_type = supplement.junction_types.get(instance.junction_type_id)
        if junction_type is None:
            raise ValueError(
                f"Junction {identifier} references undefined type {instance.junction_type_id}."
            )
        stiffness = stiffness_by_type[instance.junction_type_id]
        b_v, b_epsilon = connection_matrices(
            number_of_dofs=number_of_dofs,
            nodes=nodes,
            elements=connectivity,
            instance=instance,
            connections=connections_by_junction.get(identifier, []),
            stiffness=stiffness,
        )
        add_junction_terms(terms, b_v, b_epsilon, stiffness)
        assemblies.append(JunctionAssembly(
            identifier=identifier,
            junction_type_id=instance.junction_type_id,
            b_v=b_v,
            b_epsilon=b_epsilon,
        ))
    undefined = set(connections_by_junction) - set(supplement.junction_instances)
    if undefined:
        raise ValueError(f"Connections reference undefined junctions: {sorted(undefined)}")
    return tuple(assemblies)


def _minimum_norm_solve(matrix: FloatArray, right_hand_side: FloatArray) -> FloatArray:
    solution, _, _, _ = linalg.lstsq(
        matrix,
        right_hand_side,
        cond=None,
        lapack_driver="gelsd",
        check_finite=True,
    )
    return solution


def solve_homogenization(
    terms: HomogenizationTerms,
    periodic_matrix: FloatArray,
    volume: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Minimize the assembled energy for six macroscopic strain cases."""

    if volume <= 0.0:
        raise ValueError("The structure-gene volume must be positive.")
    p = np.asarray(periodic_matrix, dtype=float)
    e_reduced = p.T @ terms.e @ p
    d_reduced = p.T @ terms.d_h_epsilon
    constraint = terms.d_h_lambda @ p

    solve_constraint = _minimum_norm_solve(e_reduced, constraint.T)
    solve_coupling = _minimum_norm_solve(e_reduced, d_reduced)
    constraint_matrix = constraint @ solve_constraint
    constraint_right_hand_side = (
        constraint @ solve_coupling - terms.f_h_lambda
    )
    multiplier = _minimum_norm_solve(
        constraint_matrix,
        constraint_right_hand_side,
    )
    reduced_fluctuation = solve_coupling - solve_constraint @ multiplier
    full_fluctuation = p @ reduced_fluctuation
    effective_stiffness = (
        terms.d_epsilon_epsilon
        - 2.0 * full_fluctuation.T @ terms.d_h_epsilon
        + full_fluctuation.T @ terms.e @ full_fluctuation
    ) / volume
    effective_stiffness = 0.5 * (effective_stiffness + effective_stiffness.T)
    tolerance = 1.0e-12 * max(float(np.linalg.norm(effective_stiffness)), 1.0)
    effective_stiffness[np.abs(effective_stiffness) < tolerance] = 0.0
    return effective_stiffness, reduced_fluctuation, full_fluctuation


def calculate_engineering_constants(
    stiffness: FloatArray,
) -> tuple[FloatArray, dict[str, float], bool]:
    """Calculate compliance and conventional engineering constants."""

    eigenvalues = linalg.eigvalsh(stiffness, check_finite=True)
    tolerance = 1.0e-10 * max(float(np.max(np.abs(eigenvalues))), 1.0)
    has_mechanism = bool(np.min(eigenvalues) <= tolerance)
    compliance = (
        linalg.pinvh(stiffness, rtol=1.0e-10, check_finite=True)
        if has_mechanism
        else linalg.inv(stiffness, check_finite=True)
    )
    compliance_tolerance = np.finfo(float).eps * max(float(np.linalg.norm(compliance)), 1.0)

    def reciprocal(value: float) -> float:
        return float(np.nan if abs(value) <= compliance_tolerance else 1.0 / value)

    def poisson(numerator: float, denominator: float) -> float:
        return float(
            np.nan if abs(denominator) <= compliance_tolerance else -numerator / denominator
        )

    constants = {
        "E1": reciprocal(compliance[0, 0]),
        "E2": reciprocal(compliance[1, 1]),
        "E3": reciprocal(compliance[2, 2]),
        "G23": reciprocal(compliance[3, 3]),
        "G13": reciprocal(compliance[4, 4]),
        "G12": reciprocal(compliance[5, 5]),
        "nu12": poisson(compliance[1, 0], compliance[0, 0]),
        "nu13": poisson(compliance[2, 0], compliance[0, 0]),
        "nu23": poisson(compliance[2, 1], compliance[1, 1]),
    }
    return compliance, constants, has_mechanism


def homogenize(
    input_path: str | Path,
    supplement_path: str | Path | None = None,
) -> tuple[StructuralGenomeInput, HybridSupplement, HomogenizationResult]:
    """Run one of the three beam--junction homogenization modes."""

    total_started = perf_counter()
    model = read_structural_genome(input_path)
    if model.dimension != 3 or model.element_flag != 2:
        raise ValueError("This driver requires a 3D structure gene made from beam elements.")
    if model.analysis != 0 or model.temperature_flag != 0:
        raise ValueError("The beam--junction driver currently supports elastic homogenization only.")
    supplement_path = (
        Path(str(model.path) + ".msg") if supplement_path is None else Path(supplement_path)
    )
    supplement = read_hybrid_supplement(supplement_path)
    validate_hybrid_input(model, supplement)
    terms, connectivity, nodes, elements = _assemble_beams(model, supplement)
    stiffness_by_type, solution_by_type, junction_analysis_time = _load_junction_types(model, supplement)
    junction_assemblies = _assemble_junctions(
        model, supplement, terms, connectivity, nodes, stiffness_by_type
    )

    homogenization_started = perf_counter()
    p_sparse = periodic_reduction(len(nodes), model.periodic_pairs)
    p = p_sparse.toarray()
    stiffness, reduced_fluctuation, full_fluctuation = solve_homogenization(
        terms, p, model.volume
    )
    compliance, constants, has_mechanism = calculate_engineering_constants(stiffness)
    homogenization_time = perf_counter() - homogenization_started
    result = HomogenizationResult(
        effective_stiffness=stiffness,
        effective_compliance=compliance,
        engineering_constants=constants,
        has_mechanism=has_mechanism,
        full_terms=terms,
        periodic_matrix=p,
        reduced_fluctuation=reduced_fluctuation,
        full_fluctuation=full_fluctuation,
        number_of_full_dofs=terms.e.shape[0],
        number_of_independent_dofs=p.shape[1],
        number_of_junctions=len(supplement.junction_instances),
        junction_analysis_time=junction_analysis_time,
        homogenization_time=homogenization_time,
        total_time=perf_counter() - total_started,
        nodes=nodes,
        elements=elements,
        junction_stiffness_by_type=stiffness_by_type,
        junction_solution_by_type=solution_by_type,
        junction_assemblies=junction_assemblies,
    )
    return model, supplement, result
