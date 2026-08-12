"""Three-dimensional localization for OpenSG beam-and-junction models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from .beam import (
    BeamTheory,
    FloatArray,
    local_macro_strain_row,
    variable_transformation,
)
from . import beam_euler, beam_timoshenko
from .junction_solid import (
    SolidJunctionModel,
    _material_stiffness,
    _strain_matrix,
    _tetrahedron_shape_derivatives,
    analyze_junction,
)
from .sc_glb_input import GlobalFields
from .sc_hybrid_input import (
    HybridSupplement,
    StructuralGenomeInput,
    read_solid_junction,
)
from .vabs_localization import VABSSession

if TYPE_CHECKING:
    from .hybrid_homogenization import HomogenizationResult


@dataclass(frozen=True)
class LocalFields:
    """Recovered nodal fields in the OpenSG problem coordinate system."""

    coordinates: FloatArray
    displacement: FloatArray
    strain: FloatArray
    stress: FloatArray
    region: np.ndarray
    owner_id: np.ndarray


@dataclass(frozen=True)
class BeamStationState:
    """Recovered beam state with stress resultants [F1,F2,F3,M1,M2,M3]."""

    element_id: int
    beam_type_id: int
    xi: float
    center: FloatArray
    frame: FloatArray
    displacement_local: FloatArray
    rotation_local: FloatArray
    generalized_strain: FloatArray
    resultants: FloatArray


def _engineering_tensor(values: FloatArray) -> FloatArray:
    e11, e22, e33, e23, e13, e12 = values
    return np.array([
        [e11, 0.5 * e12, 0.5 * e13],
        [0.5 * e12, e22, 0.5 * e23],
        [0.5 * e13, 0.5 * e23, e33],
    ])


def _engineering_vector(tensor: FloatArray) -> FloatArray:
    return np.array([
        tensor[0, 0], tensor[1, 1], tensor[2, 2],
        2.0 * tensor[1, 2], 2.0 * tensor[0, 2], 2.0 * tensor[0, 1],
    ])


def _stress_tensor(values: FloatArray) -> FloatArray:
    s11, s22, s33, s23, s13, s12 = values
    return np.array([
        [s11, s12, s13], [s12, s22, s23], [s13, s23, s33]
    ])


def _stress_vector(tensor: FloatArray) -> FloatArray:
    return np.array([
        tensor[0, 0], tensor[1, 1], tensor[2, 2],
        tensor[1, 2], tensor[0, 2], tensor[0, 1],
    ])


def _macro_kinematics(fields: GlobalFields) -> tuple[FloatArray, FloatArray, FloatArray]:
    strain = _engineering_tensor(fields.strain)
    gradient = fields.deformation.T - np.eye(3)
    spin = 0.5 * (gradient - gradient.T)
    axial = np.array([spin[2, 1], spin[0, 2], spin[1, 0]])
    return strain, spin, axial


def _gamma_epsilon(frame: FloatArray, theory: BeamTheory) -> FloatArray:
    rows = [local_macro_strain_row(frame, 0, 0)]
    if theory == BeamTheory.TIMOSHENKO:
        rows.extend((
            local_macro_strain_row(frame, 0, 1),
            local_macro_strain_row(frame, 0, 2),
        ))
    rows.extend([np.zeros(6)] * 3)
    return np.vstack(rows)


def recover_beam_states(
    result: HomogenizationResult,
    supplement: HybridSupplement,
    fields: GlobalFields,
    xi: FloatArray,
) -> list[BeamStationState]:
    """Recover member states needed by VABS at requested axial stations."""

    fluctuation = -result.full_fluctuation @ fields.strain
    strain_tensor, spin, spin_vector = _macro_kinematics(fields)
    states: list[BeamStationState] = []
    for element in sorted(result.elements, key=lambda item: item.identifier):
        beam_type = supplement.beam_types[element.beam_type_id]
        frame = element.frame
        node_ids = element.node_ids
        dofs = np.concatenate([
            6 * int(node) + np.arange(6, dtype=np.int64) for node in node_ids
        ])
        length = float(np.linalg.norm(
            result.nodes[node_ids[1]] - result.nodes[node_ids[0]]
        ))
        local = variable_transformation(frame, len(node_ids)) @ fluctuation[dofs]
        gamma_epsilon = _gamma_epsilon(frame, beam_type.theory)
        if beam_type.theory == BeamTheory.EULER_BERNOULLI:
            local = local + beam_euler._three_node_macro_map(frame) @ fields.strain
        element_states: list[BeamStationState] = []
        for coordinate in xi:
            coordinate = float(coordinate)
            if beam_type.theory == BeamTheory.EULER_BERNOULLI:
                center_local = beam_euler.three_node_center_matrix(
                    coordinate, length
                ) @ local
                strain_matrix = beam_euler.three_node_generalized_strain_matrix(
                    coordinate, length
                )
            else:
                center_local = beam_timoshenko.shape_matrix(coordinate) @ local
                strain_matrix = beam_timoshenko.generalized_strain_matrix(
                    coordinate, length
                )
            generalized = gamma_epsilon @ fields.strain + strain_matrix @ local
            resultants = beam_type.section_stiffness @ generalized
            shape = 0.5 * (coordinate + 1.0)
            center = (
                (1.0 - shape) * result.nodes[node_ids[0]]
                + shape * result.nodes[node_ids[1]]
            )
            affine_displacement = (
                fields.displacement + strain_tensor @ center + spin @ center
            )
            displacement_local = frame @ affine_displacement + center_local[:3]
            rotation_local = frame @ spin_vector + center_local[3:]
            if beam_type.theory == BeamTheory.EULER_BERNOULLI:
                full_resultants = np.array([
                    resultants[0], 0.0, 0.0,
                    resultants[1], resultants[2], resultants[3],
                ])
            else:
                full_resultants = resultants
            element_states.append(BeamStationState(
                element_id=element.identifier,
                beam_type_id=element.beam_type_id,
                xi=coordinate,
                center=center,
                frame=frame,
                displacement_local=displacement_local,
                rotation_local=rotation_local,
                generalized_strain=generalized,
                resultants=full_resultants,
            ))
        if beam_type.theory == BeamTheory.EULER_BERNOULLI and len(element_states) > 1:
            axial = 0.5 * (np.asarray(xi, dtype=float) + 1.0) * length
            moments_2 = np.asarray([state.resultants[4] for state in element_states])
            moments_3 = np.asarray([state.resultants[5] for state in element_states])
            force_2 = -np.gradient(moments_3, axial, edge_order=1)
            force_3 = np.gradient(moments_2, axial, edge_order=1)
            element_states = [
                BeamStationState(
                    **{
                        **state.__dict__,
                        "resultants": np.array([
                            state.resultants[0], force_2[index], force_3[index],
                            *state.resultants[3:],
                        ]),
                    }
                )
                for index, state in enumerate(element_states)
            ]
        states.extend(element_states)
    return states


def _transform_vabs_fields(state: BeamStationState, fields) -> LocalFields:
    frame = state.frame
    local_coordinates = np.column_stack((
        np.zeros(len(fields.section_coordinates)), fields.section_coordinates
    ))
    coordinates = state.center + local_coordinates @ frame
    displacement = fields.displacement @ frame
    strain = np.vstack([
        _engineering_vector(frame.T @ _engineering_tensor(value) @ frame)
        for value in fields.strain
    ])
    stress = np.vstack([
        _stress_vector(frame.T @ _stress_tensor(value) @ frame)
        for value in fields.stress
    ])
    return LocalFields(
        coordinates=coordinates,
        displacement=displacement,
        strain=strain,
        stress=stress,
        region=np.full(len(coordinates), "beam", dtype="U8"),
        owner_id=np.full(len(coordinates), state.element_id, dtype=np.int64),
    )


def recover_vabs_fields(
    states: list[BeamStationState],
    supplement: HybridSupplement,
    executable: str | Path | None = None,
) -> list[LocalFields]:
    """Run VABS for all recovered beam stations."""

    missing = sorted({
        state.beam_type_id for state in states
        if state.beam_type_id not in supplement.beam_recovery
    })
    if missing:
        raise ValueError(
            f"Missing BEAM_RECOVERY records for beam types {missing}."
        )
    recovered: list[LocalFields] = []
    with tempfile.TemporaryDirectory(prefix="opensg_vabs_") as directory:
        sessions = {
            beam_type_id: VABSSession(
                definition.source,
                Path(directory) / f"beam_type_{beam_type_id}",
                executable,
            )
            for beam_type_id, definition in supplement.beam_recovery.items()
            if any(state.beam_type_id == beam_type_id for state in states)
        }
        for state in states:
            local = sessions[state.beam_type_id].localize(
                state.displacement_local,
                state.rotation_local,
                state.resultants,
            )
            recovered.append(_transform_vabs_fields(state, local))
    return recovered


_TET10_NATURAL = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0], [0.5, 0.5, 0.0],
    [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
    [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
])


def _junction_nodal_fields(
    model: SolidJunctionModel,
    displacement_local: FloatArray,
    instance,
    fields: GlobalFields,
    owner_id: int,
) -> LocalFields:
    accumulated_strain = np.zeros((len(model.nodes), 6))
    accumulated_stress = np.zeros_like(accumulated_strain)
    weights = np.zeros(len(model.nodes))
    materials = {
        identifier: _material_stiffness(*properties)
        for identifier, properties in model.materials.items()
    }
    for element_index, element in enumerate(model.elements):
        coordinates = model.nodes[element]
        nodal_displacement = displacement_local[element].reshape(-1)
        number_of_nodes = len(element)
        natural_points = (
            _TET10_NATURAL if number_of_nodes == 10 else _TET10_NATURAL[:4]
        )
        corner_derivatives = _tetrahedron_shape_derivatives(
            number_of_nodes, 0.25, 0.25, 0.25
        )
        volume = abs(float(np.linalg.det(coordinates.T @ corner_derivatives))) / 6.0
        material = materials[int(model.material_ids[element_index])]
        for local_node, (r, s, t) in enumerate(natural_points):
            natural = _tetrahedron_shape_derivatives(number_of_nodes, r, s, t)
            jacobian = coordinates.T @ natural
            derivatives = natural @ np.linalg.inv(jacobian)
            strain = _strain_matrix(derivatives) @ nodal_displacement
            stress = material @ strain
            node = int(element[local_node])
            accumulated_strain[node] += volume * strain
            accumulated_stress[node] += volume * stress
            weights[node] += volume
    if np.any(weights <= 0.0):
        raise RuntimeError("A junction node has no strain/stress contribution.")
    strain_local = accumulated_strain / weights[:, None]
    stress_local = accumulated_stress / weights[:, None]
    frame = instance.frame
    coordinates_global = instance.origin + model.nodes @ frame
    _, spin, _ = _macro_kinematics(fields)
    displacement_global = displacement_local @ frame
    displacement_global += fields.displacement + coordinates_global @ spin.T
    strain_global = np.vstack([
        _engineering_vector(frame.T @ _engineering_tensor(value) @ frame)
        for value in strain_local
    ])
    stress_global = np.vstack([
        _stress_vector(frame.T @ _stress_tensor(value) @ frame)
        for value in stress_local
    ])
    return LocalFields(
        coordinates=coordinates_global,
        displacement=displacement_global,
        strain=strain_global,
        stress=stress_global,
        region=np.full(len(model.nodes), "junction", dtype="U8"),
        owner_id=np.full(len(model.nodes), owner_id, dtype=np.int64),
    )


def _junction_solid_source(stiffness_source: Path) -> Path:
    """Return the required solid input sharing a mode-2 junction basename."""

    source_text = str(stiffness_source)
    if source_text.lower().endswith(".sc.kj"):
        solid = Path(source_text[:-3])
    elif stiffness_source.suffix.lower() == ".kj":
        solid = stiffness_source.with_suffix(".sc")
    else:
        raise ValueError(
            f"Junction mode 2 source must end in .kj: {stiffness_source}"
        )
    interface = Path(str(solid) + ".msg")
    missing = [path for path in (solid, interface) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Junction localization for {stiffness_source} requires the "
            f"same-basename solid inputs {solid} and {interface}; missing "
            + ", ".join(str(path) for path in missing)
        )
    return solid


def recover_junction_fields(
    model: StructuralGenomeInput,
    supplement: HybridSupplement,
    result: HomogenizationResult,
    fields: GlobalFields,
) -> list[LocalFields]:
    """Read supplied junction solids and recover their three-dimensional fields."""

    if not result.junction_assemblies:
        return []
    prepared: dict[int, tuple[SolidJunctionModel, FloatArray]] = {}
    for junction_type_id, junction_type in supplement.junction_types.items():
        if model.junction_flag == 1:
            solid = read_solid_junction(junction_type.source)
            solution = result.junction_solution_by_type[junction_type_id]
        else:
            stiffness = result.junction_stiffness_by_type[junction_type_id]
            solid_source = _junction_solid_source(junction_type.source)
            solid = read_solid_junction(solid_source)
            expected = stiffness.connection_points
            actual = solid.connection_points
            if len(actual) != len(expected) or any(
                left.identifier != right.identifier
                or not np.allclose(left.origin, right.origin, rtol=1.0e-9, atol=1.0e-10)
                or not np.allclose(left.frame, right.frame, rtol=1.0e-9, atol=1.0e-10)
                for left, right in zip(actual, expected, strict=False)
            ):
                raise ValueError(
                    f"Junction recovery solid {solid_source} does not match "
                    f"junction type {junction_type_id} connection geometry."
                )
            solution = analyze_junction(solid)
        prepared[junction_type_id] = (solid, solution.displacement_recovery)

    recovered = []
    for assembly in sorted(result.junction_assemblies, key=lambda item: item.identifier):
        instance = supplement.junction_instances[assembly.identifier]
        solid, displacement_recovery = prepared[assembly.junction_type_id]
        connection_displacement = (
            assembly.b_epsilon - assembly.b_v @ result.full_fluctuation
        ) @ fields.strain
        displacement_local = (
            displacement_recovery @ connection_displacement
        ).reshape(-1, 3)
        recovered.append(_junction_nodal_fields(
            solid, displacement_local, instance, fields, assembly.identifier
        ))
    return recovered


def combine_fields(parts: list[LocalFields]) -> LocalFields:
    if not parts:
        empty = np.empty((0, 3))
        return LocalFields(
            empty, empty.copy(), np.empty((0, 6)), np.empty((0, 6)),
            np.empty(0, dtype="U8"), np.empty(0, dtype=np.int64),
        )
    return LocalFields(
        coordinates=np.vstack([part.coordinates for part in parts]),
        displacement=np.vstack([part.displacement for part in parts]),
        strain=np.vstack([part.strain for part in parts]),
        stress=np.vstack([part.stress for part in parts]),
        region=np.concatenate([part.region for part in parts]),
        owner_id=np.concatenate([part.owner_id for part in parts]),
    )


def dehomogenize(
    model: StructuralGenomeInput,
    supplement: HybridSupplement,
    result: HomogenizationResult,
    fields: GlobalFields,
    stations: int = 3,
    executable: str | Path | None = None,
) -> LocalFields:
    """Recover hybrid three-dimensional fields at VABS and junction nodes."""

    if stations < 2:
        raise ValueError("At least two axial VABS stations are required.")
    xi = np.linspace(-1.0, 1.0, stations)
    states = recover_beam_states(result, supplement, fields, xi)
    beam_fields = recover_vabs_fields(states, supplement, executable)
    junction_fields = recover_junction_fields(
        model, supplement, result, fields
    )
    return combine_fields([*beam_fields, *junction_fields])
