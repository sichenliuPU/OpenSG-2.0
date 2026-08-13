"""OpenSG-style input readers for beam and hybrid structure genes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import shlex

import numpy as np

from .beam import (
    BeamElement,
    BeamTheory,
    BeamType,
    FloatArray,
    IntArray,
    beam_frame,
    frame_from_points,
)
from .junction import (
    JunctionConnection,
    JunctionConnectionPoint,
    JunctionInstance,
    JunctionStiffness,
    JunctionType,
)
from .junction_solid import SolidJunctionModel


@dataclass(frozen=True)
class StructuralGenomeInput:
    """Mesh and control information read from a standard ``.sc`` file."""

    path: Path
    analysis: int
    element_flag: int
    transformation_flag: int
    temperature_flag: int
    junction_flag: int
    dimension: int
    nodes: FloatArray
    element_ids: IntArray
    material_ids: IntArray
    connectivity: tuple[IntArray, ...]
    orientations: dict[int, FloatArray]
    periodic_pairs: list[tuple[int, int]]
    number_of_materials: int
    tail_records: tuple[str, ...]
    volume: float


@dataclass(frozen=True)
class HybridSupplement:
    """Beam properties and optional junction records from ``.sc.msg``."""

    version: int
    beam_types: dict[int, BeamType]
    beam_assignments: dict[int, int]
    junction_types: dict[int, JunctionType]
    junction_instances: dict[int, JunctionInstance]
    junction_connections: tuple[JunctionConnection, ...]
    beam_recovery: dict[int, "BeamRecovery"]


@dataclass(frozen=True)
class BeamRecovery:
    """VABS cross-section source used to localize one beam type."""

    beam_type_id: int
    source: Path


@dataclass(frozen=True)
class PeriodicOwnershipAudit:
    """Counts and mappings created by automatic periodic object ownership."""

    input_beams: int
    owned_beams: int
    input_junctions: int
    owned_junctions: int
    beam_owner_by_id: dict[int, int]
    junction_owner_by_id: dict[int, int]


def _data_lines(path: Path) -> list[str]:
    if not path.exists():
        raise ValueError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    lines: list[str] = []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read input file {path}: {error}") from error
    for raw_line in raw_lines:
        line = raw_line.split("#", 1)[0].split("!", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _integer_tokens(line: str, count: int, name: str) -> list[int]:
    values = line.split()
    if len(values) != count:
        raise ValueError(f"{name} requires {count} integer values.")
    try:
        return [int(value) for value in values]
    except ValueError as error:
        raise ValueError(f"{name} must contain only integer values: {line}") from error


def _require_records(
    lines: list[str], cursor: int, count: int, section: str, path: Path
) -> None:
    """Report a truncated input section before indexing its records."""

    if count < 0:
        raise ValueError(f"{section} count cannot be negative in {path}.")
    available = len(lines) - cursor
    if available < count:
        raise ValueError(
            f"{section} in {path} declares {count} records, but only "
            f"{max(available, 0)} remain."
        )


def _periodic_representatives(
    number_of_nodes: int, pairs: list[tuple[int, int]]
) -> list[int]:
    parent = list(range(number_of_nodes))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for slave, master in pairs:
        slave_root = root(slave)
        master_root = root(master)
        if slave_root != master_root:
            parent[slave_root] = master_root
    return [root(node) for node in range(number_of_nodes)]


def _periodic_box(nodes: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    minimum = np.min(nodes, axis=0)
    periods = np.ptp(nodes, axis=0)
    scale = max(float(np.max(periods)), 1.0)
    tolerance = 1.0e-8 * scale
    return minimum, periods, tolerance


def _canonical_point(
    point: FloatArray,
    minimum: FloatArray,
    periods: FloatArray,
    tolerance: float,
) -> tuple[FloatArray, tuple[int, int, int]]:
    """Return a point modulo the SG periods and its integer image index."""

    canonical = np.asarray(point, dtype=float).copy()
    image = np.zeros(3, dtype=int)
    for axis, period in enumerate(periods):
        if period <= tolerance:
            continue
        coordinate = (point[axis] - minimum[axis]) / period
        integer = int(np.floor(coordinate + tolerance / period))
        fraction = coordinate - integer
        if np.isclose(fraction, 1.0, rtol=0.0, atol=tolerance / period):
            integer += 1
            fraction = 0.0
        if np.isclose(fraction, 0.0, rtol=0.0, atol=tolerance / period):
            fraction = 0.0
        canonical[axis] = minimum[axis] + fraction * period
        image[axis] = integer
    return canonical, tuple(int(value) for value in image)


def _quantized(values: FloatArray, tolerance: float) -> tuple[int, ...]:
    return tuple(np.rint(np.asarray(values).ravel() / tolerance).astype(np.int64))


def _beam_frame_for_input(
    model: StructuralGenomeInput, element_id: int, node_ids: IntArray
) -> FloatArray:
    frame = model.orientations.get(element_id)
    if frame is not None:
        return frame
    start, end = model.nodes[node_ids[0]], model.nodes[node_ids[1]]
    trial = np.array([0.0, 0.0, 1.0])
    direction = (end - start) / np.linalg.norm(end - start)
    if abs(float(direction @ trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    return beam_frame(start, end, trial)


def apply_periodic_ownership(
    model: StructuralGenomeInput,
    supplement: HybridSupplement,
) -> tuple[StructuralGenomeInput, HybridSupplement, PeriodicOwnershipAudit]:
    """Keep one complete beam and junction from every periodic image orbit.

    Object ownership is separate from nodal periodic reduction: tying image
    DOFs with ``P`` does not remove the stiffness of duplicate full-section
    beams or complete junctions.  This preprocessing stage selects the image
    on the lexicographically minimum representative face/edge/corner and
    rewrites all junction connections to that owned image.
    """

    minimum, periods, tolerance = _periodic_box(model.nodes)
    beam_records: dict[tuple, dict[tuple[int, int, int], list[int]]] = {}
    element_index = {
        int(identifier): index for index, identifier in enumerate(model.element_ids)
    }
    for index, (identifier_value, node_ids) in enumerate(
        zip(model.element_ids, model.connectivity, strict=True)
    ):
        identifier = int(identifier_value)
        start = model.nodes[int(node_ids[0])]
        end = model.nodes[int(node_ids[1])]
        canonical_start, image = _canonical_point(
            start, minimum, periods, tolerance
        )
        vector = end - start
        frame = _beam_frame_for_input(model, identifier, node_ids)
        beam_type_id = supplement.beam_assignments.get(identifier)
        beam_type = supplement.beam_types.get(beam_type_id)
        beam_type_key = (
            (int(beam_type.theory), beam_type.number_of_nodes,
             _quantized(beam_type.section_stiffness, tolerance))
            if beam_type is not None else ("undefined", beam_type_id)
        )
        key = (
            _quantized(canonical_start, tolerance),
            _quantized(vector, tolerance),
            _quantized(frame, tolerance),
            int(model.material_ids[index]),
            beam_type_key,
        )
        beam_records.setdefault(key, {}).setdefault(image, []).append(identifier)

    beam_owner: dict[int, tuple[int, FloatArray]] = {}
    owned_beam_ids: set[int] = set()
    for images in beam_records.values():
        multiplicities = {len(identifiers) for identifiers in images.values()}
        if len(multiplicities) != 1:
            raise ValueError(
                "Periodic images of a beam have inconsistent multiplicities; "
                "OpenSG cannot choose ownership without changing the model."
            )
        owner_image = min(images)
        owners = sorted(images[owner_image])
        owned_beam_ids.update(owners)
        for image, identifiers in images.items():
            translation = periods * (np.asarray(image) - np.asarray(owner_image))
            for identifier, owner in zip(sorted(identifiers), owners, strict=True):
                beam_owner[identifier] = (owner, translation)

    junction_records: dict[tuple, dict[tuple[int, int, int], list[int]]] = {}
    for identifier, instance in supplement.junction_instances.items():
        canonical_origin, image = _canonical_point(
            instance.origin, minimum, periods, tolerance
        )
        key = (
            _quantized(canonical_origin, tolerance),
            _quantized(instance.frame, tolerance),
            int(instance.junction_type_id),
        )
        junction_records.setdefault(key, {}).setdefault(image, []).append(identifier)

    junction_owner: dict[int, tuple[int, FloatArray]] = {}
    owned_junction_ids: set[int] = set()
    for images in junction_records.values():
        multiplicities = {len(identifiers) for identifiers in images.values()}
        if len(multiplicities) != 1:
            raise ValueError(
                "Periodic images of a junction have inconsistent multiplicities; "
                "use the same complete junction type on each represented image."
            )
        owner_image = min(images)
        owners = sorted(images[owner_image])
        owned_junction_ids.update(owners)
        for image, identifiers in images.items():
            translation = periods * (np.asarray(image) - np.asarray(owner_image))
            for identifier, owner in zip(sorted(identifiers), owners, strict=True):
                junction_owner[identifier] = (owner, translation)

    automatic_pairs = list(model.periodic_pairs)
    parent = list(range(len(model.nodes)))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def identify(first: int, second: int) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parent[first_root] = second_root

    for slave, master in automatic_pairs:
        identify(slave, master)
    for identifier, (owner, _translation) in beam_owner.items():
        if identifier == owner:
            continue
        image_nodes = model.connectivity[element_index[identifier]]
        owner_nodes = model.connectivity[element_index[owner]]
        for image_node, owner_node in zip(image_nodes, owner_nodes, strict=True):
            image_node, owner_node = int(image_node), int(owner_node)
            if root(image_node) != root(owner_node):
                automatic_pairs.append((image_node, owner_node))
                identify(image_node, owner_node)

    kept_indices = [
        index for index, identifier in enumerate(model.element_ids)
        if int(identifier) in owned_beam_ids
    ]
    owned_model = replace(
        model,
        element_ids=np.asarray(model.element_ids[kept_indices], dtype=np.int64),
        material_ids=np.asarray(model.material_ids[kept_indices], dtype=np.int64),
        connectivity=tuple(model.connectivity[index] for index in kept_indices),
        orientations={
            identifier: frame for identifier, frame in model.orientations.items()
            if identifier in owned_beam_ids
        },
        periodic_pairs=automatic_pairs,
    )
    owned_instances = {
        identifier: instance
        for identifier, instance in supplement.junction_instances.items()
        if identifier in owned_junction_ids
    }
    owned_assignments = {
        identifier: beam_type
        for identifier, beam_type in supplement.beam_assignments.items()
        if identifier in owned_beam_ids
    }
    owned_beam_type_ids = set(owned_assignments.values())
    owned_beam_types = {
        identifier: beam_type
        for identifier, beam_type in supplement.beam_types.items()
        if identifier in owned_beam_type_ids
    }
    owned_junction_type_ids = {
        instance.junction_type_id for instance in owned_instances.values()
    }
    owned_junction_types = {
        identifier: junction_type
        for identifier, junction_type in supplement.junction_types.items()
        if identifier in owned_junction_type_ids
    }

    owned_connections: dict[tuple[int, int], JunctionConnection] = {}
    for connection in supplement.junction_connections:
        if connection.element_id not in beam_owner:
            raise ValueError(
                f"Junction connection references undefined element {connection.element_id}."
            )
        if connection.junction_id not in junction_owner:
            raise ValueError(
                f"Connection references undefined junction {connection.junction_id}."
            )
        owner_element, beam_translation = beam_owner[connection.element_id]
        owner_junction, junction_translation = junction_owner[connection.junction_id]
        rewritten = JunctionConnection(
            junction_id=owner_junction,
            connection_point_id=connection.connection_point_id,
            element_id=owner_element,
            endpoint=connection.endpoint,
            image_shift=(
                np.asarray(connection.image_shift, dtype=float)
                + beam_translation - junction_translation
            ),
        )
        key = (owner_junction, connection.connection_point_id)
        previous = owned_connections.get(key)
        if previous is None:
            owned_connections[key] = rewritten
        elif not (
            previous.element_id == rewritten.element_id
            and previous.endpoint == rewritten.endpoint
            and np.allclose(
                previous.image_shift, rewritten.image_shift,
                rtol=0.0, atol=tolerance,
            )
        ):
            raise ValueError(
                f"Periodic junction images give conflicting ownership for junction "
                f"{owner_junction}, connection point {connection.connection_point_id}."
            )

    owned_supplement = replace(
        supplement,
        beam_types=owned_beam_types,
        beam_assignments=owned_assignments,
        junction_types=owned_junction_types,
        junction_instances=owned_instances,
        junction_connections=tuple(owned_connections.values()),
        beam_recovery={
            identifier: recovery
            for identifier, recovery in supplement.beam_recovery.items()
            if identifier in owned_beam_type_ids
        },
    )
    audit = PeriodicOwnershipAudit(
        input_beams=len(model.element_ids),
        owned_beams=len(owned_model.element_ids),
        input_junctions=len(supplement.junction_instances),
        owned_junctions=len(owned_instances),
        beam_owner_by_id={key: value[0] for key, value in beam_owner.items()},
        junction_owner_by_id={key: value[0] for key, value in junction_owner.items()},
    )
    return owned_model, owned_supplement, audit


def complete_periodic_connection_shifts(
    model: StructuralGenomeInput,
    supplement: HybridSupplement,
    stiffness_by_type: dict[int, JunctionStiffness],
) -> HybridSupplement:
    """Infer lattice-vector image shifts for owned boundary junctions."""

    _, periods, tolerance = _periodic_box(model.nodes)
    connectivity = {
        int(identifier): nodes
        for identifier, nodes in zip(model.element_ids, model.connectivity, strict=True)
    }
    completed: list[JunctionConnection] = []
    for connection in supplement.junction_connections:
        instance = supplement.junction_instances[connection.junction_id]
        stiffness = stiffness_by_type[instance.junction_type_id]
        points = {point.identifier: point for point in stiffness.connection_points}
        point = points[connection.connection_point_id]
        node_ids = connectivity[connection.element_id]
        node = int(node_ids[connection.endpoint - 1])
        expected = instance.origin + instance.frame.T @ point.origin
        actual = model.nodes[node] + connection.image_shift
        residual = expected - actual
        correction = np.zeros(3)
        valid = True
        for axis, period in enumerate(periods):
            if abs(residual[axis]) <= tolerance:
                continue
            if period <= tolerance:
                valid = False
                break
            multiple = int(np.rint(residual[axis] / period))
            if not np.isclose(
                residual[axis], multiple * period, rtol=0.0, atol=tolerance
            ):
                valid = False
                break
            correction[axis] = multiple * period
        completed.append(replace(
            connection,
            image_shift=(
                connection.image_shift + correction if valid
                else connection.image_shift
            ),
        ))
    return replace(supplement, junction_connections=tuple(completed))


def _validate_periodic_pair_geometry(
    nodes: FloatArray, pairs: list[tuple[int, int]]
) -> None:
    """Check explicit pairs for an axis-aligned periodic SG box."""

    if not pairs:
        return
    spans = np.ptp(nodes, axis=0)
    scale = max(float(np.max(spans)), 1.0)
    tolerance = 1.0e-8 * scale
    for slave, master in pairs:
        shift = nodes[slave] - nodes[master]
        if np.linalg.norm(shift) <= tolerance:
            raise ValueError(
                f"Periodic nodes {slave + 1} and {master + 1} have the same coordinates."
            )
        for axis, component in enumerate(shift):
            if abs(component) <= tolerance:
                continue
            if spans[axis] <= tolerance or not np.isclose(
                abs(component), spans[axis], rtol=1.0e-8, atol=tolerance
            ):
                raise ValueError(
                    f"Periodic pair ({slave + 1}, {master + 1}) has shift {shift}; "
                    f"component {axis + 1} must be zero or the SG span {spans[axis]:g}."
                )

    # Only explicitly supplied slave/master pairs require coincident periodic
    # images.  A complete beam or junction owned by one representative
    # boundary intentionally has no duplicate object (and therefore may have
    # no opposite-face node) on the omitted image boundary.


def read_structural_genome(path: str | Path) -> StructuralGenomeInput:
    """Read the mesh and standard control sections of an OpenSG input file."""

    path = Path(path)
    lines = _data_lines(path)
    if len(lines) < 2:
        raise ValueError(f"Structure-gene input is incomplete: {path}")
    try:
        control = [int(value) for value in lines[0].split()]
    except ValueError as error:
        raise ValueError(f"The control record in {path} must contain integers.") from error
    if len(control) not in (4, 5):
        raise ValueError("The first record requires four values and an optional junction flag.")
    analysis, element_flag, transformation_flag, temperature_flag = control[:4]
    junction_flag = control[4] if len(control) == 5 else 0
    if analysis not in (0,):
        raise ValueError(
            f"analysis={analysis} is not supported by the beam--junction driver; use 0."
        )
    if element_flag not in (0, 2):
        raise ValueError(
            f"elem_flag={element_flag} is invalid here; use 2 for the beam SG or "
            "0 for a referenced 3D junction solid."
        )
    if transformation_flag not in (0, 1):
        raise ValueError("trans_flag must be zero or one.")
    if temperature_flag not in (0,):
        raise ValueError(
            f"temp_flag={temperature_flag} is not supported for this analysis; use 0."
        )
    if junction_flag not in (0, 1, 2):
        raise ValueError("junction_flag must be zero, one, or two.")

    mesh_control = _integer_tokens(lines[1], 6, "Mesh control record")
    dimension, number_of_nodes, number_of_elements, number_of_materials, number_of_slaves, number_of_layers = mesh_control
    if dimension not in (1, 2, 3):
        raise ValueError(f"Spatial dimension must be 1, 2, or 3; received {dimension}.")
    for name, value in (
        ("number of nodes", number_of_nodes),
        ("number of elements", number_of_elements),
        ("number of materials", number_of_materials),
        ("number of periodic slaves", number_of_slaves),
        ("number of layers", number_of_layers),
    ):
        if value < 0:
            raise ValueError(f"The {name} cannot be negative.")
    if number_of_nodes == 0 or number_of_elements == 0:
        raise ValueError("The structure gene requires at least one node and one element.")
    cursor = 2
    _require_records(lines, cursor, number_of_nodes, "Node section", path)
    nodes = np.empty((number_of_nodes, 3), dtype=float)
    for expected_identifier in range(1, number_of_nodes + 1):
        values = lines[cursor].split()
        cursor += 1
        if len(values) != dimension + 1:
            raise ValueError(f"Node {expected_identifier} has an invalid coordinate record.")
        identifier = int(values[0])
        if identifier != expected_identifier:
            raise ValueError("Node numbers must be consecutive and ordered from one.")
        coordinates = np.asarray(values[1:], dtype=float)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"Node {identifier} contains a non-finite coordinate.")
        nodes[identifier - 1, :dimension] = coordinates
        if dimension < 3:
            nodes[identifier - 1, dimension:] = 0.0

    _require_records(lines, cursor, number_of_elements, "Element section", path)
    element_ids = np.empty(number_of_elements, dtype=np.int64)
    material_ids = np.empty(number_of_elements, dtype=np.int64)
    connectivity: list[IntArray] = []
    for element_index in range(number_of_elements):
        values = [int(value) for value in lines[cursor].split()]
        cursor += 1
        if len(values) < 4:
            raise ValueError(f"Element record {element_index + 1} is incomplete.")
        identifier, material_id = values[:2]
        if identifier != element_index + 1:
            raise ValueError("Element numbers must be consecutive and ordered from one.")
        raw_nodes = values[2:]
        if element_flag == 0 and len(raw_nodes) >= 11 and raw_nodes[4] == 0:
            node_values = raw_nodes[:4] + raw_nodes[5:11]
        else:
            node_values = [node for node in raw_nodes if node != 0]
        if element_flag == 2 and len(node_values) != 2:
            raise ValueError(
                f"Beam element {identifier} requires exactly two endpoint nodes; "
                "OpenSG creates the interpolation nodes internally."
            )
        if any(node < 1 or node > number_of_nodes for node in node_values):
            raise ValueError(f"Element {identifier} references an undefined node.")
        if len(node_values) != len(set(node_values)):
            raise ValueError(f"Element {identifier} repeats a node in its connectivity.")
        if material_id < 1 or material_id > number_of_materials:
            raise ValueError(
                f"Element {identifier} references material {material_id}, but the input "
                f"declares materials 1 through {number_of_materials}."
            )
        element_ids[element_index] = identifier
        material_ids[element_index] = material_id
        connectivity.append(np.asarray(node_values, dtype=np.int64) - 1)

    orientations: dict[int, FloatArray] = {}
    if transformation_flag:
        _require_records(lines, cursor, number_of_elements, "Orientation section", path)
        for _ in range(number_of_elements):
            values = lines[cursor].split()
            cursor += 1
            if len(values) != 10:
                raise ValueError("An element orientation record requires an ID and nine coordinates.")
            identifier = int(values[0])
            if identifier < 1 or identifier > number_of_elements:
                raise ValueError(f"Orientation references undefined element {identifier}.")
            if identifier in orientations:
                raise ValueError(f"Element {identifier} has multiple orientation records.")
            orientations[identifier] = frame_from_points(
                np.asarray(values[1:], dtype=float).reshape(3, 3)
            )
        missing = set(range(1, number_of_elements + 1)) - set(orientations)
        if missing:
            raise ValueError(f"Missing orientation records for elements {sorted(missing)}.")

    periodic_pairs: list[tuple[int, int]] = []
    _require_records(lines, cursor, number_of_slaves, "Periodic node section", path)
    slave_nodes: set[int] = set()
    pair_keys: set[frozenset[int]] = set()
    for _ in range(number_of_slaves):
        slave, master = _integer_tokens(lines[cursor], 2, "Periodic node record")
        cursor += 1
        if not 1 <= slave <= number_of_nodes or not 1 <= master <= number_of_nodes:
            raise ValueError(
                f"Periodic pair ({slave}, {master}) references a node outside "
                f"1 through {number_of_nodes}."
            )
        if slave == master:
            raise ValueError(f"Periodic pair ({slave}, {master}) maps a node to itself.")
        if slave in slave_nodes:
            raise ValueError(f"Periodic slave node {slave} is listed more than once.")
        key = frozenset((slave, master))
        if key in pair_keys:
            raise ValueError(f"Periodic node pair ({slave}, {master}) is duplicated.")
        slave_nodes.add(slave)
        pair_keys.add(key)
        periodic_pairs.append((slave - 1, master - 1))
    _validate_periodic_pair_geometry(nodes, periodic_pairs)
    _require_records(lines, cursor, number_of_layers, "Layer section", path)
    cursor += number_of_layers
    if cursor > len(lines):
        raise ValueError("The structure-gene input ends before its layer records.")

    tail_records = tuple(lines[cursor:])
    volume = _read_volume(tail_records)
    return StructuralGenomeInput(
        path=path,
        analysis=analysis,
        element_flag=element_flag,
        transformation_flag=transformation_flag,
        temperature_flag=temperature_flag,
        junction_flag=junction_flag,
        dimension=dimension,
        nodes=nodes,
        element_ids=element_ids,
        material_ids=material_ids,
        connectivity=tuple(connectivity),
        orientations=orientations,
        periodic_pairs=periodic_pairs,
        number_of_materials=number_of_materials,
        tail_records=tail_records,
        volume=volume,
    )


def _read_volume(records: tuple[str, ...]) -> float:
    for line in reversed(records):
        values = line.split()
        if len(values) == 1:
            try:
                volume = float(values[0])
            except ValueError:
                continue
            if volume > 0.0:
                return volume
    raise ValueError("The structure-gene volume was not found.")


def read_isotropic_materials(model: StructuralGenomeInput) -> dict[int, tuple[float, float]]:
    """Read isotropic elastic properties from standard material records."""

    records = list(model.tail_records)
    materials: dict[int, tuple[float, float]] = {}
    cursor = 0
    while cursor < len(records) and len(materials) < model.number_of_materials:
        values = records[cursor].split()
        if len(values) == 3:
            try:
                identifier, isotropy, number_of_temperatures = map(int, values)
            except ValueError:
                cursor += 1
                continue
            if isotropy == 0 and number_of_temperatures > 0:
                if number_of_temperatures != 1:
                    raise ValueError(
                        "Solid junction analysis currently requires one elastic temperature point."
                    )
                final_property_record = cursor + 2 * number_of_temperatures
                if final_property_record >= len(records):
                    raise ValueError(f"Material {identifier} record is incomplete.")
                properties = records[cursor + 2].split()
                if len(properties) < 2:
                    raise ValueError(f"Material {identifier} elastic record is incomplete.")
                engineering_e, poisson_nu = map(float, properties[:2])
                if identifier < 1 or identifier > model.number_of_materials:
                    raise ValueError(
                        f"Material identifier {identifier} lies outside 1 through "
                        f"{model.number_of_materials}."
                    )
                if identifier in materials:
                    raise ValueError(f"Material {identifier} is defined more than once.")
                if not np.isfinite(engineering_e) or engineering_e <= 0.0:
                    raise ValueError(f"Material {identifier} Young's modulus must be positive.")
                if not np.isfinite(poisson_nu) or not -1.0 < poisson_nu < 0.5:
                    raise ValueError(
                        f"Material {identifier} Poisson's ratio must satisfy -1 < nu < 0.5."
                    )
                materials[identifier] = (engineering_e, poisson_nu)
                cursor = final_property_record + 1
                continue
        cursor += 1
    if len(materials) != model.number_of_materials:
        raise ValueError("Only standard isotropic material records are supported for junction solids.")
    return materials


def read_hybrid_supplement(
    path: str | Path,
    base_directory: str | Path | None = None,
) -> HybridSupplement:
    """Read beam properties and junction topology from ``.sc.msg``."""

    path = Path(path)
    base = path.parent if base_directory is None else Path(base_directory)
    lines = _data_lines(path)
    if not lines:
        raise ValueError(f"Hybrid supplement is empty: {path}")
    header = _integer_tokens(lines[0], 7, "Hybrid supplement header")
    version, number_of_beam_types, number_of_assignments, number_of_junction_types, number_of_junctions, number_of_connections, junction_transformation_flag = header
    if version != 1:
        raise ValueError(f"Unsupported hybrid supplement version: {version}")
    if junction_transformation_flag not in (0, 1):
        raise ValueError("The junction transformation flag must be zero or one.")
    for name, value in (
        ("beam-type", number_of_beam_types),
        ("beam-assignment", number_of_assignments),
        ("junction-type", number_of_junction_types),
        ("junction-instance", number_of_junctions),
        ("junction-connection", number_of_connections),
    ):
        if value < 0:
            raise ValueError(f"The {name} count cannot be negative in {path}.")
    if number_of_beam_types == 0:
        raise ValueError(f"At least one beam type must be defined in {path}.")
    cursor = 1

    beam_types: dict[int, BeamType] = {}
    for _ in range(number_of_beam_types):
        _require_records(lines, cursor, 1, "Beam-type section", path)
        identifier, theory_value, number_of_nodes, number_of_strains = _integer_tokens(
            lines[cursor], 4, "Beam type record"
        )
        cursor += 1
        if identifier <= 0:
            raise ValueError("Beam-type identifiers must be positive.")
        if number_of_strains <= 0:
            raise ValueError(f"Beam type {identifier} must have at least one strain.")
        _require_records(
            lines, cursor, number_of_strains, f"Beam type {identifier} stiffness", path
        )
        try:
            theory = BeamTheory(theory_value)
        except ValueError as error:
            raise ValueError(f"Unsupported beam theory: {theory_value}") from error
        matrix = np.asarray(
            [[float(value) for value in lines[cursor + row].split()] for row in range(number_of_strains)],
            dtype=float,
        )
        cursor += number_of_strains
        if matrix.shape != (number_of_strains, number_of_strains):
            raise ValueError(f"Beam type {identifier} has an invalid section stiffness.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Beam type {identifier} section stiffness is not finite.")
        scale = max(float(np.linalg.norm(matrix)), 1.0)
        if np.linalg.norm(matrix - matrix.T) / scale > 1.0e-10:
            raise ValueError(f"Beam type {identifier} section stiffness is not symmetric.")
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        if np.min(eigenvalues) <= 1.0e-12 * max(float(np.max(np.abs(eigenvalues))), 1.0):
            raise ValueError(
                f"Beam type {identifier} section stiffness must be positive definite."
            )
        if theory == BeamTheory.EULER_BERNOULLI and (
            number_of_nodes, number_of_strains
        ) != (3, 4):
            raise ValueError(
                "Euler--Bernoulli beam types require three generated interpolation "
                "nodes and four strains."
            )
        if theory == BeamTheory.TIMOSHENKO and (number_of_nodes, number_of_strains) != (4, 6):
            raise ValueError("Timoshenko beam types require four nodes and six strains.")
        if identifier in beam_types:
            raise ValueError(f"Beam type {identifier} is defined more than once.")
        beam_types[identifier] = BeamType(
            identifier, theory, number_of_nodes, 0.5 * (matrix + matrix.T)
        )

    assignments: dict[int, int] = {}
    _require_records(lines, cursor, number_of_assignments, "Beam-assignment section", path)
    for _ in range(number_of_assignments):
        element_id, beam_type_id = _integer_tokens(lines[cursor], 2, "Beam assignment")
        cursor += 1
        if element_id <= 0 or beam_type_id <= 0:
            raise ValueError("Beam assignments require positive element and beam-type IDs.")
        if element_id in assignments:
            raise ValueError(f"Element {element_id} has multiple beam assignments.")
        assignments[element_id] = beam_type_id

    junction_types: dict[int, JunctionType] = {}
    _require_records(lines, cursor, number_of_junction_types, "Junction-type section", path)
    for _ in range(number_of_junction_types):
        values = shlex.split(lines[cursor])
        cursor += 1
        if len(values) != 3:
            raise ValueError("A junction type requires: id nconnectionpoint source.")
        identifier, connection_point_count = map(int, values[:2])
        if identifier <= 0:
            raise ValueError("Junction-type identifiers must be positive.")
        if connection_point_count <= 0:
            raise ValueError(
                f"Junction type {identifier} must declare at least one connection point."
            )
        source = Path(values[2])
        if not source.is_absolute():
            source = base / source
        if identifier in junction_types:
            raise ValueError(f"Junction type {identifier} is defined more than once.")
        junction_types[identifier] = JunctionType(
            identifier, connection_point_count, source
        )

    instance_records: list[tuple[int, int, FloatArray]] = []
    _require_records(lines, cursor, number_of_junctions, "Junction-instance section", path)
    for _ in range(number_of_junctions):
        values = lines[cursor].split()
        cursor += 1
        if len(values) != 5:
            raise ValueError("A junction instance requires: id type_id x y z.")
        identifier, junction_type_id = map(int, values[:2])
        origin = np.asarray(values[2:], dtype=float)
        if identifier <= 0 or junction_type_id <= 0:
            raise ValueError("Junction instances require positive junction and type IDs.")
        if not np.all(np.isfinite(origin)):
            raise ValueError(f"Junction {identifier} origin is not finite.")
        instance_records.append((identifier, junction_type_id, origin))

    instance_frames: dict[int, FloatArray] = {}
    if junction_transformation_flag:
        _require_records(
            lines, cursor, number_of_junctions, "Junction orientation section", path
        )
        instance_origins = {identifier: origin for identifier, _, origin in instance_records}
        for _ in range(number_of_junctions):
            values = lines[cursor].split()
            cursor += 1
            if len(values) != 10:
                raise ValueError("A junction orientation requires an ID and nine coordinates.")
            identifier = int(values[0])
            points = np.asarray(values[1:], dtype=float).reshape(3, 3)
            if identifier not in instance_origins:
                raise ValueError(f"Orientation references undefined junction {identifier}.")
            if not np.allclose(points[0], instance_origins[identifier], atol=1.0e-12):
                raise ValueError(
                    f"Junction {identifier} orientation origin differs from its instance origin."
                )
            if identifier in instance_frames:
                raise ValueError(f"Junction {identifier} has multiple orientation records.")
            instance_frames[identifier] = frame_from_points(points)

    junction_instances: dict[int, JunctionInstance] = {}
    for identifier, junction_type_id, origin in instance_records:
        if identifier in junction_instances:
            raise ValueError(f"Junction {identifier} is defined more than once.")
        junction_instances[identifier] = JunctionInstance(
            identifier=identifier,
            junction_type_id=junction_type_id,
            origin=origin,
            frame=instance_frames.get(identifier, np.eye(3)),
        )

    connections: list[JunctionConnection] = []
    _require_records(lines, cursor, number_of_connections, "Junction-connection section", path)
    for _ in range(number_of_connections):
        values = lines[cursor].split()
        cursor += 1
        if len(values) != 7:
            raise ValueError(
                "A junction connection requires: junction_id connection_point_id "
                "element_id endpoint shift_x shift_y shift_z."
            )
        image_shift = np.asarray(values[4:], dtype=float)
        if not np.all(np.isfinite(image_shift)):
            raise ValueError(
                f"Junction {values[0]}, connection point {values[1]} has a non-finite "
                "image shift."
            )
        connections.append(
            JunctionConnection(
                junction_id=int(values[0]),
                connection_point_id=int(values[1]),
                element_id=int(values[2]),
                endpoint=int(values[3]),
                image_shift=image_shift,
            )
        )
    beam_recovery: dict[int, BeamRecovery] = {}
    while cursor < len(lines):
        values = shlex.split(lines[cursor])
        cursor += 1
        if not values:
            continue
        keyword = values[0].upper()
        if keyword == "BEAM_RECOVERY":
            if len(values) != 3:
                raise ValueError(
                    "BEAM_RECOVERY requires: beam_type_id vabs_section_source."
                )
            beam_type_id = int(values[1])
            source = Path(values[2])
            if not source.is_absolute():
                source = base / source
            if beam_type_id in beam_recovery:
                raise ValueError(
                    f"Beam type {beam_type_id} has multiple recovery records."
                )
            beam_recovery[beam_type_id] = BeamRecovery(beam_type_id, source)
        else:
            raise ValueError(
                f"Unexpected recovery record {values[0]!r} at the end of {path}."
            )
    return HybridSupplement(
        version=version,
        beam_types=beam_types,
        beam_assignments=assignments,
        junction_types=junction_types,
        junction_instances=junction_instances,
        junction_connections=tuple(connections),
        beam_recovery=beam_recovery,
    )


def validate_hybrid_input(
    model: StructuralGenomeInput, supplement: HybridSupplement
) -> None:
    """Validate relationships that span the main and supplement files."""

    element_ids = {int(identifier) for identifier in model.element_ids}
    assignment_ids = set(supplement.beam_assignments)
    missing_assignments = element_ids - assignment_ids
    extra_assignments = assignment_ids - element_ids
    if missing_assignments:
        raise ValueError(
            f"Elements {sorted(missing_assignments)} have no beam-type assignment."
        )
    if extra_assignments:
        raise ValueError(
            f"Beam assignments reference undefined elements {sorted(extra_assignments)}."
        )
    undefined_beam_types = {
        beam_type_id
        for beam_type_id in supplement.beam_assignments.values()
        if beam_type_id not in supplement.beam_types
    }
    if undefined_beam_types:
        raise ValueError(
            f"Beam assignments reference undefined beam types {sorted(undefined_beam_types)}."
        )
    unused_beam_types = set(supplement.beam_types) - set(
        supplement.beam_assignments.values()
    )
    if unused_beam_types:
        raise ValueError(f"Beam types {sorted(unused_beam_types)} are defined but unused.")

    used_nodes = set(np.concatenate(model.connectivity).tolist())
    unused_nodes = set(range(len(model.nodes))) - used_nodes
    # A boundary-ownership SG may retain a node only as the periodic image of
    # an active owner node after the duplicate boundary beam is removed. Such
    # a ghost has no independent DOF after periodic reduction and must not
    # force assembly of a second full-section beam merely to count as "used".
    representatives = _periodic_representatives(
        len(model.nodes), model.periodic_pairs
    )
    active_representatives = {representatives[node] for node in used_nodes}
    invalid_unused = {
        node for node in unused_nodes
        if representatives[node] not in active_representatives
    }
    if invalid_unused:
        raise ValueError(
            "The following nodes are not used by any element: "
            f"{[node + 1 for node in sorted(invalid_unused)]}."
        )

    if model.junction_flag == 0:
        counts = (
            len(supplement.junction_types),
            len(supplement.junction_instances),
            len(supplement.junction_connections),
        )
        if any(counts):
            raise ValueError(
                "junction_flag=0 selects a pure-beam model, but the supplement defines "
                f"{counts[0]} junction types, {counts[1]} junction instances, and "
                f"{counts[2]} junction connections. Remove those records or use flag 1 or 2."
            )
        if not model.periodic_pairs:
            raise ValueError(
                "junction_flag=0 requires periodic slave/master node records; none were supplied."
            )
    else:
        if not supplement.junction_types:
            raise ValueError(
                f"junction_flag={model.junction_flag} requires at least one junction type."
            )
        if not supplement.junction_instances:
            raise ValueError(
                f"junction_flag={model.junction_flag} requires at least one junction instance."
            )
        if not supplement.junction_connections:
            raise ValueError(
                f"junction_flag={model.junction_flag} requires junction connection records."
            )
        if not model.periodic_pairs and not any(
            np.linalg.norm(connection.image_shift) > 0.0
            for connection in supplement.junction_connections
        ):
            raise ValueError(
                "The hybrid SG has neither periodic node pairs nor nonzero junction image "
                "shifts, so periodicity has not been defined."
            )

    translations = [model.nodes[slave] - model.nodes[master] for slave, master in model.periodic_pairs]
    translations.extend(
        connection.image_shift
        for connection in supplement.junction_connections
        if np.linalg.norm(connection.image_shift) > 0.0
    )
    translation_matrix = np.asarray(translations, dtype=float).reshape((-1, 3))
    scale = max(float(np.linalg.norm(translation_matrix)), 1.0)
    translation_rank = int(np.linalg.matrix_rank(translation_matrix, tol=1.0e-10 * scale))
    if translation_rank < 3:
        raise ValueError(
            "The periodic node shifts and junction image shifts span only "
            f"{translation_rank} independent directions; a 3D periodic SG requires three."
        )

    instance_types = {
        instance.junction_type_id for instance in supplement.junction_instances.values()
    }
    undefined_junction_types = instance_types - set(supplement.junction_types)
    if undefined_junction_types:
        raise ValueError(
            f"Junction instances reference undefined types {sorted(undefined_junction_types)}."
        )
    unused_junction_types = set(supplement.junction_types) - instance_types
    if unused_junction_types:
        raise ValueError(
            f"Junction types {sorted(unused_junction_types)} are defined but unused."
        )

    connections_by_junction: dict[int, list[JunctionConnection]] = {}
    for connection in supplement.junction_connections:
        connections_by_junction.setdefault(connection.junction_id, []).append(connection)
    undefined_junctions = set(connections_by_junction) - set(supplement.junction_instances)
    if undefined_junctions:
        raise ValueError(
            f"Connections reference undefined junctions {sorted(undefined_junctions)}."
        )
    for junction_id, instance in supplement.junction_instances.items():
        junction_type = supplement.junction_types.get(instance.junction_type_id)
        if junction_type is None:
            continue
        connections = connections_by_junction.get(junction_id, [])
        connection_point_ids = [
            connection.connection_point_id for connection in connections
        ]
        expected = set(range(1, junction_type.number_of_connection_points + 1))
        if len(connection_point_ids) != len(set(connection_point_ids)):
            raise ValueError(
                f"Junction {junction_id} uses one of its connection points more than once."
            )
        if set(connection_point_ids) != expected:
            missing = sorted(expected - set(connection_point_ids))
            extra = sorted(set(connection_point_ids) - expected)
            raise ValueError(
                f"Junction {junction_id} requires connection points 1 through "
                f"{junction_type.number_of_connection_points}; missing={missing}, "
                f"invalid={extra}."
            )

    incidence: dict[int, list[tuple[int, bool]]] = {node: [] for node in range(len(model.nodes))}
    for element_id, connectivity in zip(model.element_ids, model.connectivity, strict=True):
        beam_type_id = supplement.beam_assignments[int(element_id)]
        for local_index, node in enumerate(connectivity):
            incidence[int(node)].append((beam_type_id, local_index < 2))
    # Do not require identical unreduced connectivity at periodic image nodes.
    # Under explicit boundary ownership, the complete beam is assembled on
    # one representative face only; its opposite-face node remains tied by P
    # but intentionally lacks the duplicate member. Requiring equal incidence
    # here forces exactly the double energy contribution ownership prevents.


def build_beam_discretization(
    model: StructuralGenomeInput,
    supplement: HybridSupplement,
) -> tuple[FloatArray, tuple[BeamElement, ...]]:
    """Create internal beam nodes and combine all element input records."""

    if model.element_flag != 2:
        raise ValueError("Beam homogenization requires elem_flag=2.")
    nodes = [np.asarray(node, dtype=float).copy() for node in model.nodes]
    elements: list[BeamElement] = []
    for identifier, node_ids in zip(model.element_ids, model.connectivity, strict=True):
        element_id = int(identifier)
        if element_id not in supplement.beam_assignments:
            raise ValueError(f"Element {element_id} has no beam-type assignment.")
        beam_type_id = supplement.beam_assignments[element_id]
        beam_type = supplement.beam_types.get(beam_type_id)
        if beam_type is None:
            raise ValueError(f"Element {element_id} references undefined beam type {beam_type_id}.")
        if len(node_ids) != 2:
            raise ValueError(
                f"Beam element {element_id} requires two endpoint nodes, "
                f"received {len(node_ids)}."
            )
        frame = model.orientations.get(element_id)
        if frame is None:
            start, end = model.nodes[node_ids[0]], model.nodes[node_ids[1]]
            trial = np.array([0.0, 0.0, 1.0])
            if abs(np.dot((end - start) / np.linalg.norm(end - start), trial)) > 0.9:
                trial = np.array([0.0, 1.0, 0.0])
            frame = beam_frame(start, end, trial)
        point_1 = model.nodes[node_ids[0]]
        point_2 = model.nodes[node_ids[1]]
        if beam_type.theory == BeamTheory.EULER_BERNOULLI:
            midpoint = len(nodes)
            nodes.append(0.5 * (point_1 + point_2))
            internal_node_ids = np.asarray(
                [node_ids[0], node_ids[1], midpoint], dtype=np.int64
            )
        elif beam_type.theory == BeamTheory.TIMOSHENKO:
            inner_1 = len(nodes)
            nodes.append((2.0 * point_1 + point_2) / 3.0)
            inner_2 = len(nodes)
            nodes.append((point_1 + 2.0 * point_2) / 3.0)
            internal_node_ids = np.asarray(
                [node_ids[0], node_ids[1], inner_1, inner_2], dtype=np.int64
            )
        else:
            raise ValueError(f"Unsupported beam theory: {beam_type.theory}")
        elements.append(
            BeamElement(element_id, internal_node_ids, beam_type_id, frame)
        )
    return np.asarray(nodes), tuple(elements)


def read_solid_junction(path: str | Path) -> SolidJunctionModel:
    """Read a 3D solid junction and its connection-point/interface file."""

    model = read_structural_genome(path)
    if model.element_flag != 0 or model.dimension != 3:
        raise ValueError("A solid junction source must be a 3D solid structure gene.")
    if model.junction_flag != 0:
        raise ValueError("A referenced solid junction .sc file must use junction_flag=0.")
    unsupported_sizes = sorted(
        {len(connectivity) for connectivity in model.connectivity} - {4, 10}
    )
    if unsupported_sizes:
        raise ValueError(
            "A solid junction source supports TET4 and TET10 elements; "
            f"received element sizes {unsupported_sizes}."
        )
    junction_data_path = Path(str(model.path) + ".msg")
    lines = _data_lines(junction_data_path)
    if not lines:
        raise ValueError(
            f"Solid junction connection-point/interface file is empty: {junction_data_path}"
        )
    version, number_of_connection_points = _integer_tokens(
        lines[0], 2, "Solid junction connection-point header"
    )
    if version != 1:
        raise ValueError(
            f"Unsupported solid junction connection-point/interface version: {version}"
        )
    if number_of_connection_points <= 0:
        raise ValueError("A solid junction requires at least one connection point.")
    connection_points: list[JunctionConnectionPoint] = []
    interfaces: list[IntArray] = []
    cursor = 1
    assigned_faces: set[frozenset[int]] = set()
    for expected_identifier in range(1, number_of_connection_points + 1):
        _require_records(
            lines, cursor, 1, "Solid junction connection-point section", junction_data_path
        )
        values = lines[cursor].split()
        cursor += 1
        if len(values) != 14:
            raise ValueError("A solid junction connection-point header is incomplete.")
        identifier = int(values[0])
        number_of_faces = int(values[1])
        if identifier != expected_identifier:
            raise ValueError(
                "Solid junction connection points must be ordered consecutively from one."
            )
        if number_of_faces <= 0:
            raise ValueError(
                f"Connection point {identifier} must have at least one interface face."
            )
        numbers = np.asarray(values[2:14], dtype=float)
        if not np.all(np.isfinite(numbers)):
            raise ValueError(
                f"Connection point {identifier} contains non-finite geometry data."
            )
        frame = numbers[3:].reshape(3, 3)
        if not np.allclose(frame @ frame.T, np.eye(3), atol=1.0e-9):
            raise ValueError(f"Connection point {identifier} frame is not orthonormal.")
        if np.linalg.det(frame) <= 0.0:
            raise ValueError(f"Connection point {identifier} frame is not right-handed.")
        _require_records(
            lines,
            cursor,
            number_of_faces,
            f"Connection point {identifier} interface section",
            junction_data_path,
        )
        face_records = [
            [int(value) for value in lines[cursor + face].split()]
            for face in range(number_of_faces)
        ]
        cursor += number_of_faces
        face_sizes = {len(face) for face in face_records}
        if len(face_sizes) != 1 or next(iter(face_sizes)) not in (3, 6):
            raise ValueError(
                f"Connection point {identifier} interface must consistently use TRI3 or TRI6 "
                "connectivity."
            )
        faces = np.asarray(face_records, dtype=np.int64) - 1
        if np.any(faces < 0) or np.any(faces >= len(model.nodes)):
            raise ValueError(
                f"Connection point {identifier} interface references an undefined solid node."
            )
        for face_index, face in enumerate(faces, start=1):
            if len(set(face.tolist())) != len(face):
                raise ValueError(
                    f"Connection point {identifier}, interface face {face_index} repeats a node."
                )
            key = frozenset(int(node) for node in face)
            if key in assigned_faces:
                raise ValueError(
                    f"Connection point {identifier}, interface face {face_index} is assigned more "
                    "than once."
                )
            assigned_faces.add(key)
        origin = numbers[:3]
        face_nodes = np.unique(faces)
        scale = max(float(np.ptp(model.nodes[face_nodes], axis=0).max()), 1.0)
        normal_coordinates = (frame @ (model.nodes[face_nodes] - origin).T)[0]
        plane_error = float(np.max(np.abs(normal_coordinates)))
        if plane_error > 1.0e-8 * scale:
            raise ValueError(
                f"Connection point {identifier} interface does not lie in its plane; "
                f"maximum normal offset={plane_error:g}."
            )
        connection_points.append(JunctionConnectionPoint(identifier, origin, frame))
        interfaces.append(faces)
    if cursor != len(lines):
        raise ValueError(f"Unexpected records at the end of {junction_data_path}.")
    _validate_solid_interfaces(model.connectivity, assigned_faces)
    return SolidJunctionModel(
        nodes=model.nodes,
        elements=tuple(model.connectivity),
        material_ids=model.material_ids,
        materials=read_isotropic_materials(model),
        connection_points=tuple(connection_points),
        interface_faces=tuple(interfaces),
    )


def _validate_solid_interfaces(
    connectivity: tuple[IntArray, ...], assigned_faces: set[frozenset[int]]
) -> None:
    """Require every declared interface face to be an exterior solid face."""

    local_faces_by_size = {
        4: ((0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        10: (
            (0, 1, 2, 4, 5, 6),
            (0, 1, 3, 4, 8, 7),
            (1, 2, 3, 5, 9, 8),
            (2, 0, 3, 6, 7, 9),
        ),
    }
    occurrences: dict[frozenset[int], int] = {}
    for element in connectivity:
        local_faces = local_faces_by_size.get(len(element))
        if local_faces is None:
            raise ValueError(
                f"Interface validation does not support {len(element)}-node solid "
                "elements."
            )
        for local_face in local_faces:
            key = frozenset(int(element[index]) for index in local_face)
            occurrences[key] = occurrences.get(key, 0) + 1
    for face in assigned_faces:
        count = occurrences.get(face, 0)
        if count == 0:
            raise ValueError(
                "A declared junction interface face is not a face of any solid element."
            )
        if count != 1:
            raise ValueError(
                "A declared junction interface face is internal rather than exterior."
            )
