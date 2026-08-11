"""Validate standardized OpenSG beam input against the BCCH research scripts."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TIMOSHENKO = ROOT / "Timoshenko"
if str(TIMOSHENKO) not in sys.path:
    sys.path.insert(0, str(TIMOSHENKO))

import Timoshenko_cross as timoshenko_reference
import Timoshenko_cross_3Djunc_ML_angle as hybrid_reference
import bcch_lambda06_benchmark as benchmark
import bcch_resolved_homogenization as resolved_reference
from bcch_resolved_junctions import frames as junction_frames
from tet10_boolean_backend import (
    build_boolean_tet10_connector_mesh,
    build_tet10_kinematic_connection_stiffness,
)

from fe_jax.hybrid_homogenization import homogenize
from fe_jax.junction import (
    JunctionConnectionPoint,
    JunctionStiffness,
    write_junction_stiffness,
)


def _read_swiftcomp_stiffness() -> np.ndarray:
    path = TIMOSHENKO / "BCCH_nSG3_3D_C3D10_periodic_h010.sc.k"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(
        index for index, line in enumerate(lines)
        if "The Effective Stiffness Matrix" in line
    )
    rows = []
    for line in lines[header + 1 :]:
        values = line.split()
        if len(values) == 6:
            try:
                rows.append([float(value) for value in values])
            except ValueError:
                continue
            if len(rows) == 6:
                return np.asarray(rows)
    raise ValueError(f"Effective stiffness matrix was not found in {path}.")


def _write_rigid_input(directory: Path) -> Path:
    geometry = benchmark.build_geometry()
    transverse_directions = benchmark.beam_frames(geometry.nodes, geometry.beams)
    nodes = np.asarray(geometry.nodes)
    elements = np.asarray(geometry.beams, dtype=np.int64)
    periodic_pairs = timoshenko_reference.detect_periodic_pairs(nodes)
    input_path = directory / "bcch_rigid.sc"
    lines = [
        "0 2 1 0 0",
        "",
        f"3 {len(nodes)} {len(elements)} 1 {len(periodic_pairs)} 0",
        "",
    ]
    lines.extend(
        f"{identifier} " + " ".join(f"{value:.16e}" for value in point)
        for identifier, point in enumerate(nodes, start=1)
    )
    lines.append("")
    lines.extend(
        f"{identifier} 1 " + " ".join(str(int(node) + 1) for node in element)
        for identifier, element in enumerate(elements, start=1)
    )
    lines.append("")
    for identifier, (element, transverse) in enumerate(
        zip(elements, transverse_directions, strict=True), start=1
    ):
        start = nodes[element[0]]
        end = nodes[element[1]]
        transverse_point = start + transverse
        values = np.concatenate((start, end, transverse_point))
        lines.append(
            f"{identifier} " + " ".join(f"{value:.16e}" for value in values)
        )
    lines.append("")
    lines.extend(f"{slave + 1} {master + 1}" for slave, master in periodic_pairs)
    lines.extend(
        (
            "",
            "1 0 1",
            "0.0 1.0",
            f"{benchmark.MATERIAL_E:.16e} {benchmark.MATERIAL_NU:.16e}",
            "",
            f"{benchmark.CELL_SIDE**3:.16e}",
        )
    )
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    section = benchmark.section_matrix(
        geometry.radius, benchmark.MATERIAL_E, benchmark.MATERIAL_NU
    )
    supplement = [f"1 1 {len(elements)} 0 0 0 0", "", "1 1 4 6"]
    supplement.extend(
        " ".join(f"{value:.16e}" for value in row) for row in section
    )
    supplement.append("")
    supplement.extend(f"{identifier} 1" for identifier in range(1, len(elements) + 1))
    Path(str(input_path) + ".msg").write_text(
        "\n".join(supplement) + "\n", encoding="utf-8"
    )
    return input_path


def validate_rigid() -> dict[str, float]:
    """Compare complete rigid-junction stiffness matrices."""

    _, reference = benchmark.solve()
    with tempfile.TemporaryDirectory() as directory:
        input_path = _write_rigid_input(Path(directory))
        _, _, standardized = homogenize(input_path)
    difference = standardized.effective_stiffness - reference
    swiftcomp = _read_swiftcomp_stiffness()
    swiftcomp_difference = standardized.effective_stiffness - swiftcomp
    swiftcomp_ex = float(benchmark.engineering_constants(swiftcomp)[0])
    standardized_ex = float(
        benchmark.engineering_constants(standardized.effective_stiffness)[0]
    )
    return {
        "relative_matrix_error": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "reference_ex": float(benchmark.engineering_constants(reference)[0]),
        "standardized_ex": standardized_ex,
        "swiftcomp_relative_matrix_error": float(
            np.linalg.norm(swiftcomp_difference) / np.linalg.norm(swiftcomp)
        ),
        "swiftcomp_maximum_absolute_error": float(np.max(np.abs(swiftcomp_difference))),
        "swiftcomp_ex": swiftcomp_ex,
        "swiftcomp_ex_error_percent": 100.0 * (standardized_ex / swiftcomp_ex - 1.0),
    }


def _resolved_junction_data(geometry, mesh_factor: float = 0.4):
    groups = resolved_reference.junction_groups(geometry)
    junctions = []
    branch_records = []
    for name, items, distance_factor, representative, shifts in groups:
        distance = distance_factor * geometry.radius
        records = []
        directions = []
        for item in items:
            node, _other, direction = resolved_reference.endpoint_data(geometry, item)
            wrapped = geometry.nodes[node] + distance * direction
            unwrapped = wrapped + shifts[node]
            records.append((item, wrapped, unwrapped, direction))
            directions.append(direction)
        connection_frames = junction_frames(np.asarray(directions))
        mesh = build_boolean_tet10_connector_mesh(
            connection_frames,
            geometry.radius,
            distance,
            mesh_factor * geometry.radius,
            interface_tolerance=1.0e-6,
            cross_section="circular",
            circular_back_extension=0.0,
        )
        result = build_tet10_kinematic_connection_stiffness(
            mesh,
            benchmark.MATERIAL_E,
            benchmark.MATERIAL_NU,
            length_scale=2.0 * geometry.radius,
        )
        junctions.append(
            {
                "name": name,
                "representative": representative,
                "distance": distance,
                "frames": connection_frames,
                "matrix": result.k_j,
                "records": records,
            }
        )
        branch_records.extend(records)
    return junctions, branch_records


def _write_resolved_input(directory: Path, geometry, junctions, branch_records) -> Path:
    connection_nodes = [record[1] for record in branch_records]
    transverse_directions = []
    midpoint_indices = {}
    nodes = list(connection_nodes)
    half_beams = []
    for branch, (item, _wrapped, _unwrapped, direction) in enumerate(branch_records):
        beam, _endpoint = item
        trial = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        transverse = trial - np.dot(trial, direction) * direction
        transverse_directions.append(transverse / np.linalg.norm(transverse))
        if beam not in midpoint_indices:
            midpoint_indices[beam] = len(nodes)
            n1, n2 = geometry.beams[beam]
            nodes.append(0.5 * (geometry.nodes[n1] + geometry.nodes[n2]))
        half_beams.append((branch, midpoint_indices[beam]))
    nodes = np.asarray(nodes)
    elements = np.asarray(half_beams, dtype=np.int64)

    input_path = directory / "bcch_resolved.sc"
    lines = [
        "0 2 1 0 2",
        "",
        f"3 {len(nodes)} {len(elements)} 1 0 0",
        "",
    ]
    lines.extend(
        f"{identifier} " + " ".join(f"{value:.16e}" for value in point)
        for identifier, point in enumerate(nodes, start=1)
    )
    lines.append("")
    lines.extend(
        f"{identifier} 1 " + " ".join(str(int(node) + 1) for node in element)
        for identifier, element in enumerate(elements, start=1)
    )
    lines.append("")
    for identifier, (element, transverse) in enumerate(
        zip(elements, transverse_directions, strict=True), start=1
    ):
        start = nodes[element[0]]
        end = nodes[element[1]]
        values = np.concatenate((start, end, start + transverse))
        lines.append(f"{identifier} " + " ".join(f"{value:.16e}" for value in values))
    lines.extend(
        (
            "",
            "1 0 1",
            "0.0 1.0",
            f"{benchmark.MATERIAL_E:.16e} {benchmark.MATERIAL_NU:.16e}",
            "",
            f"{benchmark.CELL_SIDE**3:.16e}",
        )
    )
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    section = benchmark.section_matrix(
        geometry.radius, benchmark.MATERIAL_E, benchmark.MATERIAL_NU
    )
    supplement = [
        f"1 1 {len(elements)} {len(junctions)} {len(junctions)} {len(branch_records)} 0",
        "",
        "1 1 4 6",
    ]
    supplement.extend(" ".join(f"{value:.16e}" for value in row) for row in section)
    supplement.append("")
    supplement.extend(f"{identifier} 1" for identifier in range(1, len(elements) + 1))
    supplement.append("")
    for junction_id, junction in enumerate(junctions, start=1):
        source_name = f"junction_{junction_id}.kj"
        supplement.append(
            f"{junction_id} {len(junction['records'])} {source_name}"
        )
        connection_points = tuple(
            JunctionConnectionPoint(
                identifier=connection_point_id,
                origin=record[2] - junction["representative"],
                frame=junction["frames"][connection_point_id - 1],
            )
            for connection_point_id, record in enumerate(
                junction["records"], start=1
            )
        )
        write_junction_stiffness(
            directory / source_name,
            JunctionStiffness(
                connection_points=connection_points, matrix=junction["matrix"]
            ),
        )
    supplement.append("")
    for junction_id, junction in enumerate(junctions, start=1):
        origin = junction["representative"]
        supplement.append(
            f"{junction_id} {junction_id} "
            + " ".join(f"{value:.16e}" for value in origin)
        )
    supplement.append("")
    branch = 0
    for junction_id, junction in enumerate(junctions, start=1):
        for connection_point_id, record in enumerate(
            junction["records"], start=1
        ):
            shift = record[2] - record[1]
            supplement.append(
                f"{junction_id} {connection_point_id} {branch + 1} 1 "
                + " ".join(f"{value:.16e}" for value in shift)
            )
            branch += 1
    Path(str(input_path) + ".msg").write_text(
        "\n".join(supplement) + "\n", encoding="utf-8"
    )
    return input_path


def validate_resolved() -> dict[str, float]:
    """Compare the complete all-solid-junction BCCH stiffness matrices."""

    geometry = benchmark.build_geometry()
    junctions, branch_records = _resolved_junction_data(geometry)
    stiffness_by_name = {junction["name"]: junction["matrix"] for junction in junctions}

    def stiffness_provider(name, _frames, _radius, _distance):
        return stiffness_by_name[name]

    _, assembled, _ = resolved_reference.build_resolved_system(
        junction_stiffness_provider=stiffness_provider
    )
    reference = hybrid_reference.compute_effective_stiffness(assembled)[0]
    with tempfile.TemporaryDirectory() as directory:
        input_path = _write_resolved_input(
            Path(directory), geometry, junctions, branch_records
        )
        _, _, standardized = homogenize(input_path)
    difference = standardized.effective_stiffness - reference
    swiftcomp = _read_swiftcomp_stiffness()
    swiftcomp_difference = standardized.effective_stiffness - swiftcomp
    swiftcomp_ex = float(benchmark.engineering_constants(swiftcomp)[0])
    standardized_ex = float(
        benchmark.engineering_constants(standardized.effective_stiffness)[0]
    )
    return {
        "relative_matrix_error": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "reference_ex": float(benchmark.engineering_constants(reference)[0]),
        "standardized_ex": standardized_ex,
        "swiftcomp_relative_matrix_error": float(
            np.linalg.norm(swiftcomp_difference) / np.linalg.norm(swiftcomp)
        ),
        "swiftcomp_maximum_absolute_error": float(np.max(np.abs(swiftcomp_difference))),
        "swiftcomp_ex": swiftcomp_ex,
        "swiftcomp_ex_error_percent": 100.0 * (standardized_ex / swiftcomp_ex - 1.0),
    }


def main() -> None:
    for name, result in (("rigid", validate_rigid()), ("resolved", validate_resolved())):
        print(name)
        for key, value in result.items():
            print(f"  {key}: {value:.16e}")


if __name__ == "__main__":
    main()
