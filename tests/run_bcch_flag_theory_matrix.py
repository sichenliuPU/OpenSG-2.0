"""Run the BCCH benchmark for every beam theory and junction flag."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TIMOSHENKO = ROOT / "Timoshenko"
if str(TIMOSHENKO) not in sys.path:
    sys.path.insert(0, str(TIMOSHENKO))

import bcch_lambda06_benchmark as benchmark
import bcch_resolved_homogenization as resolved_reference
import Timoshenko_cross as timoshenko_reference
from bcch_resolved_junctions import frames as junction_frames
from tet10_boolean_backend import build_boolean_tet10_connector_mesh

from fe_jax.hybrid_homogenization import homogenize


def _swiftcomp_stiffness() -> np.ndarray:
    path = TIMOSHENKO / "BCCH_nSG3_3D_C3D10_periodic_h010.sc.k"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(
        index for index, line in enumerate(lines)
        if "The Effective Stiffness Matrix" in line
    )
    rows = []
    for line in lines[header + 1 :]:
        values = line.split()
        if len(values) != 6:
            continue
        try:
            rows.append([float(value) for value in values])
        except ValueError:
            continue
        if len(rows) == 6:
            return np.asarray(rows)
    raise ValueError(f"Effective stiffness matrix was not found in {path}.")


def _junction_data(geometry, mesh_factor: float = 0.4):
    junctions = []
    branch_records = []
    for name, items, distance_factor, representative, shifts in (
        resolved_reference.junction_groups(geometry)
    ):
        distance = distance_factor * geometry.radius
        directions = []
        records = []
        for item in items:
            node, _other, direction = resolved_reference.endpoint_data(geometry, item)
            wrapped = geometry.nodes[node] + distance * direction
            unwrapped = wrapped + shifts[node]
            directions.append(direction)
            records.append((item, wrapped, unwrapped, direction))
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
        junctions.append(
            {
                "name": name,
                "representative": representative,
                "records": records,
                "mesh": mesh,
            }
        )
        branch_records.extend(records)
    return junctions, branch_records


def _beam_connectivity(nodes, beams, theory: str):
    nodes = [np.asarray(node, dtype=float) for node in nodes]
    elements = []
    for start, end in beams:
        start = int(start)
        end = int(end)
        if theory not in ("euler", "timoshenko"):
            raise ValueError(f"Unknown beam theory: {theory}")
        elements.append((start, end))
    return np.asarray(nodes), np.asarray(elements, dtype=np.int64)


def _beam_topology(geometry, junction_flag: int, branch_records, theory: str):
    if junction_flag == 0:
        nodes, elements = _beam_connectivity(geometry.nodes, geometry.beams, theory)
        transverse = benchmark.beam_frames(geometry.nodes, geometry.beams)
        pairs = timoshenko_reference.detect_periodic_pairs(nodes)
        return nodes, elements, np.asarray(transverse), pairs

    nodes = [record[1] for record in branch_records]
    half_beams = []
    transverse = []
    midpoint_indices = {}
    for branch, (item, _wrapped, _unwrapped, direction) in enumerate(branch_records):
        beam, _endpoint = item
        if beam not in midpoint_indices:
            midpoint_indices[beam] = len(nodes)
            n1, n2 = geometry.beams[beam]
            nodes.append(0.5 * (geometry.nodes[n1] + geometry.nodes[n2]))
        half_beams.append((branch, midpoint_indices[beam]))
        trial = (
            np.array([0.0, 0.0, 1.0])
            if abs(direction[2]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        direction_2 = trial - np.dot(trial, direction) * direction
        transverse.append(direction_2 / np.linalg.norm(direction_2))
    nodes, elements = _beam_connectivity(nodes, half_beams, theory)
    return nodes, elements, np.asarray(transverse), []


def _write_solid_junction(directory: Path, identifier: int, junction) -> Path:
    mesh = junction["mesh"]
    path = directory / f"junction_{identifier}.sc"
    lines = [
        "0 0 0 0",
        "",
        f"3 {len(mesh.nodes)} {len(mesh.tet10)} 1 0 0",
        "",
    ]
    lines.extend(
        f"{node_id} " + " ".join(f"{value:.16e}" for value in point)
        for node_id, point in enumerate(mesh.nodes, start=1)
    )
    lines.append("")
    for element_id, element in enumerate(mesh.tet10, start=1):
        nodes = (element + 1).tolist()
        connectivity = nodes[:4] + [0] + nodes[4:]
        lines.append(
            f"{element_id} 1 " + " ".join(str(node) for node in connectivity)
        )
    lines.extend(
        (
            "",
            "1 0 1",
            "0.0 1.0",
            f"{benchmark.MATERIAL_E:.16e} {benchmark.MATERIAL_NU:.16e}",
            "",
            f"{mesh.volume:.16e}",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    connection_lines = [f"1 {len(mesh.tri6_by_connection)}", ""]
    for connection_point_id, faces in enumerate(mesh.tri6_by_connection, start=1):
        values = np.concatenate(
            (
                mesh.connection_centers[connection_point_id - 1],
                mesh.beta_matrices[connection_point_id - 1].reshape(-1),
            )
        )
        connection_lines.append(
            f"{connection_point_id} {len(faces)} "
            + " ".join(f"{value:.16e}" for value in values)
        )
        connection_lines.extend(
            " ".join(str(int(node) + 1) for node in face) for face in faces
        )
        connection_lines.append("")
    Path(str(path) + ".msg").write_text(
        "\n".join(connection_lines) + "\n", encoding="utf-8"
    )
    return path


def _write_model(
    directory: Path,
    geometry,
    junctions,
    branch_records,
    theory: str,
    junction_flag: int,
) -> Path:
    nodes, elements, transverse, periodic_pairs = _beam_topology(
        geometry, junction_flag, branch_records, theory
    )
    path = directory / f"bcch_{theory}_flag{junction_flag}.sc"
    lines = [
        f"0 2 1 0 {junction_flag}",
        "",
        f"3 {len(nodes)} {len(elements)} 1 {len(periodic_pairs)} 0",
        "",
    ]
    lines.extend(
        f"{node_id} " + " ".join(f"{value:.16e}" for value in point)
        for node_id, point in enumerate(nodes, start=1)
    )
    lines.append("")
    lines.extend(
        f"{element_id} 1 "
        + " ".join(str(int(node) + 1) for node in element)
        for element_id, element in enumerate(elements, start=1)
    )
    lines.append("")
    for element_id, (element, direction_2) in enumerate(
        zip(elements, transverse, strict=True), start=1
    ):
        start = nodes[element[0]]
        end = nodes[element[1]]
        points = np.concatenate((start, end, start + direction_2))
        lines.append(
            f"{element_id} " + " ".join(f"{value:.16e}" for value in points)
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    full_section = benchmark.section_matrix(
        geometry.radius, benchmark.MATERIAL_E, benchmark.MATERIAL_NU
    )
    if theory == "euler":
        indices = np.array([0, 3, 4, 5])
        section = full_section[np.ix_(indices, indices)]
        beam_record = "1 0 3 4"
    else:
        section = full_section
        beam_record = "1 1 4 6"
    supplement = [
        f"1 1 {len(elements)} "
        + (
            "0 0 0 0"
            if junction_flag == 0
            else f"{len(junctions)} {len(junctions)} {len(branch_records)} 0"
        ),
        "",
        beam_record,
    ]
    supplement.extend(" ".join(f"{value:.16e}" for value in row) for row in section)
    supplement.append("")
    supplement.extend(f"{element_id} 1" for element_id in range(1, len(elements) + 1))
    if junction_flag:
        supplement.append("")
        for junction_id, junction in enumerate(junctions, start=1):
            source = (
                f"junction_{junction_id}.sc"
                if junction_flag == 1
                else f"junction_{junction_id}.sc.kj"
            )
            supplement.append(
                f"{junction_id} {len(junction['records'])} {source}"
            )
        supplement.append("")
        for junction_id, junction in enumerate(junctions, start=1):
            supplement.append(
                f"{junction_id} {junction_id} "
                + " ".join(
                    f"{value:.16e}" for value in junction["representative"]
                )
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
    Path(str(path) + ".msg").write_text(
        "\n".join(supplement) + "\n", encoding="utf-8"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write and run the six OpenSG BCCH beam--junction examples."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Retain all generated inputs and outputs in this directory.",
    )
    arguments = parser.parse_args()
    geometry = benchmark.build_geometry()
    swiftcomp = _swiftcomp_stiffness()
    swiftcomp_ex = benchmark.engineering_constants(swiftcomp)[0]
    junctions, branch_records = _junction_data(geometry)
    if arguments.output_directory is None:
        directory_context = tempfile.TemporaryDirectory()
    else:
        arguments.output_directory.mkdir(parents=True, exist_ok=True)
        directory_context = nullcontext(str(arguments.output_directory.resolve()))
    with directory_context as temporary:
        directory = Path(temporary)
        for identifier, junction in enumerate(junctions, start=1):
            _write_solid_junction(directory, identifier, junction)
        results = {}
        for theory in ("euler", "timoshenko"):
            for junction_flag in (0, 1, 2):
                path = _write_model(
                    directory,
                    geometry,
                    junctions,
                    branch_records,
                    theory,
                    junction_flag,
                )
                _, _, result = homogenize(path)
                matrix = result.effective_stiffness
                effective_e = benchmark.engineering_constants(matrix)[0]
                results[(theory, junction_flag)] = (matrix, effective_e, result)
                difference = matrix - swiftcomp
                print(f"\n{theory} junction_flag={junction_flag}")
                print(np.array2string(matrix, precision=8, suppress_small=True))
                print(f"E1={effective_e:.10f} MPa")
                print(f"E1 error={100.0 * (effective_e / swiftcomp_ex - 1.0):+.8f}%")
                print(
                    "matrix error="
                    f"{100.0 * np.linalg.norm(difference) / np.linalg.norm(swiftcomp):.8f}%"
                )
                print(f"maximum absolute error={np.max(np.abs(difference)):.10f} MPa")
                print(
                    f"junction analysis time={result.junction_analysis_time:.6f} s, "
                    f"homogenization time={result.homogenization_time:.6f} s, "
                    f"total time={result.total_time:.6f} s"
                )

        print("\nflag 1 versus flag 2")
        for theory in ("euler", "timoshenko"):
            matrix_1 = results[(theory, 1)][0]
            matrix_2 = results[(theory, 2)][0]
            print(
                f"{theory}: relative difference="
                f"{np.linalg.norm(matrix_1 - matrix_2) / np.linalg.norm(matrix_1):.16e}"
            )


if __name__ == "__main__":
    main()
