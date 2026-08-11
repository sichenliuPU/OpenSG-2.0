"""Example-specific simple-cubic comparison with SwiftComp fields."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.interpolate import LinearNDInterpolator

OPENSG_ROOT = Path(__file__).parents[2]
if str(OPENSG_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENSG_ROOT))

from fe_jax.beam import beam_frame
from fe_jax.dehomogenization import (
    recover_beam_states,
    recover_vabs_fields,
)
from fe_jax.hybrid_homogenization import homogenize
from fe_jax.junction import JunctionConnectionPoint, write_junction_stiffness
from tools.junction_boolean import (
    build_boolean_junction,
    write_solid_junction_input,
)
from c3d20_recovery import (
    build_simple_cubic_c3d20_recovery,
    recover_c3d20_centerline_stress,
)
from fe_jax.junction_solid import analyze_junction
from fe_jax.sc_glb_input import read_global_fields


ROOT = Path(__file__).parents[3]
DIRECTIONS = np.vstack((np.eye(3), -np.eye(3)))[[0, 3, 1, 4, 2, 5]]
TRANSVERSE = np.array([
    [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
])
TIMOSHENKO_SECTION = np.diag([
    7.0e6, 2.2435897436e6, 2.2435897436e6,
    3.7847663855e1, 5.8333333333e1, 5.8333333333e1,
])
EULER_SECTION = TIMOSHENKO_SECTION[np.ix_([0, 3, 4, 5], [0, 3, 4, 5])]
SWIFTCOMP_COLUMNS = {
    "S11": 0, "S22": 1, "S33": 2,
    # The archived CSV's stored values follow SwiftComp [23,13,12], although
    # its final three headings read [S12,S13,S23].
    "S23": 3, "S13": 4, "S12": 5,
}


def _write_line_comparison(
    output_directory: Path,
    x: np.ndarray,
    junction_mask: np.ndarray,
    reference: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> Path:
    """Write pointwise centerline values for example post-processing."""

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "simple_cubic_plus_x_stress_line.csv"
    component_names = tuple(SWIFTCOMP_COLUMNS)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "x", "region",
            *[f"SwiftComp_{name}" for name in component_names],
            *[
                f"OpenSG_{theory}_{name}"
                for theory in predictions for name in component_names
            ],
        ])
        for index, coordinate in enumerate(x):
            ordered_reference = [
                reference[index, SWIFTCOMP_COLUMNS[name]]
                for name in component_names
            ]
            writer.writerow([
                coordinate,
                "junction" if junction_mask[index] else "beam",
                *ordered_reference,
                *[
                    predictions[theory][index, SWIFTCOMP_COLUMNS[name]]
                    for theory in predictions for name in component_names
                ],
            ])

    return csv_path


def _connection_points() -> tuple[JunctionConnectionPoint, ...]:
    points = []
    for identifier, (direction, transverse) in enumerate(
        zip(DIRECTIONS, TRANSVERSE, strict=True), start=1
    ):
        points.append(JunctionConnectionPoint(
            identifier=identifier,
            origin=0.015 * direction,
            frame=beam_frame(np.zeros(3), direction, transverse),
        ))
    return tuple(points)


def _write_model(directory: Path, vabs_source: Path, theory: str) -> Path:
    if theory not in {"Euler", "Timoshenko"}:
        raise ValueError(f"Unknown OpenSG beam theory: {theory}.")
    points = _connection_points()
    recovery_path = directory / "simple_cubic_junction.sc"
    junction_path = Path(str(recovery_path) + ".kj")
    if not junction_path.exists():
        junction_model = build_boolean_junction(
            points, "square", 0.005, 0.003, 1, 1, (70.0e9, 0.3)
        )
        junction = analyze_junction(junction_model)
        write_junction_stiffness(junction_path, junction.stiffness)
        write_solid_junction_input(recovery_path, junction_model)

    nodes = np.vstack((0.015 * DIRECTIONS, 0.1 * DIRECTIONS))
    lines = ["0 2 1 0 2", "", "3 12 6 1 3 0", ""]
    lines.extend(
        f"{index} " + " ".join(f"{value:.16e}" for value in point)
        for index, point in enumerate(nodes, start=1)
    )
    lines.append("")
    lines.extend(f"{index} 1 {index} {index + 6}" for index in range(1, 7))
    lines.append("")
    for index, (start, direction, transverse) in enumerate(
        zip(nodes[:6], DIRECTIONS, TRANSVERSE, strict=True), start=1
    ):
        orientation = np.vstack((start, start + direction, start + transverse))
        lines.append(
            f"{index} " + " ".join(f"{value:.16e}" for value in orientation.ravel())
        )
    lines.extend((
        "", "8 7", "10 9", "12 11", "",
        "1 0 1", "0.0 1.0", "70000000000.0 0.3", "", "0.008", "",
    ))
    model_path = directory / f"simple_cubic_{theory.lower()}.sc"
    model_path.write_text("\n".join(lines), encoding="utf-8")

    if theory == "Euler":
        beam_record = "1 0 3 4"
        section = EULER_SECTION
    else:
        beam_record = "1 1 4 6"
        section = TIMOSHENKO_SECTION
    supplement = ["1 1 6 1 1 6 0", "", beam_record]
    supplement.extend(
        " ".join(f"{value:.16e}" for value in row) for row in section
    )
    supplement.append("")
    supplement.extend(f"{index} 1" for index in range(1, 7))
    supplement.extend(("", f'1 6 "{junction_path.name}"', "", "1 1 0 0 0", ""))
    supplement.extend(f"1 {index} {index} 1 0 0 0" for index in range(1, 7))
    supplement.extend((
        "",
        f'BEAM_RECOVERY 1 "{vabs_source}"',
        "",
    ))
    Path(str(model_path) + ".msg").write_text(
        "\n".join(supplement), encoding="utf-8"
    )
    state = np.load(ROOT / "SC_random_macro_state.npz")
    global_lines = [
        " ".join(f"{value:.16e}" for value in state["macro_displacement"]),
        "1 0 0", "0 1 0", "0 0 1", "1",
        " ".join(f"{value:.16e}" for value in state["macro_strain"]), "",
    ]
    Path(str(model_path) + ".glb").write_text(
        "\n".join(global_lines) + "\n", encoding="utf-8"
    )
    return model_path


def _section_center_stress(part) -> np.ndarray:
    transverse = part.coordinates - part.coordinates.mean(axis=0)
    radius = np.linalg.norm(transverse, axis=1)
    if radius.min() < 1.0e-12:
        return part.stress[np.argmin(radius)]
    # Use the two coordinates with the largest spread as section coordinates.
    axes = np.argsort(np.ptp(part.coordinates, axis=0))[-2:]
    interpolator = LinearNDInterpolator(part.coordinates[:, axes], part.stress)
    center = part.coordinates.mean(axis=0)[axes]
    value = np.asarray(interpolator(center), dtype=float).reshape(-1)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        return part.stress[np.argmin(radius)]
    return value


def run_comparison(
    vabs: Path | None = None,
    stations: int = 11,
    output_directory: Path | None = None,
) -> Path:
    vabs_source = ROOT / "VABS" / "simple cubic" / "square0.sg"
    reference_path = ROOT / "VABS" / "simple cubic" / "cross_nSG3_3D_C3D20Rpbc_nodal_stress.csv"
    reference = np.genfromtxt(
        reference_path, delimiter=",", names=True, dtype=float
    )
    x_reference = np.asarray(reference["X"], dtype=float)
    junction_mask = x_reference <= 0.015 + 1.0e-12
    reference_stress = np.column_stack([
        reference[name] for name in ("S11", "S22", "S33", "S12", "S13", "S23")
    ])

    # The structured mesh is common to both beam theories and places real
    # 20-node solid nodes at all SwiftComp junction line coordinates.
    recovery = build_simple_cubic_c3d20_recovery(
        np.asarray([point.frame for point in _connection_points()]),
        side=0.01,
        stub_length=0.01,
        young=70.0e9,
        poisson=0.3,
        elements_per_side=4,
    )

    predictions: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="opensg_sc_verify_") as directory_name:
        directory = Path(directory_name)
        for theory in ("Euler", "Timoshenko"):
            input_path = _write_model(directory, vabs_source, theory)
            _model, supplement, result = homogenize(input_path)
            global_fields = read_global_fields(
                Path(str(input_path) + ".glb"),
                result.effective_stiffness,
                result.effective_compliance,
            )
            xi = np.linspace(-1.0, 1.0, stations)
            states = recover_beam_states(result, supplement, global_fields, xi)
            plus_x_states = [state for state in states if state.element_id == 1]
            beam_parts = recover_vabs_fields(plus_x_states, supplement, vabs)
            beam_x = np.asarray([state.center[0] for state in plus_x_states])
            beam_stress = np.vstack([
                _section_center_stress(part) for part in beam_parts
            ])

            assembly = result.junction_assemblies[0]
            junction_displacement = (
                assembly.b_epsilon - assembly.b_v @ result.full_fluctuation
            ) @ global_fields.strain
            junction_x, junction_stress = recover_c3d20_centerline_stress(
                recovery,
                junction_displacement,
                young=70.0e9,
                poisson=0.3,
                x_min=0.0,
                x_max=0.015,
            )
            if not np.allclose(
                x_reference[junction_mask], junction_x, rtol=0.0, atol=1.0e-12
            ):
                raise RuntimeError(
                    "The 20-node solid centerline nodes do not coincide with the "
                    "SwiftComp junction line points."
                )
            prediction = np.empty((len(reference), 6))
            prediction[junction_mask] = junction_stress
            for component in range(6):
                prediction[~junction_mask, component] = np.interp(
                    x_reference[~junction_mask], beam_x, beam_stress[:, component]
                )
            predictions[theory] = prediction

    destination = output_directory or (
        Path(__file__).parent / "results"
    )
    return _write_line_comparison(
        destination,
        x_reference,
        junction_mask,
        reference_stress,
        predictions,
    )


def run_cli_smoke(vabs: Path | None = None) -> tuple[int, int]:
    """Run the public H/L commands and validate the two localization files."""

    vabs_source = ROOT / "VABS" / "simple cubic" / "square0.sg"
    with tempfile.TemporaryDirectory(prefix="opensg_sc_cli_") as directory_name:
        directory = Path(directory_name)
        local_vabs_source = directory / vabs_source.name
        shutil.copyfile(vabs_source, local_vabs_source)
        input_path = _write_model(
            directory, Path(local_vabs_source.name), "Timoshenko"
        )
        base = [sys.executable, "-m", "fe_jax.opensg", str(input_path), "3D"]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(OPENSG_ROOT), environment.get("PYTHONPATH", "")
        )))
        command = [*base, "L", "--stations", "2"]
        if vabs is not None:
            command.extend(("--vabs", str(vabs)))
        subprocess.run(command, check=True, cwd=directory, env=environment)
        subprocess.run(
            [*base, "H"], check=True, cwd=directory, env=environment
        )
        displacement_lines = Path(str(input_path) + ".u").read_text().splitlines()
        nodal_lines = Path(str(input_path) + ".sn").read_text().splitlines()
        if not displacement_lines or not nodal_lines:
            raise RuntimeError("The public localization command wrote an empty field file.")
        if any(len(line.split()) != 4 for line in displacement_lines):
            raise RuntimeError("A .u record does not contain four columns.")
        if any(len(line.split()) != 15 for line in nodal_lines):
            raise RuntimeError("A .sn record does not contain fifteen columns.")
        return len(displacement_lines), len(nodal_lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vabs", type=Path, default=None)
    parser.add_argument("--stations", type=int, default=11)
    parser.add_argument("--cli-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    if arguments.cli_smoke:
        displacement_rows, nodal_rows = run_cli_smoke(arguments.vabs)
        print(f"u_rows={displacement_rows}")
        print(f"sn_rows={nodal_rows}")
        return
    csv_path = run_comparison(
        arguments.vabs, arguments.stations, arguments.output_dir
    )
    print(f"line_data={csv_path}")


if __name__ == "__main__":
    main()
