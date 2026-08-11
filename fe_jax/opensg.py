"""Command-line entry point for OpenSG beam and hybrid homogenization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from .hybrid_homogenization import homogenize
from .dehomogenization import dehomogenize
from .sc_glb_input import read_global_fields
from .sc_hybrid_output import write_echo, write_effective_properties
from .sc_local_output import write_local_outputs


def main() -> None:
    """Run OpenSG homogenization from a standard structure-gene input."""

    parser = argparse.ArgumentParser(
        description="OpenSG beam and hybrid beam--junction homogenization."
    )
    parser.add_argument("input", type=Path, help="OpenSG structure-gene input file.")
    parser.add_argument("dimension", nargs="?", default="3D")
    parser.add_argument("operation", nargs="?", default="H")
    parser.add_argument(
        "--supplement",
        type=Path,
        default=None,
        help="Optional beam/junction supplement; defaults to INPUT.msg.",
    )
    parser.add_argument(
        "--stations", type=int, default=3,
        help="Number of VABS axial stations per beam for localization.",
    )
    parser.add_argument(
        "--vabs", type=Path, default=None,
        help="Optional VABS executable; otherwise use VABS_EXE or the standard Windows path.",
    )
    arguments = parser.parse_args()
    operation = arguments.operation.upper()
    if arguments.dimension.upper() != "3D" or operation not in {"H", "L"}:
        parser.error("This entry point supports: 3D H or 3D L")
    try:
        model, supplement, result = homogenize(arguments.input, arguments.supplement)
    except (ValueError, OSError, RuntimeError, np.linalg.LinAlgError) as error:
        parser.exit(2, f"OpenSG input error: {error}\n")
    echo_path = Path(str(arguments.input) + ".ech")
    write_echo(echo_path, model, supplement)
    if operation == "H":
        output_path = Path(str(arguments.input) + ".k")
        write_effective_properties(output_path, model, result)
        print(f"Effective properties: {output_path}")
        print(f"Input echo: {echo_path}")
    else:
        global_path = Path(str(arguments.input) + ".glb")
        try:
            global_fields = read_global_fields(
                global_path,
                result.effective_stiffness,
                result.effective_compliance,
            )
            local_fields = dehomogenize(
                model, supplement, result, global_fields,
                stations=arguments.stations, executable=arguments.vabs,
            )
            displacement_path, nodal_path = write_local_outputs(
                arguments.input, local_fields
            )
        except (ValueError, OSError, RuntimeError, np.linalg.LinAlgError) as error:
            parser.exit(2, f"OpenSG localization error: {error}\n")
        print(f"Local displacement: {displacement_path}")
        print(f"Local nodal fields: {nodal_path}")
    if result.has_mechanism:
        print(
            "OpenSG warning: the effective stiffness contains a zero-energy mechanism; "
            "check connectivity and periodic constraints.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
