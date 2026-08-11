"""Command-line entry point for OpenSG beam and hybrid homogenization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from .hybrid_homogenization import homogenize
from .sc_hybrid_output import write_echo, write_effective_properties


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
    arguments = parser.parse_args()
    if arguments.dimension.upper() != "3D" or arguments.operation.upper() != "H":
        parser.error("This entry point currently supports only: 3D H")

    try:
        model, supplement, result = homogenize(arguments.input, arguments.supplement)
    except (ValueError, OSError, RuntimeError, np.linalg.LinAlgError) as error:
        parser.exit(2, f"OpenSG input error: {error}\n")
    echo_path = Path(str(arguments.input) + ".ech")
    write_echo(echo_path, model, supplement)
    output_path = Path(str(arguments.input) + ".k")
    write_effective_properties(output_path, model, result)
    print(f"Effective properties: {output_path}")
    print(f"Input echo: {echo_path}")
    if result.has_mechanism:
        print(
            "OpenSG warning: the effective stiffness contains a zero-energy mechanism; "
            "check connectivity and periodic constraints.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
