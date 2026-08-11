"""Build a reusable OpenSG solid-junction input from connection data."""

from __future__ import annotations

import argparse
from pathlib import Path

from fe_jax.junction import read_junction_stiffness
from junction_boolean import (
    build_boolean_junction,
    write_solid_junction_input,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("junction_stiffness", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--section", choices=("square", "circular"), required=True)
    parser.add_argument("--section-size", type=float, required=True)
    parser.add_argument("--mesh-size", type=float, required=True)
    parser.add_argument("--element-flag", type=int, choices=(0, 1), default=1)
    parser.add_argument("--material-id", type=int, default=1)
    parser.add_argument("--young", type=float, required=True)
    parser.add_argument("--poisson", type=float, required=True)
    arguments = parser.parse_args()

    junction = read_junction_stiffness(arguments.junction_stiffness)
    model = build_boolean_junction(
        junction.connection_points,
        arguments.section,
        arguments.section_size,
        arguments.mesh_size,
        arguments.element_flag,
        arguments.material_id,
        (arguments.young, arguments.poisson),
    )
    solid_path, interface_path = write_solid_junction_input(
        arguments.output, model
    )
    print(f"solid_input={solid_path}")
    print(f"interface_input={interface_path}")


if __name__ == "__main__":
    main()
