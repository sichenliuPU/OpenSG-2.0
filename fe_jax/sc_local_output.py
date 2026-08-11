"""SwiftComp-compatible text output for localized three-dimensional fields."""

from __future__ import annotations

from pathlib import Path

from .dehomogenization import LocalFields


def write_local_displacement(path: str | Path, fields: LocalFields) -> None:
    """Write ``node_no u1 u2 u3`` records to ``.u``."""

    lines = [
        f"{node:10d} " + " ".join(f"{value:20.10E}" for value in displacement)
        for node, displacement in enumerate(fields.displacement, start=1)
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_local_nodal_fields(path: str | Path, fields: LocalFields) -> None:
    """Write coordinates, engineering strain, and Cauchy stress to ``.sn``."""

    lines = []
    for coordinates, strain, stress in zip(
        fields.coordinates, fields.strain, fields.stress, strict=True
    ):
        values = [*coordinates, *strain, *stress]
        lines.append(" ".join(f"{value:20.10E}" for value in values))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_local_outputs(input_path: str | Path, fields: LocalFields) -> tuple[Path, Path]:
    """Write the standard SwiftComp ``.u`` and ``.sn`` files."""

    input_path = Path(input_path)
    displacement_path = Path(str(input_path) + ".u")
    nodal_path = Path(str(input_path) + ".sn")
    write_local_displacement(displacement_path, fields)
    write_local_nodal_fields(nodal_path, fields)
    return displacement_path, nodal_path
