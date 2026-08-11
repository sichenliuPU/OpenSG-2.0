"""SwiftComp-compatible global fields used for three-dimensional localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .beam import FloatArray


@dataclass(frozen=True)
class GlobalFields:
    """Macroscopic state at one point of a three-dimensional model."""

    displacement: FloatArray
    deformation: FloatArray
    input_flag: int
    strain: FloatArray
    stress: FloatArray


def _tokens(path: Path) -> list[str]:
    values: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].split("!", 1)[0]
        values.extend(line.replace(",", " ").split())
    return values


def read_global_fields(
    path: str | Path,
    effective_stiffness: FloatArray,
    effective_compliance: FloatArray,
) -> GlobalFields:
    """Read an elastic 3D ``.glb`` file in SwiftComp free format."""

    path = Path(path)
    values = _tokens(path)
    if len(values) != 19:
        raise ValueError(
            f"A 3D elastic global-fields file requires 19 values; "
            f"received {len(values)} in {path}."
        )
    try:
        displacement = np.asarray(values[:3], dtype=float)
        deformation = np.asarray(values[3:12], dtype=float).reshape(3, 3)
        input_flag = int(values[12])
        supplied = np.asarray(values[13:19], dtype=float)
    except ValueError as error:
        raise ValueError(f"Invalid numeric value in {path}.") from error
    if not np.all(np.isfinite(np.concatenate((displacement, deformation.ravel(), supplied)))):
        raise ValueError(f"Global fields contain a non-finite value: {path}")
    if input_flag == 1:
        strain = supplied
        stress = np.asarray(effective_stiffness, dtype=float) @ strain
    elif input_flag == 0:
        stress = supplied
        strain = np.asarray(effective_compliance, dtype=float) @ stress
    else:
        raise ValueError("The global-fields input flag must be 0 (stress) or 1 (strain).")
    return GlobalFields(
        displacement=displacement,
        deformation=deformation,
        input_flag=input_flag,
        strain=strain,
        stress=stress,
    )
