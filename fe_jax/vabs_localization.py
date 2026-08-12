"""VABS cross-section localization used by OpenSG beam members."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .beam import FloatArray


@dataclass(frozen=True)
class VABSFields:
    node_ids: np.ndarray
    section_coordinates: FloatArray
    displacement: FloatArray
    strain: FloatArray
    stress: FloatArray


def rotation_matrix(rotation_vector: FloatArray) -> FloatArray:
    """Return VABS rotation rows for a local rotation vector."""

    vector = np.asarray(rotation_vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-15:
        return np.eye(3)
    axis = vector / angle
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    active = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    return active.T


def global_file_text(
    displacement: FloatArray,
    rotation: FloatArray,
    resultants: FloatArray,
) -> str:
    """Create a VABS global-fields file from [F1,F2,F3,M1,M2,M3]."""

    values = np.asarray(resultants, dtype=float).reshape(-1)
    if values.shape != (6,):
        raise ValueError("VABS beam stress resultants must be [F1,F2,F3,M1,M2,M3].")
    f1, f2, f3, m1, m2, m3 = values
    rows = [" ".join(f"{value:.16e}" for value in displacement)]
    rows.extend(
        " ".join(f"{value:.16e}" for value in row)
        for row in rotation_matrix(rotation)
    )
    rows.append(" ".join(f"{value:.16e}" for value in (f1, m1, m2, m3)))
    rows.append(" ".join(f"{value:.16e}" for value in (f2, f3)))
    rows.extend(["0 0 0 0 0 0"] * 4)
    return "\n".join(rows) + "\n\n"


def _averaged_records(path: Path, width: int) -> FloatArray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.replace("D", "E").split()
        if len(fields) < width:
            continue
        try:
            rows.append([float(value) for value in fields[:width]])
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"VABS output has no field records: {path}")
    data = np.asarray(rows, dtype=float)
    node_ids = np.unique(data[:, 0].astype(np.int64))
    return np.vstack([
        data[data[:, 0].astype(np.int64) == node].mean(axis=0)
        for node in node_ids
    ])


def _resolve_executable(executable: str | Path | None) -> Path:
    if executable is not None:
        result = Path(executable)
    elif os.environ.get("VABS_EXE"):
        result = Path(os.environ["VABS_EXE"])
    else:
        discovered = shutil.which("VABS") or shutil.which("vabs")
        result = (
            Path(discovered)
            if discovered is not None
            else Path(r"C:\Program Files (x86)\VABS\Windows 4.0\VABS.exe")
        )
    if not result.exists():
        raise FileNotFoundError(
            f"VABS executable was not found: {result}. Use --vabs, set VABS_EXE, "
            "or add VABS to PATH."
        )
    return result


class VABSSession:
    """Prepared VABS cross section that can localize many axial stations."""

    def __init__(
        self,
        source: str | Path,
        work_directory: str | Path,
        executable: str | Path | None = None,
    ) -> None:
        self.executable = _resolve_executable(executable)
        self.work = Path(work_directory)
        self.work.mkdir(parents=True, exist_ok=True)
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"VABS section source was not found: {source}")
        self.input = self.work / source.name
        lines = source.read_text(encoding="utf-8").splitlines()
        nonempty = [index for index, line in enumerate(lines) if line.strip()]
        if len(nonempty) < 2:
            raise ValueError(f"Invalid VABS section input: {source}")
        # A shearable section is used for both OpenSG beam theories so that
        # transverse shear stresses can be recovered from equilibrium forces.
        lines[nonempty[1]] = "1 0 0"
        self.input.write_text("\n".join(lines) + "\n\n", encoding="utf-8")
        self._run([self.input.name])
        if not Path(str(self.input) + ".K").exists():
            raise RuntimeError(f"VABS did not prepare section {self.input.name}.")

    def _run(self, arguments: list[str]) -> None:
        completed = subprocess.run(
            [str(self.executable), *arguments],
            cwd=self.work,
            text=True,
            capture_output=True,
            timeout=180,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"VABS failed ({completed.returncode}):\n"
                f"{completed.stdout}\n{completed.stderr}"
            )

    def localize(
        self,
        displacement: FloatArray,
        rotation: FloatArray,
        resultants: FloatArray,
    ) -> VABSFields:
        Path(str(self.input) + ".glb").write_text(
            global_file_text(displacement, rotation, resultants), encoding="utf-8"
        )
        self._run([self.input.name, "2"])
        displacement_rows = _averaged_records(Path(str(self.input) + ".U"), 6)
        strain_rows = _averaged_records(Path(str(self.input) + ".EN"), 9)
        stress_rows = _averaged_records(Path(str(self.input) + ".SN"), 9)
        if not (
            np.array_equal(displacement_rows[:, 0], strain_rows[:, 0])
            and np.array_equal(displacement_rows[:, 0], stress_rows[:, 0])
        ):
            raise RuntimeError("VABS displacement, strain, and stress node IDs differ.")
        # VABS nodal tensor order is [11,12,13,22,23,33].
        swift_order = [0, 3, 5, 4, 2, 1]
        return VABSFields(
            node_ids=displacement_rows[:, 0].astype(np.int64),
            section_coordinates=displacement_rows[:, 1:3],
            displacement=displacement_rows[:, 3:6],
            strain=strain_rows[:, 3:9][:, swift_order],
            stress=stress_rows[:, 3:9][:, swift_order],
        )
