"""Plot separate simple-cubic OpenSG/SwiftComp centerline comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("S11", "S22", "S33", "S23", "S13", "S12")
COMPONENT_LABELS = {
    "S11": r"$\sigma_{11}$",
    "S22": r"$\sigma_{22}$",
    "S33": r"$\sigma_{33}$",
    "S23": r"$\sigma_{23}$",
    "S13": r"$\sigma_{13}$",
    "S12": r"$\sigma_{12}$",
}


def plot_centerline_components(
    source: str | Path, output_directory: str | Path
) -> tuple[Path, ...]:
    """Create one manuscript-ready plot for each stress component."""

    source = Path(source)
    output_directory = Path(output_directory)
    records = np.genfromtxt(source, delimiter=",", names=True, dtype=None)
    x = np.asarray(records["x"], dtype=float)
    styles = {
        "Euler": ("#2166ac", "--"),
        "Timoshenko": ("#b2182b", "-"),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for component in COMPONENTS:
        figure, axis = plt.subplots(
            figsize=(7.2, 4.8), constrained_layout=True
        )
        for theory in ("Timoshenko", "Euler"):
            color, linestyle = styles[theory]
            axis.plot(
                x,
                np.asarray(records[f"OpenSG_{theory}_{component}"]) / 1.0e6,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=f"OpenSG {theory}",
                zorder=3 if theory == "Euler" else 2,
            )
        axis.scatter(
            x,
            np.asarray(records[f"SwiftComp_{component}"]) / 1.0e6,
            color="black",
            s=13.0,
            label="SwiftComp",
            zorder=4,
        )
        axis.axvline(0.015, color="0.4", linestyle="--", linewidth=1.0)
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(f"{COMPONENT_LABELS[component]} (MPa)")
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        order = [
            labels.index("OpenSG Euler"),
            labels.index("OpenSG Timoshenko"),
            labels.index("SwiftComp"),
        ]
        axis.legend(
            [handles[index] for index in order],
            [labels[index] for index in order],
            frameon=False,
        )
        output = output_directory / f"simple_cubic_plus_x_{component}.png"
        figure.savefig(output, dpi=300, facecolor="white")
        plt.close(figure)
        outputs.append(output)
    return tuple(outputs)


def main() -> None:
    directory = Path(__file__).parent / "results"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=directory / "simple_cubic_plus_x_stress_line.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=directory,
    )
    arguments = parser.parse_args()
    for output in plot_centerline_components(arguments.source, arguments.output_dir):
        print(f"line_plot={output}")


if __name__ == "__main__":
    main()
