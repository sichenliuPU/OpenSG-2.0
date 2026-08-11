"""Plot the simple-cubic OpenSG/SwiftComp centerline comparison data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("S11", "S22", "S33", "S23", "S13", "S12")


def plot_centerline(source: str | Path, output: str | Path) -> Path:
    """Create the six-component stress plot from a comparison CSV file."""

    source = Path(source)
    output = Path(output)
    records = np.genfromtxt(source, delimiter=",", names=True, dtype=None)
    x = np.asarray(records["x"], dtype=float)
    figure, axes = plt.subplots(
        3, 2, figsize=(11.0, 10.0), sharex=True, constrained_layout=True
    )
    styles = {
        "Euler": ("#2166ac", "--"),
        "Timoshenko": ("#b2182b", "-"),
    }
    for axis, component in zip(axes.flat, COMPONENTS, strict=True):
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
        axis.set_ylabel(f"{component} (MPa)")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [
        labels.index("OpenSG Euler"),
        labels.index("OpenSG Timoshenko"),
        labels.index("SwiftComp"),
    ]
    axes[0, 0].legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
    )
    for axis in axes[-1]:
        axis.set_xlabel("x along the junction-center to +X-beam line")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)
    return output


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
        "--output",
        type=Path,
        default=directory / "simple_cubic_plus_x_stress_line.png",
    )
    arguments = parser.parse_args()
    print(f"line_plot={plot_centerline(arguments.source, arguments.output)}")


if __name__ == "__main__":
    main()
