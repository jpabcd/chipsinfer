from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_wafer_map(
    report_df: pd.DataFrame,
    output_path: Path | str,
    figsize: tuple[float, float] = (10.0, 8.0),
    chip_aspect: float = 5.0,
) -> Path:
    """Plot the chip results as a red/green wafer map.

    ``Bin`` must contain 0 for OK and 1 for NG. Rows without numeric X/Y
    coordinates are omitted from the plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"X", "Y", "Bin"}
    missing = sorted(required_columns.difference(report_df.columns))
    if missing:
        raise ValueError(
            "Wafer map data is missing required columns: " + ", ".join(missing)
        )

    plot_df = report_df.copy()
    plot_df["X"] = pd.to_numeric(plot_df["X"], errors="coerce")
    plot_df["Y"] = pd.to_numeric(plot_df["Y"], errors="coerce")
    plot_df["Bin"] = pd.to_numeric(plot_df["Bin"], errors="coerce")
    plot_df = plot_df.dropna(subset=["X", "Y", "Bin"])
    plot_df = plot_df[plot_df["Bin"].isin([0, 1])]

    if len(figsize) != 2 or any(float(value) <= 0 for value in figsize):
        raise ValueError(f"figsize must contain two positive values, got {figsize!r}")
    if float(chip_aspect) <= 0:
        raise ValueError(f"chip_aspect must be positive, got {chip_aspect!r}")

    fig, ax = plt.subplots(
        figsize=(float(figsize[0]), float(figsize[1])),
        constrained_layout=True,
    )
    ok_df = plot_df[plot_df["Bin"] == 0]
    ng_df = plot_df[plot_df["Bin"] == 1]

    ax.scatter(ok_df["X"], ok_df["Y"], c="green", marker="s", s=36)
    ax.scatter(ng_df["X"], ng_df["Y"], c="red", marker="s", s=36)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Wafer Map - Overall Defect Result")
    # X/Y are chip-grid indices, not physical coordinates. One chip is about
    # five times wider than it is high, so one X step must occupy more pixels
    # than one Y step. Matplotlib's aspect is display-units(Y) / display-units(X).
    ax.set_aspect(1.0 / float(chip_aspect), adjustable="box")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="s", color="w", label="OK", 
                markerfacecolor="green", markersize=8,
            ),
            Line2D(
                [0], [0], marker="s", color="w", label="NG",
                markerfacecolor="red", markersize=8,
            ),
        ],
        loc="best",
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
