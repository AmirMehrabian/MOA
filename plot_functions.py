import matplotlib
#matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines  import Line2D


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_grid_base(ax, obstacles, rows, cols):

    for row in range(rows):
        for col in range(cols):

            if obstacles[row, col]:
                ax.add_patch(mpatches.Rectangle(
                    (col, row), 1, 1,
                    facecolor="#777777",
                    edgecolor="#555555",
                    linewidth=0.5
                ))
                ax.plot([col+0.2, col+0.8], [row+0.2, row+0.8],
                        color="#aaaaaa", lw=1.0)
                ax.plot([col+0.8, col+0.2], [row+0.2, row+0.8],
                        color="#aaaaaa", lw=1.0)

            else:
                ax.add_patch(mpatches.Rectangle(
                    (col, row), 1, 1,
                    facecolor="#f0f0f0",
                    edgecolor="#cccccc",
                    linewidth=0.4
                ))

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_xticklabels(np.arange(cols), fontsize=7)
    ax.set_yticklabels(np.arange(rows), fontsize=7)
    ax.tick_params(length=0)
    ax.set_xlabel("col", fontsize=8)
    ax.set_ylabel("row", fontsize=8)


def draw_start_goal(ax, start, goal):

    for (col, row), label, color in [
        (start, "S", "#2ea043"),
        (goal,  "G", "#1f6feb")
    ]:
        ax.add_patch(plt.Circle(
            (col + 0.5, row + 0.5), 0.35,
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=5
        ))
        ax.text(
            col + 0.5, row + 0.5, label,
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            color="white", zorder=6
        )


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1 — Grid with all Pareto paths
# ─────────────────────────────────────────────────────────────────────────────

def plot_grid_with_paths(results, obstacles, rows, cols,
                         start, goal,
                         save_path="figure1_paths.png"):
    """
    results — dict from run_algorithm1_with_paths()
              uses results["pareto_paths"] and results["pareto_costs"]
    """

    paths = results["pareto_paths"]
    costs = results["pareto_costs"]

    fig, ax = plt.subplots(figsize=(8, 8))

    draw_grid_base(ax, obstacles, rows, cols)

    colors = plt.colormaps.get_cmap("tab20")(
        np.linspace(0, 1, len(paths))
    )

    legend_handles = []

    for i, (path, g) in enumerate(zip(paths, costs)):

        col_coords = [node[0] + 0.5 for node in path]
        row_coords = [node[1] + 0.5 for node in path]

        ax.plot(
            col_coords, row_coords,
            color=colors[i],
            linewidth=1.8,
            alpha=0.80,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3
        )

        legend_handles.append(Line2D(
            [0], [0],
            color=colors[i],
            linewidth=2,
            label=(
                f"Sol {i+1:2d}  "
                f"d={g[0]:.1f}  "
                f"j={g[1]:.1f}  "
                f"f={g[2]:.1f}"
            )
        ))

    draw_start_goal(ax, start, goal)

    ax.set_title(
        f"MOA*  —  {len(paths)} Pareto-Optimal Paths\n"
        f"d=distance   j=jamming   f=friendly deficit",
        fontsize=10, pad=8
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        framealpha=0.9,
        title="  Sol    dist    jam    def",
        title_fontsize=7
    )

    plt.tight_layout()
   # plt.savefig(save_path, dpi=150, bbox_inches="tight")
   # print(f"  Saved  {save_path}")
   # plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2 — Jamming heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_jamming_heatmap(grid_jam, obstacles, rows, cols,
                         start, goal,
                         save_path="figure2_jamming.png"):

    cmap = LinearSegmentedColormap.from_list(
        "jam", ["#d4f1d4", "#ffcc00", "#ff4500"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    masked = np.ma.masked_where(obstacles, grid_jam)

    im = ax.imshow(
        masked,
        cmap=cmap,
        origin="upper",
        extent=[0, cols, rows, 0],
        aspect="equal",
        vmin=grid_jam[~obstacles].min(),
        vmax=grid_jam[~obstacles].max()
    )

    for row in range(rows):
        for col in range(cols):
            if obstacles[row, col]:
                ax.add_patch(plt.Rectangle(
                    (col, row), 1, 1,
                    facecolor="#888888",
                    edgecolor="#666666",
                    linewidth=0.5
                ))
                ax.plot([col+0.2, col+0.8], [row+0.2, row+0.8],
                        color="#aaaaaa", lw=0.8)
                ax.plot([col+0.8, col+0.2], [row+0.2, row+0.8],
                        color="#aaaaaa", lw=0.8)

            else:
                ax.text(
                    col + 0.5, row + 0.5,
                    str(int(grid_jam[row, col])),
                    ha="center", va="center",
                    fontsize=7, color="black", alpha=0.7
                )

    for x in range(cols + 1):
        ax.axvline(x, color="#cccccc", linewidth=0.3)
    for y in range(rows + 1):
        ax.axhline(y, color="#cccccc", linewidth=0.3)

    draw_start_goal(ax, start, goal)

    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(labelsize=7)

    ax.set_title("Jamming Power  (high = red = worse)", fontsize=10, pad=8)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_xticklabels(np.arange(cols), fontsize=7)
    ax.set_yticklabels(np.arange(rows), fontsize=7)
    ax.tick_params(length=0)
    ax.set_xlabel("col", fontsize=8)
    ax.set_ylabel("row", fontsize=8)

    plt.tight_layout()
   # plt.savefig(save_path, dpi=150, bbox_inches="tight")
   # print(f"  Saved  {save_path}")
   # plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 — Friendly deficit heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_deficit_heatmap(grid_deficit, obstacles, rows, cols,
                         start, goal,
                         save_path="figure3_deficit.png"):

    cmap = LinearSegmentedColormap.from_list(
        "def", ["#c8f5c8", "#ffe066", "#ff8c00"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    masked = np.ma.masked_where(obstacles, grid_deficit)

    im = ax.imshow(
        masked,
        cmap=cmap,
        origin="upper",
        extent=[0, cols, rows, 0],
        aspect="equal",
        vmin=grid_deficit[~obstacles].min(),
        vmax=grid_deficit[~obstacles].max()
    )

    for row in range(rows):
        for col in range(cols):
            if obstacles[row, col]:
                ax.add_patch(plt.Rectangle(
                    (col, row), 1, 1,
                    facecolor="#888888",
                    edgecolor="#666666",
                    linewidth=0.5
                ))
                ax.plot([col+0.2, col+0.8], [row+0.2, row+0.8],
                        color="#aaaaaa", lw=0.8)
                ax.plot([col+0.8, col+0.2], [row+0.2, row+0.8],
                        color="#aaaaaa", lw=0.8)

            else:
                ax.text(
                    col + 0.5, row + 0.5,
                    str(int(grid_deficit[row, col])),
                    ha="center", va="center",
                    fontsize=7, color="black", alpha=0.7
                )

    for x in range(cols + 1):
        ax.axvline(x, color="#cccccc", linewidth=0.3)
    for y in range(rows + 1):
        ax.axhline(y, color="#cccccc", linewidth=0.3)

    draw_start_goal(ax, start, goal)

    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(labelsize=7)

    ax.set_title("Friendly Deficit  (high = orange = worse)", fontsize=10, pad=8)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_xticklabels(np.arange(cols), fontsize=7)
    ax.set_yticklabels(np.arange(rows), fontsize=7)
    ax.tick_params(length=0)
    ax.set_xlabel("col", fontsize=8)
    ax.set_ylabel("row", fontsize=8)

    plt.tight_layout()
   # plt.savefig(save_path, dpi=150, bbox_inches="tight")
    #print(f"  Saved  {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4 — Pareto front scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_front(results, save_path="figure4_pareto_front.png"):
    """
    results — dict from run_algorithm1_with_paths()
              uses results["pareto_costs"]
    X = jamming   Y = deficit   colour = distance
    """

    costs = results["pareto_costs"]

    dist_vals    = [g[0] for g in costs]
    jam_vals     = [g[1] for g in costs]
    deficit_vals = [g[2] for g in costs]

    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(
        jam_vals, deficit_vals,
        c=dist_vals,
        cmap="plasma_r",
        s=80,
        edgecolors="grey",
        linewidths=0.6,
        zorder=3
    )

    for i, (j, d) in enumerate(zip(jam_vals, deficit_vals)):
        ax.annotate(
            str(i + 1), (j, d),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
            color="#444444"
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Distance", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Jamming (accumulated)",  fontsize=9)
    ax.set_ylabel("Deficit (accumulated)",  fontsize=9)
    ax.set_title("Pareto Front — Objective Space", fontsize=10, pad=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
  #  plt.savefig(save_path, dpi=150, bbox_inches="tight")
  #  print(f"  Saved  {save_path}")
   # plt.close()