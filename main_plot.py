import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from RadioMap import build_radio_maps
from config import ROWS, COLS, CELL_SIZE, CENTER_X, CENTER_Y, X_MAX, X_MIN, Y_MIN, Y_MAX
from obstacles import make_obstacle_grid

# ── Floor bounds (from mesh-floor vertex_positions) ───────────────────────────


print(f"Grid: {ROWS} rows × {COLS} cols  |  cell = {CELL_SIZE} m")
print(f"Center: ({CENTER_X:.2f}, {CENTER_Y:.2f})")

# ── Build radio maps ──────────────────────────────────────────────────────────
maps = build_radio_maps()

grid_jam_pwr = maps["grid_jam_pwr"]  # (H, W)  [0, 1]
grid_fav_deficit = maps["grid_fav_deficit"]  # (H, W)  [0, 1]

# ── Build obstacle grid ───────────────────────────────────────────────────────
obstacles = make_obstacle_grid(
    rows=ROWS,
    cols=COLS,
    cell_size=[CELL_SIZE, CELL_SIZE],
    center=[CENTER_X, CENTER_Y, 0],
    size=[X_MAX - X_MIN, Y_MAX - Y_MIN],
)

# ── Plot all three maps ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

extent = [X_MIN, X_MAX, Y_MIN, Y_MAX]  # for world-coordinate axes


# Helper: overlay obstacle hatching on any ax
def _overlay_obstacles(ax):
    ax.imshow(obstacles, origin="lower", extent=extent, aspect="auto",
              cmap="Greys", alpha=0.35, interpolation="nearest")


# ── Panel 1: Jamming power ────────────────────────────────────────────────────
jam_cmap = LinearSegmentedColormap.from_list("jam", ["#d4f1d4", "#ffcc00", "#ff4500"])
im1 = axes[0].imshow(grid_jam_pwr, origin="lower", extent=extent,
                     aspect="auto", cmap=jam_cmap, vmin=0, vmax=1)
_overlay_obstacles(axes[0])
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04).set_label("[0 = weak, 1 = strong]")
axes[0].set_title("Jammer TX — Path Gain Cost", fontweight="bold")
axes[0].set_xlabel("x [m]");
axes[0].set_ylabel("y [m]")

# ── Panel 2: Friendly deficit ─────────────────────────────────────────────────
def_cmap = LinearSegmentedColormap.from_list("def", ["#c8f5c8", "#ffe066", "#ff8c00"])
im2 = axes[1].imshow(grid_fav_deficit, origin="lower", extent=extent,
                     aspect="auto", cmap=def_cmap, vmin=0, vmax=1)
_overlay_obstacles(axes[1])
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04).set_label("[0 = covered, 1 = no coverage]")
axes[1].set_title("Friendly TX — Coverage Deficit", fontweight="bold")
axes[1].set_xlabel("x [m]");
axes[1].set_ylabel("y [m]")

# ── Panel 3: Obstacles only ───────────────────────────────────────────────────
axes[2].imshow(obstacles, origin="lower", extent=extent,
               aspect="auto", cmap="Greys", interpolation="nearest")
axes[2].set_title("Building Obstacles", fontweight="bold")
axes[2].set_xlabel("x [m]");
axes[2].set_ylabel("y [m]")

# Reference lines at origin
for ax in axes:
    ax.axhline(0, color="white", lw=0.5, alpha=0.4)
    ax.axvline(0, color="white", lw=0.5, alpha=0.4)

fig.suptitle("Étoile Scene — MOA* Input Grids", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("moa_input_grids.png", dpi=150, bbox_inches="tight")
plt.show()
