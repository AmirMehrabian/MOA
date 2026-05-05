"""
obstacles.py
============
Hardcoded building footprints parsed from the Étoile scene vertex data.
Each building is defined as (x_min, x_max, y_min, y_max) in world metres.

Usage
-----
    from obstacles import make_obstacle_grid
    obstacles = make_obstacle_grid(rows, cols, cell_size, center, size)
"""

import numpy as np

# ── Building footprints (x_min, x_max, y_min, y_max) in world metres ─────────
# Derived from vertex_positions, keeping only XY extent (z ignored here).

BUILDINGS = [
    # name                   x_min    x_max    y_min    y_max
    ("building_2",           32.36,   63.48,   10.34,   38.22),   # 1 building

    ("obj_250497792_A",     -15.12,   16.00,    9.57,   37.46),   # \  merged
    ("obj_250497792_B",     -62.41,  -31.29,    9.57,   37.46),   # /  object

    ("building_6",          -15.12,   16.00,  -36.50,   -8.61),   # 1 building

    ("obj_250526992_A",     -62.11,  -30.99,  -36.50,   -8.61),   # \  merged
    ("obj_250526992_B",      31.52,   62.64,  -36.50,   -8.61),   # /  object
]


def make_obstacle_grid(rows: int,
                       cols: int,
                       cell_size: list,
                       center: list,
                       size: list) -> np.ndarray:
    """
    Convert BUILDINGS footprints into a boolean (rows, cols) obstacle grid.

    Each building bounding box is filled entirely with True.
    Grid cell (row, col) corresponds to world point:
        x = origin_x + (col + 0.5) * dx
        y = origin_y + (row + 0.5) * dy

    Parameters match SOLVER_CFG in RadioMap.py.
    """
    obstacles = np.zeros((rows, cols), dtype=bool)

    dx, dy   = cell_size[0], cell_size[1]
    origin_x = center[0] - size[0] / 2
    origin_y = center[1] - size[1] / 2

    for name, x_min, x_max, y_min, y_max in BUILDINGS:

        # Convert world metres → grid indices
        col_lo = int((x_min - origin_x) / dx)
        col_hi = int((x_max - origin_x) / dx)
        row_lo = int((y_min - origin_y) / dy)
        row_hi = int((y_max - origin_y) / dy)

        # Clamp to grid bounds
        col_lo = max(col_lo, 0);   col_hi = min(col_hi, cols - 1)
        row_lo = max(row_lo, 0);   row_hi = min(row_hi, rows - 1)

        # Fill the bounding box
        obstacles[row_lo:row_hi + 1, col_lo:col_hi + 1] = True

        print(f"  {name:22s}  "
              f"cols [{col_lo:3d}→{col_hi:3d}]  "
              f"rows [{row_lo:3d}→{row_hi:3d}]  "
              f"cells={(col_hi-col_lo+1)*(row_hi-row_lo+1)}")

    print(f"\n  Total blocked: {obstacles.sum()} / {rows*cols} cells "
          f"({100*obstacles.mean():.1f}%)")
    return obstacles


# ── Quick visual check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Match RadioMap.py SOLVER_CFG exactly
    ROWS = COLS = 100
    obs = make_obstacle_grid(
        rows=ROWS, cols=COLS,
        cell_size=[2, 2],
        center=[0, 0, 0],
        size=[200, 200],
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(obs, origin="lower", cmap="Greys", interpolation="nearest")
    plt.title("Building obstacles — Étoile scene")
    plt.xlabel("col (x)");  plt.ylabel("row (y)")
    plt.tight_layout()
    plt.show()