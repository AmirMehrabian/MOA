import numpy as np

from RadioMap import build_radio_maps
from obstacles import make_obstacle_grid
from path_finding_functions import run_algorithm1_with_paths, find_goal
from plot_functions import plot_grid_with_paths, plot_jamming_heatmap, plot_deficit_heatmap, plot_pareto_front
from config import *

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — DEMO ON 5×5 GRID
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Grid is 5 rows × 5 cols
    # node convention: (col, row)  →  col=x, row=y
    # start = (0,4) bottom-left    goal = (4,0) top-right



    maps = build_radio_maps()

    grid_jam_pwr = maps["grid_jam_pwr"]  # (H, W)  [0, 1]
    grid_fav_deficit = maps["grid_fav_deficit"]  # (H, W)  [0, 1]
    ROWS, COLS = grid_jam_pwr.shape
    # ── Build obstacle grid ───────────────────────────────────────────────────────
    obstacles = make_obstacle_grid(
        rows=ROWS,
        cols=COLS,
        cell_size=[CELL_SIZE, CELL_SIZE],
        center=[CENTER_X, CENTER_Y, 0],
        size=[X_MAX - X_MIN, Y_MAX - Y_MIN],
    )

    GOAL = find_goal(grid_jam_pwr,
                     obstacles,
                     START, 1000/CELL_SIZE)
    print(f"GOAL found at {GOAL}")
    GOAL = (36, 29)
    # ── Heuristic maps for objectives 2 and 3 ────────────────────────────────────
    Pj_goal = grid_jam_pwr[GOAL[1], GOAL[0]]
    Pd_goal = grid_fav_deficit[GOAL[1], GOAL[0]]

    h2_map = np.maximum(0, Pj_goal - grid_jam_pwr)
    h3_map = np.maximum(0, Pd_goal - grid_fav_deficit)
    # Obstacle map  (row, col indexing for numpy)
    # True = blocked,  matches the drawn grid: (2,1) and (1,3)
    # obstacles = np.zeros((ROWS, COLS), dtype=bool)
    # obstacles[1, 2] = True  # cell (col=2, row=1)
    # obstacles[3, 1] = True  # cell (col=1, row=3)
    #
    # # Jamming power grid  (row, col)
    # # Higher = more jammed = worse
    # x = np.array([
    #     [2, 3, 5, 3, 1],  # row 0
    #     [3, 5, 0, 4, 2],  # row 1  (0 = obstacle, never visited)
    #     [4, 7, 9, 5, 3],  # row 2
    #     [3, 0, 6, 4, 2],  # row 3  (0 = obstacle)
    #     [2, 3, 5, 3, 2],  # row 4
    # ], dtype=float)
    #
    # # Friendly power deficit grid  (row, col)
    # # Higher = less friendly coverage = worse
    # # x = np.array([
    # #     [8, 6, 5, 3, 1],  # row 0
    # #     [7, 6, 0, 3, 2],  # row 1
    # #     [7, 6, 5, 4, 2],  # row 2
    # #     [8, 0, 5, 4, 3],  # row 3
    # #     [9, 8, 6, 4, 2],  # row 4
    # # # ], dtype=float)
    #
    #grid_x, grid_y = ROWS, COLS
    # grid_jam_pwr = np.random.rand(grid_x, grid_y)
    #grid_jam_pwr = np.zeros((grid_x, grid_y))
    # #grid_jam_pwr[0:5, 0:5] = x
    # #grid_jam_pwr[45:50, 45:50] = x
    #
    #grid_fav_deficit = np.zeros((ROWS, COLS))  #
    # #grid_fav_deficit = np.random.rand(grid_x, grid_y)
    # grid_obsticle = np.random.randint(2, size=(grid_x, grid_y))
    #
    # obstacles = np.zeros((grid_x, grid_y), dtype=bool)
    # obstacles[1, 2] = True  # cell (col=2, row=1)
    # obstacles[3, 1] = True  # cell (col=1, row=3)

    # Run the search
    results = run_algorithm1_with_paths(
        start_node=START,
        goal_node=GOAL,
        grid_rows=ROWS,
        grid_cols=COLS,
        obstacles=obstacles,
        grid_jam_pwr=grid_jam_pwr,
        grid_fav_deficit=grid_fav_deficit,
        h2_map=h2_map,  # use 0 for now — safe and admissible
        h3_map=h3_map,
    )

    # Print each Pareto-optimal path
    print("\n── Pareto-optimal paths ──────────────────────────────────")
    for i, (path, cost) in enumerate(
            zip(results["pareto_paths"], results["pareto_costs"])):
        print(f"\nSolution {i + 1}:")
        print(f"Cost: dist={cost[0]:.2f}  jam={cost[1]:.2f}  deficit={cost[2]:.2f}")
        print(f"Path: {' → '.join(str(n) for n in path)}")

plot_grid_with_paths(results, obstacles, ROWS, COLS, START, GOAL)
plot_jamming_heatmap(grid_jam_pwr, obstacles, ROWS, COLS, START, GOAL)
plot_deficit_heatmap(grid_fav_deficit, obstacles, ROWS, COLS, START, GOAL)
plot_pareto_front(results)
