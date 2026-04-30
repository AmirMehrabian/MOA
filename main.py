
import numpy as np
import heapq
from typing import Optional

from path_finding_functions import create_start_label, is_dominated_by_frontier, remove_dominated_from_frontier, \
    get_neighbors, create_child_label, reconstruct_path
from plot_functions import plot_grid_with_paths, plot_jamming_heatmap, plot_deficit_heatmap, plot_pareto_front


def run_algorithm1_with_paths(start_node: tuple,
                              goal_node: tuple,
                              grid_rows: int,
                              grid_cols: int,
                              obstacles: np.ndarray,
                              grid_jam_pwr: np.ndarray,
                              grid_fav_deficit: np.ndarray,
                              h2_map: Optional[np.ndarray] = None,
                              h3_map: Optional[np.ndarray] = None) -> dict:
    """
    Same as run_algorithm1 but also reconstructs the full node path
    for each Pareto-optimal solution.

    The difference: we store goal labels (with parent pointers) in
    a separate list so we can walk back through them after search ends.
    """

    alpha = {}
    OPEN = []
    counter = 0
    goal_labels = []  # stores every label that reached the goal
    expansions = 0

    start_label = create_start_label(start_node, goal_node, h2_map, h3_map)
    heapq.heappush(OPEN, (*start_label["f"], counter, start_label))
    counter += 1
    alpha[start_node] = []

    print(f"Search started:  {start_node}  →  {goal_node}")
    print(f"Grid: {grid_rows} rows × {grid_cols} cols\n")
    c = 0
    while OPEN:
        c = c+1
        print(c)
        *_, current = heapq.heappop(OPEN)
        node = current["node"]
        g = current["g"]
        f = current["f"]

        # ── Check 1 — Frontier check (compare g vs alpha(node)) ──────────────
        frontier_at_node = alpha.get(node, [])
        if is_dominated_by_frontier(g, frontier_at_node):
            continue

        # ── Check 2 — Solution check (compare f vs alpha(goal)) ──────────────
        frontier_at_goal = alpha.get(goal_node, [])
        if is_dominated_by_frontier(f, frontier_at_goal):
            continue

        # ── Update frontier ───────────────────────────────────────────────────
        alpha[node] = remove_dominated_from_frontier(g, frontier_at_node)
        alpha[node].append(g)

        # ── Goal reached ──────────────────────────────────────────────────────
        if node == goal_node:
            goal_labels.append(current)  # store label WITH parent pointer
            print(f"  Solution found!  g = {tuple(round(x, 2) for x in g)}")
            continue

        # ── Expand ───────────────────────────────────────────────────────────
        expansions += 1
        neighbors = get_neighbors(node, grid_rows, grid_cols, obstacles)

        for (child_col, child_row, dist_cost) in neighbors:
            child_node = (child_col, child_row)
            child_label = create_child_label(
                current, child_node, dist_cost,
                goal_node, grid_jam_pwr, grid_fav_deficit,
                h2_map, h3_map
            )
            g_new = child_label["g"]
            f_new = child_label["f"]

            child_frontier = alpha.get(child_node, [])
            if is_dominated_by_frontier(g_new, child_frontier):
                continue

            if is_dominated_by_frontier(f_new, alpha.get(goal_node, [])):
                continue

            heapq.heappush(OPEN, (*f_new, counter, child_label))
            counter += 1

    # ── Reconstruct all Pareto paths ──────────────────────────────────────────
    pareto_paths = [reconstruct_path(lbl) for lbl in goal_labels]
    pareto_costs = [lbl["g"] for lbl in goal_labels]

    print(f"\nSearch finished.")
    print(f"  Pareto-optimal solutions found : {len(pareto_costs)}")
    print(f"  Total label expansions         : {expansions}")
    for i, (cost, path) in enumerate(zip(pareto_costs, pareto_paths)):
        print(f"  Solution {i + 1}: "
              f"dist={cost[0]:.2f}  "
              f"jam={cost[1]:.2f}  "
              f"deficit={cost[2]:.2f}  "
              f"path_length={len(path)} nodes")

    return {
        "pareto_costs": pareto_costs,
        "pareto_paths": pareto_paths,
        "goal_labels": goal_labels,
        "alpha": alpha,
        "expansions": expansions,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — DEMO ON 5×5 GRID
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Grid is 5 rows × 5 cols
    # node convention: (col, row)  →  col=x, row=y
    # start = (0,4) bottom-left    goal = (4,0) top-right

    ROWS, COLS = 20, 20
    START = (0, 1)
    GOAL = (18, 19)

    # Obstacle map  (row, col indexing for numpy)
    # True = blocked,  matches the drawn grid: (2,1) and (1,3)
    obstacles = np.zeros((ROWS, COLS), dtype=bool)
    obstacles[1, 2] = True  # cell (col=2, row=1)
    obstacles[3, 1] = True  # cell (col=1, row=3)

    # Jamming power grid  (row, col)
    # Higher = more jammed = worse
    x = np.array([
        [2, 3, 5, 3, 1],  # row 0
        [3, 5, 0, 4, 2],  # row 1  (0 = obstacle, never visited)
        [4, 7, 9, 5, 3],  # row 2
        [3, 0, 6, 4, 2],  # row 3  (0 = obstacle)
        [2, 3, 5, 3, 2],  # row 4
    ], dtype=float)

    # Friendly power deficit grid  (row, col)
    # Higher = less friendly coverage = worse
    # x = np.array([
    #     [8, 6, 5, 3, 1],  # row 0
    #     [7, 6, 0, 3, 2],  # row 1
    #     [7, 6, 5, 4, 2],  # row 2
    #     [8, 0, 5, 4, 3],  # row 3
    #     [9, 8, 6, 4, 2],  # row 4
    # ], dtype=float)

    grid_x, grid_y = ROWS, COLS
    grid_jam_pwr = np.random.rand(grid_x, grid_y)
    #grid_jam_pwr = np.zeros((grid_x, grid_y))
    #grid_jam_pwr[0:5, 0:5] = x
    #grid_jam_pwr[45:50, 45:50] = x

    grid_fav_deficit = np.zeros((grid_x, grid_y)) #
    #grid_fav_deficit = np.random.rand(grid_x, grid_y)
    grid_obsticle = np.random.randint(2, size=(grid_x, grid_y))

    obstacles = np.zeros((grid_x, grid_y), dtype=bool)
    obstacles[1, 2] = True  # cell (col=2, row=1)
    obstacles[3, 1] = True  # cell (col=1, row=3)

    # Run the search
    results = run_algorithm1_with_paths(
        start_node=START,
        goal_node=GOAL,
        grid_rows=grid_x,
        grid_cols=grid_y,
        obstacles=obstacles,
        grid_jam_pwr=grid_jam_pwr,
        grid_fav_deficit=grid_fav_deficit,
        h2_map=None,  # use 0 for now — safe and admissible
        h3_map=None,
    )

    # Print each Pareto-optimal path
    print("\n── Pareto-optimal paths ──────────────────────────────────")
    for i, (path, cost) in enumerate(
            zip(results["pareto_paths"], results["pareto_costs"])):
        print(f"\nSolution {i + 1}:")
        print(f"Cost: dist={cost[0]:.2f}  jam={cost[1]:.2f}  deficit={cost[2]:.2f}")
        print(f"Path: {' → '.join(str(n) for n in path)}")



plot_grid_with_paths(results, obstacles, ROWS, COLS, START, GOAL)
plot_jamming_heatmap(grid_jam_pwr,     obstacles, ROWS, COLS, START, GOAL)
plot_deficit_heatmap(grid_fav_deficit, obstacles, ROWS, COLS, START, GOAL)
plot_pareto_front(results)