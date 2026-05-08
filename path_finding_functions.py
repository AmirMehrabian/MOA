import heapq
from typing import Optional

import numpy as np


def dominates(u: tuple, v: tuple) -> bool:
    """
    Returns True if cost vector u dominates cost vector v.
    Works for any number of objectives (2, 3, 4, ...).

    Two conditions must BOTH be true:
      Condition 1 — u is no worse than v in EVERY objective  (u[i] <= v[i])
      Condition 2 — u is strictly better in AT LEAST ONE    (u[i] <  v[i])
    """

    # ── Condition 1 ──────────────────────────────────────────────────────────
    # u must be no worse than v in every single objective.
    # One violation is enough to fail — return False immediately.

    for i in range(len(u)):
        if u[i] > v[i]:
            return False  # u is worse in objective i — cannot dominate

    # ── Condition 2 ──────────────────────────────────────────────────────────
    # Condition 1 passed — u is <= v in everything.
    # Now u must be strictly better in at least one objective.
    # One strict improvement is enough — return True immediately.

    for i in range(len(u)):
        if u[i] < v[i]:
            return True  # found a strict improvement — dominance holds

    # ── Both loops finished without triggering ────────────────────────────────
    # u and v must be exactly equal — equal is NOT dominance.

    return False


def is_dominated_by_frontier(vec: tuple, frontier: list) -> bool:
    """
    Returns True if 'vec' is dominated by any vector in 'frontier'.
    Used for:
      - Frontier Check  : compare g(l) against alpha(node)
      - Solution Check  : compare f(l) against alpha(goal)
    """
    return any(dominates(f, vec) for f in frontier)


def remove_dominated_from_frontier(new_vec: tuple, frontier: list) -> list:
    """
    Removes from 'frontier' every vector that 'new_vec' dominates.
    Called inside UpdateFrontier when a new label passes both checks.
    Returns the cleaned frontier.
    """
    return [f for f in frontier if not dominates(new_vec, f)]


def get_neighbors(node: tuple,
                  grid_rows: int,
                  grid_cols: int,
                  obstacles: np.ndarray) -> list:
    """
    Returns all valid 8-connected neighbors of 'node'.
    It assumes node starts at (0, 0) and ends at (grid_cols-1, grid_rows-1).
    Each neighbor is returned as (col, row, distance_cost) where:
      distance_cost = 1.0 for cardinal moves (up/down/left/right)
      distance_cost = 1.4 for diagonal moves (approximates sqrt(2))

    A neighbor is invalid if:
      - It falls outside the grid boundary
      - It is marked as an obstacle
    """
    col, row = node

    # All 8 directions: (delta_col, delta_row)
    directions = [
        (-1, -1), (0, -1), (1, -1),  # top-left,  top,   top-right
        (-1, 0), (1, 0),  # left,             right
        (-1, 1), (0, 1), (1, 1),  # bot-left,  bottom, bot-right
    ]

    valid_neighbors = []
    for dc, dr in directions:
        nc, nr = col + dc, row + dr

        # check grid boundary
        if not (0 <= nc < grid_cols and 0 <= nr < grid_rows):
            continue

        # check obstacle
        if obstacles[nr, nc]:
            continue

        # diagonal = 1.4, cardinal = 1.0
        is_diagonal = (dc != 0 and dr != 0)
        distance_cost = np.sqrt(2) if is_diagonal else 1.0

        valid_neighbors.append((nc, nr, distance_cost))

    return valid_neighbors


def compute_heuristic(node: tuple,
                      goal_node: tuple,
                      h2_map: Optional[np.ndarray] = None,
                      h3_map: Optional[np.ndarray] = None) -> tuple:
    """
    Admissible heuristic vector h(node) = (h1, h2, h3).

    h1 — Euclidean distance to goal.
         Always admissible for 8-connected grid (real path >= straight line).

    h2 — Minimum remaining jamming cost.
         If h2_map is provided (precomputed via backwards Dijkstra), use it.
         Otherwise defaults to 0.0 (always admissible, just less informed).

    h3 — Minimum remaining friendly deficit.
         Same logic as h2.

    Note: h2_map and h3_map should be precomputed ONCE before search starts
    by running single-objective Dijkstra backwards from the goal node.
    """
    col, row = node

    h1 = float(np.linalg.norm(np.array(node) - np.array(goal_node)))
    h2 = float(h2_map[row, col]) if h2_map is not None else 0.0
    h3 = float(h3_map[row, col]) if h3_map is not None else 0.0

    return h1, h2, h3


def compute_edge_cost(current_node: tuple,
                      child_node: tuple,
                      distance_cost: float,
                      grid_jam_pwr: np.ndarray,
                      grid_fav_deficit: np.ndarray) -> tuple:
    """
    Compute the cost vector of moving from current_node to child_node.

    Edge cost = cost of ENTERING child_node:
      c1 — distance    : 1.0 (cardinal) or 1.4 (diagonal)
      c2 — jamming     : jamming power at child_node
      c3 — deficit     : friendly power deficit at child_node
    """
    child_col, child_row = child_node

    c1 = distance_cost
    c2 = float(grid_jam_pwr[child_row, child_col])
    c3 = float(grid_fav_deficit[child_row, child_col])

    return c1, c2, c3


def create_start_label(start_node: tuple,
                       goal_node: tuple,
                       h2_map: Optional[np.ndarray] = None,
                       h3_map: Optional[np.ndarray] = None) -> dict:
    """
    Creates the very first label for the start node.
    g = (0, 0, 0) because no cost has been accumulated yet.
    """
    g = (0.0, 0.0, 0.0)
    h = compute_heuristic(start_node, goal_node, h2_map, h3_map)
    f = tuple(gi + hi for gi, hi in zip(g, h))

    return {
        "node": start_node,
        "g": g,  # accumulated cost so far
        "h": h,  # heuristic estimate to goal
        "f": f,  # g + h  → used to sort OPEN
        "parent": None  # no parent for start node
    }


def create_child_label(current_label: dict,
                       child_node: tuple,
                       distance_cost: float,
                       goal_node: tuple,
                       grid_jam_pwr: np.ndarray,
                       grid_fav_deficit: np.ndarray,
                       h2_map: Optional[np.ndarray] = None,
                       h3_map: Optional[np.ndarray] = None) -> dict:
    """
    Creates a new label for child_node by extending current_label.

    Steps:
      1. Compute edge cost  c = cost of entering child_node
      2. Accumulate         g_new = g_current + c   (element-wise)
      3. Compute heuristic  h_new = h(child_node)
      4. Compute estimate   f_new = g_new + h_new   (element-wise)
      5. Set parent pointer to current_label for path reconstruction
    """
    # step 1 — edge cost of entering child_node
    edge = compute_edge_cost(
        current_label["node"], child_node,
        distance_cost, grid_jam_pwr, grid_fav_deficit
    )

    # step 2 — new accumulated cost (element-wise addition)
    g_cur = current_label["g"]
    g_new = (g_cur[0] + edge[0],
             g_cur[1] + edge[1],
             g_cur[2] + edge[2])

    # step 3 — heuristic at child node
    h_new = compute_heuristic(child_node, goal_node, h2_map, h3_map)

    # step 4 — f = g + h
    f_new = (g_new[0] + h_new[0],
             g_new[1] + h_new[1],
             g_new[2] + h_new[2])

    # step 5 — full label
    return {
        "node": child_node,
        "g": g_new,
        "h": h_new,
        "f": f_new,
        "parent": current_label  # pointer for path reconstruction later
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — PATH RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_path(goal_label: dict) -> list:
    """
    Walks backwards through parent pointers from a goal label
    to rebuild the full path from start to goal.

    Returns a list of nodes in order: [start, ..., goal]
    """
    path = []
    current = goal_label

    while current is not None:
        path.append(current["node"])
        current = current["parent"]

    path.reverse()  # we built it backwards, flip it
    return path


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
        c = c + 1
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


def find_goal(grid_jam_pwr: np.ndarray,
              obstacles: np.ndarray,
              start: tuple,
              d_0: float) -> tuple:
    """
    Find the best goal cell within distance d_0 (in cells) from start.

    Steps
    -----
    1. Keep only cells within euclidean distance d_0 from start.
    2. Remove obstacle cells.
    3. Among remaining cells find the minimum jamming power.
    4. Among cells at that minimum, pick the closest one to start.

    Parameters
    ----------
    grid_jam_pwr : (rows, cols) normalised jammer cost [0, 1]
    obstacles    : (rows, cols) bool
    start        : (col, row) start node
    d_0          : maximum distance in cells from start to goal

    Returns
    -------
    (col, row) of the selected goal
    """
    rows, cols = grid_jam_pwr.shape
    start_col, start_row = start

    # Distance from start to every cell
    col_idx, row_idx = np.meshgrid(np.arange(cols), np.arange(rows))
    dist_from_start = np.sqrt((col_idx - start_col) ** 2 +
                              (row_idx - start_row) ** 2)

    # Valid: within range and not an obstacle
    valid = (dist_from_start <= d_0) & (~obstacles)

    if not valid.any():
        raise ValueError(f"No valid goal found within d_0={d_0} cells of {start}.")

    # Minimum jamming among valid cells
    jam_valid = np.where(valid, grid_jam_pwr, np.inf)
    min_jam = jam_valid.min()

    # Among cells at minimum jamming, pick closest to start
    at_min_jam = valid & (grid_jam_pwr == min_jam)
    dist_at_min = np.where(at_min_jam, dist_from_start, np.inf)
    best_idx = np.argmin(dist_at_min)
    goal_row, goal_col = np.unravel_index(best_idx, (rows, cols))

    goal = (int(goal_col), int(goal_row))
    print(f"[find_goal]  Goal: {goal}  "
          f"jam={min_jam:.3f}  "
          f"dist_from_start={dist_from_start[goal_row, goal_col]:.1f} cells")
    return goal