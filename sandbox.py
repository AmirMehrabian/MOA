"""
Algorithm 1 — Multi-Objective A* (MOA*) for Jammer-Aware Path Planning
=======================================================================
Three objectives:
  g1 — distance        (minimize)
  g2 — jamming power   (minimize)
  g3 — friendly deficit (minimize)

Grid convention:
  node = (col, row)  →  col=x (left→right), row=y (top→bottom)
  cost arrays indexed as  grid[row, col]  (numpy standard)

Reference:
  Ren et al. (2022) — EMOA*, Algorithm 1 (Search Framework)
  Mandow & Pérez de la Cruz (2010) — NAMOA*
"""

import numpy as np
import heapq
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — DOMINANCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def dominates(u: tuple, v: tuple) -> bool:
    """
    Returns True if cost vector u dominates vector v.

    u dominates v means:
      - u is no worse than v in ALL objectives
      - u is strictly better than v in AT LEAST ONE objective
    """
    at_least_as_good  = all(ui <= vi for ui, vi in zip(u, v))
    strictly_better   = any(ui <  vi for ui, vi in zip(u, v))
    return at_least_as_good and strictly_better


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


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — GRID HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_neighbors(node: tuple,
                  grid_rows: int,
                  grid_cols: int,
                  obstacles: np.ndarray) -> list:
    """
    Returns all valid 8-connected neighbors of 'node'.

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
        (-1, -1), (0, -1), (1, -1),   # top-left,  top,   top-right
        (-1,  0),           (1,  0),   # left,             right
        (-1,  1), (0,  1), (1,  1),   # bot-left,  bottom, bot-right
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
        is_diagonal   = (dc != 0 and dr != 0)
        distance_cost = 1.4 if is_diagonal else 1.0

        valid_neighbors.append((nc, nr, distance_cost))

    return valid_neighbors


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — LABEL CREATION
# ─────────────────────────────────────────────────────────────────────────────

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

    return (h1, h2, h3)


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

    return (c1, c2, c3)


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
        "node"   : start_node,
        "g"      : g,          # accumulated cost so far
        "h"      : h,          # heuristic estimate to goal
        "f"      : f,          # g + h  → used to sort OPEN
        "parent" : None        # no parent for start node
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
        "node"   : child_node,
        "g"      : g_new,
        "h"      : h_new,
        "f"      : f_new,
        "parent" : current_label   # pointer for path reconstruction later
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

    path.reverse()   # we built it backwards, flip it
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — MAIN ALGORITHM (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm1(start_node: tuple,
                   goal_node: tuple,
                   grid_rows: int,
                   grid_cols: int,
                   obstacles: np.ndarray,
                   grid_jam_pwr: np.ndarray,
                   grid_fav_deficit: np.ndarray,
                   h2_map: Optional[np.ndarray] = None,
                   h3_map: Optional[np.ndarray] = None) -> dict:
    """
    Multi-Objective A* — Algorithm 1 (lazy check variant).

    Returns a dict with:
      "pareto_labels" : list of goal labels on the Pareto front
                        (each contains g-vector and parent pointer)
      "pareto_costs"  : list of g-vectors at goal (the Pareto front)
      "pareto_paths"  : list of node-sequences, one per Pareto solution
      "alpha"         : the full frontier dict (for inspection)
      "expansions"    : how many labels were actually expanded
    """

    # ── SETUP ────────────────────────────────────────────────────────────────
    #
    # alpha[node] = list of non-dominated g-vectors found for that node.
    # We use a dict so we only allocate entries for nodes we actually visit.
    #
    alpha = {}          # node → list of g-vectors (the frontier)

    # OPEN is a min-heap sorted by (f1, f2, f3, counter).
    # counter is a unique integer that breaks ties without comparing labels.
    OPEN  = []
    counter = 0

    # Create and push the start label
    start_label = create_start_label(start_node, goal_node, h2_map, h3_map)
    heapq.heappush(OPEN, (*start_label["f"], counter, start_label))
    counter += 1

    # Initialise the frontier for start node
    alpha[start_node] = []

    # Track how many labels we fully expanded (useful for analysis)
    expansions = 0

    print(f"Search started:  {start_node}  →  {goal_node}")
    print(f"Grid: {grid_rows} rows × {grid_cols} cols\n")

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────
    while OPEN:

        # Pop the label with lex-minimum f-vector
        *_, popped_label = heapq.heappop(OPEN)
        node = popped_label["node"]
        g    = popped_label["g"]
        f    = popped_label["f"]

        # ── CHECK 1 — FRONTIER CHECK ─────────────────────────────────────────
        # "Is this label already beaten by a better path to the same node?"
        #
        # We compare g(l) against g-vectors in alpha(node).
        # Same node → same future heuristic → comparing g is sufficient.
        #
        # This is the LAZY CHECK: the label may have been valid when pushed
        # but become stale while sitting in OPEN.
        #
        frontier_at_node = alpha.get(node, [])
        if is_dominated_by_frontier(g, frontier_at_node):
            continue    # stale — skip silently

        # ── CHECK 2 — SOLUTION CHECK ─────────────────────────────────────────
        # "Is this label beaten by a known complete solution?"
        #
        # We compare f(l) against g-vectors in alpha(goal).
        # f(l) = g(l) + h(node) is the BEST this label could ever achieve.
        # g-vectors at goal equal f-vectors there since h(goal) = (0,0,0).
        #
        frontier_at_goal = alpha.get(goal_node, [])
        if is_dominated_by_frontier(f, frontier_at_goal):
            continue    # a known solution already beats even the best case

        # ── UPDATE FRONTIER ───────────────────────────────────────────────────
        # Label passed both checks — it is genuinely useful.
        # 1. Remove from alpha(node) any vector that g now dominates
        # 2. Add g to alpha(node)
        #
        alpha[node] = remove_dominated_from_frontier(g, frontier_at_node)
        alpha[node].append(g)

        # ── GOAL CHECK ────────────────────────────────────────────────────────
        # If we just reached the goal, record the solution but do NOT expand.
        # We keep searching — more Pareto-optimal solutions may still exist.
        #
        if node == goal_node:
            print(f"  Solution found!  g = {tuple(round(x,2) for x in g)}")
            continue    # record done, do not expand goal

        # ── EXPAND ────────────────────────────────────────────────────────────
        # Generate all valid 8-connected neighbors and apply both checks
        # before adding them to OPEN.
        #
        expansions += 1
        neighbors = get_neighbors(node, grid_rows, grid_cols, obstacles)

        for (child_col, child_row, dist_cost) in neighbors:
            child_node = (child_col, child_row)

            # Build the child label
            child_label = create_child_label(
                popped_label, child_node, dist_cost,
                goal_node, grid_jam_pwr, grid_fav_deficit,
                h2_map, h3_map
            )
            g_new = child_label["g"]
            f_new = child_label["f"]

            # Apply Check 1 at generation time (early rejection)
            child_frontier = alpha.get(child_node, [])
            if is_dominated_by_frontier(g_new, child_frontier):
                continue

            # Apply Check 2 at generation time (early rejection)
            if is_dominated_by_frontier(f_new, frontier_at_goal):
                continue

            # Push onto OPEN
            heapq.heappush(OPEN, (*f_new, counter, child_label))
            counter += 1

    # ── COLLECT RESULTS ───────────────────────────────────────────────────────
    # Walk through all goal labels in alpha and reconstruct paths.
    # We need to recover the actual labels (not just g-vectors) for path
    # reconstruction, so we keep goal labels in a separate list.
    #
    # NOTE: We cannot reconstruct paths from g-vectors alone — we need
    # the parent pointers. So we need to store goal labels separately.
    # Modify the goal check above to do this:
    # (See the revised version below in run_algorithm1_with_paths)

    pareto_costs = alpha.get(goal_node, [])
    print(f"\nSearch finished.")
    print(f"  Pareto-optimal solutions found : {len(pareto_costs)}")
    print(f"  Total label expansions         : {expansions}")
    for i, cost in enumerate(pareto_costs):
        print(f"  Solution {i+1}: dist={cost[0]:.2f}  jam={cost[1]:.2f}  deficit={cost[2]:.2f}")

    return {
        "pareto_costs" : pareto_costs,
        "alpha"        : alpha,
        "expansions"   : expansions,
    }


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

    alpha        = {}
    OPEN         = []
    counter      = 0
    goal_labels  = []   # stores every label that reached the goal
    expansions   = 0

    start_label  = create_start_label(start_node, goal_node, h2_map, h3_map)
    heapq.heappush(OPEN, (*start_label["f"], counter, start_label))
    counter += 1
    alpha[start_node] = []

    print(f"Search started:  {start_node}  →  {goal_node}")
    print(f"Grid: {grid_rows} rows × {grid_cols} cols\n")

    while OPEN:

        *_, current = heapq.heappop(OPEN)
        node = current["node"]
        g    = current["g"]
        f    = current["f"]

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
            goal_labels.append(current)   # store label WITH parent pointer
            print(f"  Solution found!  g = {tuple(round(x,2) for x in g)}")
            continue

        # ── Expand ───────────────────────────────────────────────────────────
        expansions += 1
        neighbors = get_neighbors(node, grid_rows, grid_cols, obstacles)

        for (child_col, child_row, dist_cost) in neighbors:
            child_node  = (child_col, child_row)
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
    pareto_paths  = [reconstruct_path(lbl) for lbl in goal_labels]
    pareto_costs  = [lbl["g"] for lbl in goal_labels]

    print(f"\nSearch finished.")
    print(f"  Pareto-optimal solutions found : {len(pareto_costs)}")
    print(f"  Total label expansions         : {expansions}")
    for i, (cost, path) in enumerate(zip(pareto_costs, pareto_paths)):
        print(f"  Solution {i+1}: "
              f"dist={cost[0]:.2f}  "
              f"jam={cost[1]:.2f}  "
              f"deficit={cost[2]:.2f}  "
              f"path_length={len(path)} nodes")

    return {
        "pareto_costs"  : pareto_costs,
        "pareto_paths"  : pareto_paths,
        "goal_labels"   : goal_labels,
        "alpha"         : alpha,
        "expansions"    : expansions,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — DEMO ON 5×5 GRID
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Grid is 5 rows × 5 cols
    # node convention: (col, row)  →  col=x, row=y
    # start = (0,4) bottom-left    goal = (4,0) top-right

    ROWS, COLS = 5, 5
    START      = (0, 4)
    GOAL       = (4, 0)

    # Obstacle map  (row, col indexing for numpy)
    # True = blocked,  matches the drawn grid: (2,1) and (1,3)
    obstacles = np.zeros((ROWS, COLS), dtype=bool)
    obstacles[1, 2] = True    # cell (col=2, row=1)
    obstacles[3, 1] = True    # cell (col=1, row=3)

    # Jamming power grid  (row, col)
    # Higher = more jammed = worse
    grid_jam_pwr = np.array([
        [2, 3, 5, 3, 1],   # row 0
        [3, 5, 0, 4, 2],   # row 1  (0 = obstacle, never visited)
        [4, 7, 9, 5, 3],   # row 2
        [3, 0, 6, 4, 2],   # row 3  (0 = obstacle)
        [2, 3, 5, 3, 2],   # row 4
    ], dtype=float)

    # Friendly power deficit grid  (row, col)
    # Higher = less friendly coverage = worse
    grid_fav_deficit = np.array([
        [8, 6, 5, 3, 1],   # row 0
        [7, 6, 0, 3, 2],   # row 1
        [7, 6, 5, 4, 2],   # row 2
        [8, 0, 5, 4, 3],   # row 3
        [9, 8, 6, 4, 2],   # row 4
    ], dtype=float)

    # Run the search
    results = run_algorithm1_with_paths(
        start_node       = START,
        goal_node        = GOAL,
        grid_rows        = ROWS,
        grid_cols        = COLS,
        obstacles        = obstacles,
        grid_jam_pwr     = grid_jam_pwr,
        grid_fav_deficit = grid_fav_deficit,
        h2_map           = None,   # use 0 for now — safe and admissible
        h3_map           = None,
    )

    # Print each Pareto-optimal path
    print("\n── Pareto-optimal paths ──────────────────────────────────")
    for i, (path, cost) in enumerate(
            zip(results["pareto_paths"], results["pareto_costs"])):
        print(f"\nSolution {i+1}:")
        print(f"  Cost   : dist={cost[0]:.2f}  jam={cost[1]:.2f}  deficit={cost[2]:.2f}")
        print(f"  Path   : {' → '.join(str(n) for n in path)}")