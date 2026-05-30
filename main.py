import matplotlib.pyplot as plt
import numpy as np

from RadioMap import build_radio_maps
from obstacles import make_obstacle_grid
from comm.channel_functions import init_channel_solver
from env_functions import env_response
from path_finding_functions import run_algorithm1_with_paths, find_goal, cell_to_position, position_to_cell
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

    START = position_to_cell(RX_POS[0], RX_POS[1], CELL_SIZE, origin_x, origin_y)
    print(f"START cell is {START}")
    GOAL = find_goal(grid_jam_pwr,
                     obstacles,
                     START, 200 / CELL_SIZE)
    print(f"GOAL found at {GOAL}")
    # GOAL = (36, 29)
    # ── Heuristic maps for objectives 2 and 3 ────────────────────────────────────
    Pj_goal = grid_jam_pwr[GOAL[1], GOAL[0]]
    Pd_goal = grid_fav_deficit[GOAL[1], GOAL[0]]

    h2_map = np.maximum(0, grid_jam_pwr - Pj_goal)
    h3_map = np.maximum(0, grid_fav_deficit - Pd_goal)

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

        for col, row in path:
            print(cell_to_position(col, row, CELL_SIZE, origin_x, origin_y, rx_height=1.5), end='->')
        print()
        for col, row in path:
            x, y, h = cell_to_position(col, row, CELL_SIZE, origin_x, origin_y, rx_height=1.5)
            print(position_to_cell(x, y, CELL_SIZE, origin_x, origin_y), end='->')
        print()
# ── build once, reuse across all metrics ─────────────────────────────────────
rm = build_radio_maps()
channel_dict = init_channel_solver(rm["scene"], config_dict)

metrics = ["distance", "jamming", "deficit"]
num_iter = 5
num_sym = config_dict["num_data_symbols"]

all_error = []
all_error_no_mit = []
all_labels = []

for i, metric in enumerate(metrics):
    costs = [cost[i] for cost in results["pareto_costs"]]
    best_idx = int(np.argmin(costs))
    best_path = results["pareto_paths"][best_idx]
    best_cost = results["pareto_costs"][best_idx]
    print(f"\nBest by {metric}: dist={best_cost[0]:.2f}  jam={best_cost[1]:.2f}  deficit={best_cost[2]:.2f}")

    error_vec = []  # one entry per cell on the path
    error_vec_no_mit = []

    for col, row in best_path:
        x, y, h = cell_to_position(col, row, CELL_SIZE, origin_x, origin_y, rx_height=1.5)  # uses simplified version

        cell_error = 0.0
        cell_error_no_mit = 0.0

        for ii in range(num_iter):
            z = env_response(config_dict, channel_dict, rm["scene"], rx_world_pos=(x, y, h))
            cell_error += z[0]
            cell_error_no_mit += z[2]

        # average over iterations, normalise by num_sym
        error_vec.append(cell_error / (num_iter * num_sym))
        error_vec_no_mit.append(cell_error_no_mit / (num_iter * num_sym))

    all_error.append(error_vec)
    all_error_no_mit.append(error_vec_no_mit)
    all_labels.append(metric)

    print(f"  BER along path (mitigated)    : {error_vec}")
    print(f"  BER along path (no mitigation): {error_vec_no_mit}")
    print(f"  Mean BER (mit)    : {np.mean(error_vec):.4f}")
    print(f"  Mean BER (no mit) : {np.mean(error_vec_no_mit):.4f}")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, ax in enumerate(axes):
    steps = range(len(all_error[i]))
    ax.plot(steps, all_error[i], lw=2, marker="o", markersize=3, label="mitigated")
    ax.plot(steps, all_error_no_mit[i], lw=2, marker="s", markersize=3, label="no mitigation", linestyle="--")
    ax.set_title(f"Best by {all_labels[i]}", fontsize=10)
    ax.set_xlabel("Path step")
    ax.set_ylabel("BER")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("BER along best path — per objective", fontsize=12)
plt.tight_layout()


# ── Plot 2: Mean BER bar chart ────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 4))
x = np.arange(len(metrics))
width = 0.35

ax2.bar(x - width / 2, [np.mean(e) for e in all_error], width, label="mitigated")
ax2.bar(x + width / 2, [np.mean(e) for e in all_error_no_mit], width, label="no mitigation")
ax2.set_xticks(x)
ax2.set_xticklabels([f"best by\n{m}" for m in all_labels])
ax2.set_ylabel("Mean BER")
ax2.set_title("Mean BER — best path per objective")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")
plt.tight_layout()


print("*" * 50)
for label, pos in [("TX (friendly)", FRIENDLY_POS),
                   ("JN (jammer)", JAMMER_POS),
                   ("RX", RX_POS)]:
    col, row = position_to_cell(pos[0], pos[1], CELL_SIZE, origin_x, origin_y)
    x_recal, y_recal, z_recal = cell_to_position(col, row, CELL_SIZE, origin_x, origin_y, rx_height=1.5)

    print(f"{label}")
    print(f"  real   : x={pos[0]:8.2f}  y={pos[1]:8.2f}  z={pos[2]:.2f}")
    print(f"  cell   : col={col:3d}  row={row:3d}")
    print(f"  recal  : x={x_recal:8.2f}  y={y_recal:8.2f}  z={z_recal:.2f}")
    print(f"  error  : dx={abs(pos[0] - x_recal):.3f} m  dy={abs(pos[1] - y_recal):.3f} m")
    print()

plot_grid_with_paths(results, obstacles, ROWS, COLS, START, GOAL)
plot_jamming_heatmap(grid_jam_pwr, obstacles, ROWS, COLS, START, GOAL)
plot_deficit_heatmap(grid_fav_deficit, obstacles, ROWS, COLS, START, GOAL)
plot_pareto_front(results)
plt.imshow(h2_map)
plt.title("Heuristic h2: Jamming Power")

plt.imshow(h3_map)
plt.title("Heuristic h2: Defecit friendly Power")
plt.show()
