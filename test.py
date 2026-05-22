from comm.channel_functions import init_channel_solver
from config import *
from env_functions import env_response
from comm.channel_functions import init_channel_solver
from RadioMap import build_radio_maps
import matplotlib.pyplot as plt
from pprint import pprint

snr_range = np.arange(-34, 10, 2)
num_iter = 10

rx_pos = config_dict["rx_pos"]
num_sym = config_dict["num_data_symbols"]

results = build_radio_maps()
channel_dict = init_channel_solver(results["scene"], config_dict)

error_vec = []
error_vec_no_mit = []

print("Channel dict:")
pprint(channel_dict)
print("Config dict:")
pprint(config_dict)

for snr in snr_range:

    error = 0
    error_no_mit = 0

    for ii in range(num_iter):
        config_dict["snr_tn"] = snr

        z = env_response(config_dict, channel_dict, results["scene"], rx_world_pos=rx_pos)
        error += z[0] / num_sym
        error_no_mit += z[2] / num_sym

    print(f"Testing SNR={snr} dB- SAJ error: {z[0]} sym per frame: {z[1]}- No Mitigation error: {z[2]} sym per frame: {z[3]}")
    error_vec.append(error / num_iter)
    error_vec_no_mit.append(error_no_mit / num_iter)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

# Main curves
ax.semilogy(snr_range, error_vec,
            color="#2196F3", linewidth=2.2, marker="o",
            markersize=6, markerfacecolor="white", markeredgewidth=2,
            label="With Mitigation")

ax.semilogy(snr_range, error_vec_no_mit,
            color="#F44336", linewidth=2.2, marker="s",
            markersize=6, markerfacecolor="white", markeredgewidth=2,
            linestyle="--", label="No Mitigation")

# ── Axes labels & limits ──────────────────────────────────────────────────────
ax.set_xlabel("SNR (dB)", fontsize=13, labelpad=8)
ax.set_ylabel("Symbol Error Rate (SER)", fontsize=13, labelpad=8)
ax.set_xlim(snr_range[0], snr_range[-1])
ax.set_ylim(1e-4, 1.2)

# ── Grid ──────────────────────────────────────────────────────────────────────
ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.5)
ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
ax.set_axisbelow(True)

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(fontsize=11, framealpha=0.9, edgecolor="#cccccc",
          loc="upper right", handlelength=2.5)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title("SER vs. SNR — Interference Mitigation Comparison",
             fontsize=14, fontweight="bold", pad=12)

# ── Gain annotation (arrow between curves at a chosen SNR point) ──────────────
ref_idx = len(snr_range) // 2  # pick the middle SNR point
ref_snr = snr_range[ref_idx]
y_mit = error_vec[ref_idx]
y_no_mit = error_vec_no_mit[ref_idx]

ax.annotate("",
            xy=(ref_snr, y_mit), xycoords="data",
            xytext=(ref_snr, y_no_mit), textcoords="data",
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.5))

gain_db = 10 * np.log10(y_no_mit / y_mit)
ax.text(ref_snr + 0.8, np.sqrt(y_mit * y_no_mit),
        f"{gain_db:.1f} dB\ngain",
        fontsize=9, color="#555555", va="center")

# ── Tick formatting ───────────────────────────────────────────────────────────
ax.tick_params(axis="both", which="major", labelsize=11, length=5)
ax.tick_params(axis="both", which="minor", length=3)
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.xaxis.set_minor_locator(plt.MultipleLocator(2))

# ── Style tweaks ──────────────────────────────────────────────────────────────
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

plt.tight_layout()
plt.savefig("ser_vs_snr.png", dpi=180, bbox_inches="tight")
plt.show()
