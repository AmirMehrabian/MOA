"""
sionna_channels.py
==================
Replaces Nakagami draws with Sionna RT geometry-based channels.

Verified PathSolver API (Sionna 2.0.1):
    solver(scene, max_depth, samples_per_src, synthetic_array, los,
           specular_reflection, diffuse_reflection, refraction, ...)
    → NO active_transmitters argument

Both TXs are in the scene simultaneously.
One solver call returns taps for ALL TXs at once:
    shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_taps]

TX ordering matches insertion order in scene.transmitters dict:
    tx_friendly added first → index 0
    tx_jammer   added second → index 1

At t=0: Doppler phase = exp(j·2π·f_Δ·0) = 1  →  h at t=0 is the true static channel
regardless of rx velocity. So one solver call gives both h (t=0, tx=0) and
g (all t, tx=1) efficiently.

Setup — call ONCE before loop:
    solver, rx = init_channel_solver(scene, config_dict)

Per-iteration — call inside loop:
    h, g = get_sionna_channels(scene, solver, rx, rx_position, config_dict)

Returns
-------
h : (num_sn,)                    complex ndarray — friendly channel (static)
                                  use as: tn_chan_vec = h.reshape(-1, 1)
g : (num_sn, num_pilot+num_data) complex ndarray — jammer channel (time-varying)
                                  use as: jn_chan_mat_p1 = g[:, :num_pilot]
                                          jn_chan_mat_d  = g[:, num_pilot:]
"""

import numpy as np
import sionna
from sionna.rt import PathSolver, Transmitter, Receiver, PlanarArray, load_scene


# ─────────────────────────────────────────────────────────────────────────────
#  One-time setup  —  call ONCE before the loop
# ─────────────────────────────────────────────────────────────────────────────

def init_channel_solver(scene, config_dict: dict):
    """
    Configure arrays, add both TXs and RX to scene, create PathSolver.

    config_dict keys:
        "num_sn"          – number of receive antennas
        "rx_height"       – receiver height [m], default 1.5
        "friendly_pos"    – [x, y, z] of friendly TX
        "jammer_pos"      – [x, y, z] of jammer TX

    Returns
    -------
    solver : PathSolver  (reused every call)
    rx     : Receiver    (position updated every call, never removed)
    tx_friendly_idx : int  (index in taps tensor for friendly TX)
    tx_jammer_idx   : int  (index in taps tensor for jammer TX)
    """
    num_sn       = config_dict["num_sn"]
    rx_height    = config_dict.get("rx_height", 1.5)
    friendly_pos = config_dict["friendly_pos"]
    jammer_pos   = config_dict["jammer_pos"]

    # TX: single isotropic element
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )

    # RX: 1 × num_sn isotropic array  (num_sn antennas = num_sn in env_functions)
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=num_sn,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )

    # Add both TXs — insertion order determines TX index in taps tensor
    tx_friendly = Transmitter(name="tx_friendly", position=friendly_pos)
    tx_jammer   = Transmitter(name="tx_jammer",   position=jammer_pos)
    scene.add(tx_friendly)   # → index 0
    scene.add(tx_jammer)     # → index 1

    # Confirm ordering
    tx_names = list(scene.transmitters.keys())
    tx_friendly_idx = tx_names.index("tx_friendly")
    tx_jammer_idx   = tx_names.index("tx_jammer")
    print(f"[init]  TX ordering: {tx_names}")
    print(f"[init]  tx_friendly_idx={tx_friendly_idx}  tx_jammer_idx={tx_jammer_idx}")

    # Add RX once — only position/velocity updated per call
    rx = Receiver(name="rx_node", position=[0.0, 0.0, rx_height])
    scene.add(rx)

    solver = PathSolver()

    return solver, rx, tx_friendly_idx, tx_jammer_idx


# ─────────────────────────────────────────────────────────────────────────────
#  Per-call channel computation  —  call inside the loop
# ─────────────────────────────────────────────────────────────────────────────

def get_sionna_channels(scene,
                        solver,
                        rx,
                        tx_friendly_idx: int,
                        tx_jammer_idx: int,
                        rx_position: list,
                        config_dict: dict) -> tuple:
    """
    One solver call → taps for both TXs → index to separate h and g.

    taps shape: [num_rx=1, num_rx_ant=num_sn, num_tx=2, num_tx_ant=1, T, 1]

    h = taps[0, :, tx_friendly_idx, 0, 0, 0]   shape (num_sn,)  — t=0 is static
    g = taps[0, :, tx_jammer_idx,   0, :, 0]   shape (num_sn, T) — time-varying

    config_dict keys:
        "num_sn"            – receive antennas
        "num_pilot_symbols" – pilot block length
        "num_data_symbols"  – data block length
        "bandwidth"         – system bandwidth [Hz]  (small → flat fading)
        "symbol_period"     – Ts [s]
        "rx_velocity"       – [vx, vy, vz] m/s, default [0, 10, 0]
        "rx_height"         – z coordinate [m], default 1.5
    """
    num_sn    = config_dict["num_sn"]
    num_pilot = config_dict["num_pilot_symbols"]
    num_data  = config_dict["num_data_symbols"]
    bandwidth = config_dict["bandwidth"]
    Ts        = config_dict["symbol_period"]
    rx_vel    = config_dict.get("rx_velocity", [0.0, 10.0, 0.0])
    rx_height = config_dict.get("rx_height", 1.5)

    T = num_pilot + num_data   # total time steps needed for g

    # ── Update RX state (only thing that changes per call) ────────────────────
    rx.position = [rx_position[0], rx_position[1], rx_height]
    rx.velocity  = rx_vel       # Doppler on all paths from both TXs
                                # at t=0: exp(j·2π·f_Δ·0)=1 → h unaffected

    # ── Single solver call — both TXs computed simultaneously ─────────────────
    paths = solver(
        scene              = scene,
        max_depth          = 5,
        samples_per_src    = 10 ** 5,
        synthetic_array    = True,
    )

    # ── Extract taps: shape [1, num_sn, 2, 1, T, 1] ──────────────────────────
    taps = paths.taps(
        bandwidth          = bandwidth,
        l_min              = 0,
        l_max              = 0,          # flat fading → 1 tap
        sampling_frequency = 1.0 / Ts,  # one sample per symbol
        num_time_steps     = T,
        out_type           = "numpy",
    )
    # taps shape: [num_rx=1, num_rx_ant=num_sn, num_tx=2, num_tx_ant=1, T, num_taps=1]

    # ── h: friendly channel at t=0 (static) ──────────────────────────────────
    # index: [rx=0, :, tx_friendly, tx_ant=0, t=0, tap=0] → (num_sn,)
    h = taps[0, :, tx_friendly_idx, 0, 0, 0]   # (num_sn,)

    # ── g: jammer channel all time steps (time-varying via Doppler) ──────────
    # index: [rx=0, :, tx_jammer, tx_ant=0, :, tap=0] → (num_sn, T)
    g = taps[0, :, tx_jammer_idx, 0, :, 0]     # (num_sn, T)

    return h, g