"""
RadioMap.py
===========
Builds two path-gain cost grids for the MOA* solver using Sionna RT.

    maps = build_radio_maps()

    maps["grid_jam_pwr"]      – jammer cost,    shape (H, W), range [0, 1]
    maps["grid_fav_deficit"]  – friendly deficit,shape (H, W), range [0, 1]
    maps["rows"], maps["cols"]– use these as ROWS, COLS in main.py
"""

import numpy as np
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, RadioMapSolver
from config import SOLVER_CFG, FRIENDLY_POS, JAMMER_POS
import numpy as np
# ── Helpers ───────────────────────────────────────────────────────────────────

def _solve(scene, name, position):
    """Add TX, solve radio map, remove TX. Returns (H, W) linear path-gain."""
    scene.add(Transmitter(name=name, position=position, display_radius=2))
    rm = RadioMapSolver()(scene=scene, **SOLVER_CFG)
    pg = rm.path_gain.numpy().squeeze()
    scene.remove(name)
    return pg


def _to_db(linear):
    return 10.0 * np.log10(np.maximum(linear, 1e-14))


def _normalise(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


# ── Public API ────────────────────────────────────────────────────────────────

def build_radio_maps():
    """
    Returns a dict with ready-to-use MOA* cost grids and the grid dimensions.
    No resizing — MOA* grid size is derived directly from the radio map shape.
    """
    scene = load_scene(sionna.rt.scene.simple_street_canyon)
    scene.tx_array = scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )

    friendly_db = _to_db(_solve(scene, "tx_friendly", FRIENDLY_POS))
    jammer_db = _to_db(_solve(scene, "tx_jammer", JAMMER_POS))

    rows, cols = friendly_db.shape

    return {
        # MOA* cost grids  (no resize — grid is sized from the radio map)
        "grid_jam_pwr": _normalise(jammer_db),  # high = strongly jammed
        "grid_fav_deficit": 1.0 - _normalise(friendly_db),  # high = poorly covered
        # Grid dimensions for main.py
        "rows": rows,
        "cols": cols,
        "scene": scene,  # for visualization in main.py (optional)
    }
