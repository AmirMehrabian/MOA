import numpy as np

X_MIN, X_MAX = -93.97, +92.43
Y_MIN, Y_MAX = -60.33, +60.81
size_X = X_MAX - X_MIN  # = 186.4
size_Y = Y_MAX - Y_MIN  # = 121.1

CENTER_X = (X_MIN + X_MAX) / 2  # = -0.77
CENTER_Y = (Y_MIN + Y_MAX) / 2  # = +0.24
CELL_SIZE = 4

COLS = int(np.floor((X_MAX - X_MIN) / CELL_SIZE))  # = 93
ROWS = int(np.floor((Y_MAX - Y_MIN) / CELL_SIZE))

origin_x = CENTER_X - (X_MAX - X_MIN) / 2   # = X_MIN
origin_y = CENTER_Y - (Y_MAX - Y_MIN) / 2   # = Y_MIN

FRIENDLY_POS = [0.5, 0.0, 2.0]
JAMMER_POS = [-90.5, -60.0, 2.0]
RX_POS = [0.5, 20.0, 2.0]

RX_Velocity = [0.0, 5.0, 0.0]

rx_tn_d = int(np.linalg.norm(np.array(RX_POS) - np.array(FRIENDLY_POS)))
rx_jn_d = int(np.linalg.norm(np.array(RX_POS) - np.array(JAMMER_POS)))

START = (20, 1)

SOLVER_CFG = dict(
    max_depth=5,
    cell_size=[CELL_SIZE, CELL_SIZE],
    samples_per_tx=10 ** 6,
    size=[size_X, size_Y],  # (X_MAX-X_MIN), (Y_MAX-Y_MIN)
    orientation=[0, 0, 0],
    center=[CENTER_X, CENTER_Y, 0],  # CENTER_X, CENTER_Y
)

config_dict = {
    # Number of antennas, sensing nodes (SN), and jamming nodes (JN)
    "num_jn": 1,
    "num_sn": 4,
    # Signal-to-noise ratios
    "snr_tn": 10,  # in dB
    "snr_jn": 20,  # in dB
    "num_pilot_symbols": 20,  # 10,#10, #20,
    "num_data_symbols": 500,  # 250,#250,#500,
    "friendly_pos": FRIENDLY_POS,
    "jammer_pos": JAMMER_POS,
    "rx_pos": RX_POS,
    "bandwidth": 0.1e6,  # 20 MHz
    "noise_power": -105,  # noise power (linear scale)
    "rx_velocity": RX_Velocity,
    "rx-tx distance": rx_tn_d,
    "rx-jn distance": rx_jn_d,
}
