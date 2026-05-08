import numpy as np
from scipy.special import j0
from scipy.linalg import svd, null_space
from scipy.stats import nakagami


def nakagami_channel(config_dict, a, b):
    m = config_dict['nakagami_shape_param']
    abs_value = nakagami.rvs(m, scale=np.sqrt(1.0), size=(a, b))
    phase_value = np.random.uniform(0, 2 * np.pi, size=(a, b))
    channel_matrix = abs_value * np.exp(1j * phase_value)
    return channel_matrix


def db2pow(db):
    return 10 ** (db / 10)


def pow2db(power):
    return 10 * np.log10(power)


def pskmod(symbols, mod_order):
    return np.exp(1j * 2 * np.pi * symbols / mod_order)


def pskdemod(signal, mod_order):
    phase = np.angle(signal) % (2 * np.pi)
    symbols = np.round(phase / (2 * np.pi / mod_order)) % mod_order
    return symbols.astype(int)


def env_response(config_dict):
    nb = config_dict["num_pilot_block"]
    num_sn = config_dict["num_sn"]
    num_jn = config_dict["num_jn"]
    snr_tn = config_dict["snr_tn"]
    snr_jn = config_dict["snr_jn"]

    num_pilot = config_dict["num_pilot_symbols"]
    num_data = config_dict["num_data_symbols"]
    num_antennas = config_dict["num_antennas"]

    noise_variance = -105 #dBm

    mod_order = 4


    i_complex = 1j  # imaginary unit



    amp_jn_signal = np.sqrt(db2pow(snr_jn - 30))
    amp_tn_signal = np.sqrt(db2pow(snr_tn -30))

    # Noise matrices (complex Gaussian)
    noise_matrix_p1 = np.sqrt(noise_variance / 2) * (
            np.random.randn(num_sn, num_pilot) + i_complex * np.random.randn(num_sn, num_pilot))
    noise_matrix_d = np.sqrt(noise_variance / 2) * (
            np.random.randn(num_sn, num_data) + i_complex * np.random.randn(num_sn, num_data))
    # noise_matrix_p2 = np.sqrt(noise_variance / 2) * (
    #         np.random.randn(num_sn, num_pilot) + i_complex * np.random.randn(num_sn, num_pilot))

    # Generate random symbols (as integers) and modulate them using PSK
    pilot_symbols = np.random.randint(0, mod_order, size=num_pilot)
    data_symbols = np.random.randint(0, mod_order, size=num_data)
    mod_pilot_symbols = pskmod(pilot_symbols, mod_order).reshape(1, num_pilot)
    mod_data_symbols = pskmod(data_symbols, mod_order).reshape(1, num_data)

    h, g = get_sionna_channels(scene, solver, rx, tx_f_idx, tx_j_idx,
                               rx_world_pos, config_dict)
    tn_chan_vec = h.reshape(-1, 1)
    # Channel realizations (assumed provided by nakagami_channel function)
    #tn_chan_vec = nakagami_channel(config_dict, num_sn, 1)

    jamming_symbols_p1 = pskmod(np.random.randint(0, mod_order, size=num_pilot), mod_order).reshape(1, num_pilot)
    jamming_symbols_d = pskmod(np.random.randint(0, mod_order, size=num_data), mod_order).reshape(1, num_data)
    # jamming_symbols_p2 = pskmod(np.random.randint(0, mod_order, size=num_pilot), mod_order).reshape(1, num_pilot)

    jn_chan_vec_p1 = nakagami_channel(config_dict, num_sn, 1)

    solver, rx, tx_f_idx, tx_j_idx = init_channel_solver(scene, config_dict)

    # Inside loop at each node position

    jn_chan_mat_p1 = g[:, :num_pilot]
    jn_chan_mat_d = g[:, num_pilot:]

    # Received signals
    sn_rx_mat_p1 = (amp_tn_signal * tn_chan_vec @ mod_pilot_symbols) + (
            amp_jn_signal * jn_chan_mat_p1 @ np.diag(jamming_symbols_p1.reshape(-1))) + noise_matrix_p1  # Pilot 1
    sn_rx_mat_d = (amp_tn_signal * tn_chan_vec @ mod_data_symbols) + (
            amp_jn_signal * jn_chan_mat_d @ np.diag(jamming_symbols_d.reshape(-1))) + noise_matrix_d  # Data
    # sn_rx_mat_p2 = (amp_tn_signal * tn_chan_vec @ mod_pilot_symbols) + (
    #         amp_jn_signal * jn_chan_mat_p2 @ np.diag(jamming_symbols_p2.reshape(-1))) + noise_matrix_p2  # Next
    # pilot

    # Control Channel
    # fc_chan_mat = nakagami_channel(config_dict, num_antennas, num_sn)
    # fc_noise_mat_p1 = np.sqrt(noise_variance / 2) * (
    #         np.random.randn(num_antennas, num_pilot) + i_complex * np.random.randn(num_antennas, num_pilot))
    # fc_noise_mat_d = np.sqrt(noise_variance / 2) * (
    #         np.random.randn(num_antennas, data_block_length) + i_complex * np.random.randn(num_antennas,
    #                                                                                        data_block_length))
    # fc_noise_mat_p2 = np.sqrt(noise_variance / 2) * (
    #         np.random.randn(num_antennas, num_pilot) + i_complex * np.random.randn(num_antennas, num_pilot))

    fc_rx_mat_p1 = sn_rx_mat_p1
    fc_rx_mat_d =  sn_rx_mat_d
    # Least squares equalization of channel matrix

    est_sn_rx_mat_p1 =  fc_rx_mat_p1
    est_sn_rx_mat_d =  fc_rx_mat_d

    # Compute null space for mod_pilot_symbols (treat mod_pilot_symbols as a 1 x num_pilot row vector)
    pilots_perp_mat = null_space(mod_pilot_symbols)

    # Estimated jammer signal (projecting est_sn_rx_mat_p1 onto the null space)
    est_jn_signal_p1 = est_sn_rx_mat_p1 @ pilots_perp_mat

    # Calculate jammer power (using Frobenius norm)
    #power_jam = np.linalg.norm(est_jn_signal_p1, 'fro') ** 2 / (np.prod(est_jn_signal_p1.shape))
    #power_jam_db = pow2db(power_jam)

    # SVD to extract dominant components of estimated jammer channel
    u_mat_p1, _, _ = svd(est_jn_signal_p1)
    est_jn_chan_vec_p1 = u_mat_p1[:, 0]


  # Compute null space of the row vector of est_jn_chan_vec_p1

    chan_perp_mat = np.conj(u_mat_p1[:, num_jn:].T)

    rx_mat_canceled_jam = chan_perp_mat @ est_sn_rx_mat_p1
    #power_signal_db = pow2db(np.linalg.norm(rx_mat_canceled_jam, 'fro') ** 2 / (np.prod(rx_mat_canceled_jam.shape)))

    est_tn_chan_vec = (rx_mat_canceled_jam @ np.conj(mod_pilot_symbols.T) / (
            np.linalg.norm(mod_pilot_symbols) ** 2)) / amp_tn_signal

    est_data_symbol_vec = (est_tn_chan_vec.conj().T / (np.linalg.norm(est_tn_chan_vec) ** 2)) @ (
            chan_perp_mat @ est_sn_rx_mat_d)

    demod_est_data_symbol_vec = pskdemod(est_data_symbol_vec, mod_order)

    # Count correctly received symbols:
    errors = np.sum(demod_est_data_symbol_vec != data_symbols)
    num_correct_sym = num_data - errors


    return errors, num_correct_sym
#
#
# def env_response_with_mitigation(config_dict, mitigation=False):
#
#     nb = config_dict["num_pilot_block"]
#     num_sn = config_dict["num_sn"]
#     num_jn = config_dict["num_jn"]
#     snr_tn = config_dict["snr_tn"]
#     snr_jn = config_dict["snr_jn"]
#     num_coherence = config_dict["num_coherence_symbols"]
#     num_pilot = config_dict["num_pilot_symbols"]
#     num_data = config_dict["num_data_symbols"]
#     num_antennas = config_dict["num_antennas"]
#
#     noise_variance = 1
#
#     mod_order = 4
#
#     new_num_data = num_data - (nb - 1) * num_pilot
#
#     data_block_length = int(new_num_data / nb)
#     new_num_data_main = new_num_data
#     data_block_length_main = data_block_length
#     total_reward = 0
#     i_complex = 1j  # imaginary unit
#
#     # tau
#     tau = data_block_length / num_coherence
#     ts_tc_ratio = tau / data_block_length  # ratio of one symbol time to coherence time of channel
#
#     vec_idx = np.arange(1, num_data + 2 * num_pilot + 1)
#     vec_rho_g = j0(2 * np.pi * vec_idx * 0.423 * ts_tc_ratio)
#     rho1_matrix = np.diag(vec_rho_g)
#     rho1_matrix_complement = np.diag(np.sqrt(1 - vec_rho_g ** 2))
#
#     if nb == 1:
#         nb_vec = np.arange(1, nb + 1)  # [1,...,nb]
#     else:
#         nb_vec = np.arange(0, nb + 1)  # [0,...,nb]
#
#     correlation_vector = np.zeros(len(nb_vec), dtype=complex)
#     cc = 0
#
#     for sec in nb_vec:
#         cc += 1
#         if sec == 0:
#             new_num_data = num_data
#             data_block_length = new_num_data  # When sec==0, use the full num_data
#
#         amp_jn_signal = np.sqrt(db2pow(snr_jn) * noise_variance)
#         amp_tn_signal = np.sqrt(db2pow(snr_tn) * noise_variance)
#
#         # Noise matrices (complex Gaussian)
#         noise_matrix_p1 = np.sqrt(noise_variance / 2) * (
#                 np.random.randn(num_sn, num_pilot) + i_complex * np.random.randn(num_sn, num_pilot))
#         noise_matrix_d = np.sqrt(noise_variance / 2) * (
#                 np.random.randn(num_sn, data_block_length) + i_complex * np.random.randn(num_sn, data_block_length))
#         noise_matrix_p2 = np.sqrt(noise_variance / 2) * (
#                 np.random.randn(num_sn, num_pilot) + i_complex * np.random.randn(num_sn, num_pilot))
#
#         # Generate random symbols (as integers) and modulate them using PSK
#         pilot_symbols = np.random.randint(0, mod_order, size=num_pilot)
#         data_symbols = np.random.randint(0, mod_order, size=data_block_length)
#         mod_pilot_symbols = pskmod(pilot_symbols, mod_order).reshape(1, num_pilot)
#         mod_data_symbols = pskmod(data_symbols, mod_order).reshape(1, data_block_length)
#
#         # Channel realizations (assumed provided by nakagami_channel function)
#         tn_chan_vec = nakagami_channel(config_dict, num_sn, 1)
#
#         jamming_symbols_p1 = pskmod(np.random.randint(0, mod_order, size=num_pilot), mod_order).reshape(1, num_pilot)
#         jamming_symbols_d = pskmod(np.random.randint(0, mod_order, size=data_block_length), mod_order).reshape(1,
#                                                                                                                data_block_length)
#         jamming_symbols_p2 = pskmod(np.random.randint(0, mod_order, size=num_pilot), mod_order).reshape(1, num_pilot)
#
#         jn_chan_vec_p1 = nakagami_channel(config_dict, num_sn, 1)
#
#         dependent_jn_chan_mat = np.tile(jn_chan_vec_p1, (1, 2 * num_pilot + data_block_length))
#
#         independent_chan_mat = nakagami_channel(config_dict, num_sn, 2 * num_pilot + data_block_length)
#
#         correlated_jn_chan_mat = dependent_jn_chan_mat @ rho1_matrix[:2 * num_pilot + data_block_length,
#                                                          :2 * num_pilot + data_block_length] + independent_chan_mat @ rho1_matrix_complement[
#                                                                                                                       :2 * num_pilot + data_block_length,
#                                                                                                                       :2 * num_pilot + data_block_length]
#         jn_chan_mat_p1 = correlated_jn_chan_mat[:, :num_pilot]
#         jn_chan_mat_d = correlated_jn_chan_mat[:, num_pilot:num_pilot + data_block_length]
#         jn_chan_mat_p2 = correlated_jn_chan_mat[:, num_pilot + data_block_length:2 * num_pilot + data_block_length]
#
#         # Received signals
#         sn_rx_mat_p1 = (amp_tn_signal * tn_chan_vec @ mod_pilot_symbols) + (
#                 amp_jn_signal * jn_chan_mat_p1 @ np.diag(jamming_symbols_p1.reshape(-1))) + noise_matrix_p1  # Pilot 1
#         sn_rx_mat_d = (amp_tn_signal * tn_chan_vec @ mod_data_symbols) + (
#                 amp_jn_signal * jn_chan_mat_d @ np.diag(jamming_symbols_d.reshape(-1))) + noise_matrix_d  # Data
#         sn_rx_mat_p2 = (amp_tn_signal * tn_chan_vec @ mod_pilot_symbols) + (
#                 amp_jn_signal * jn_chan_mat_p2 @ np.diag(jamming_symbols_p2.reshape(-1))) + noise_matrix_p2  # Next
#         # pilot
#
#         # Control Channel
#         fc_chan_mat = nakagami_channel(config_dict, num_antennas, num_sn)
#         fc_noise_mat_p1 = np.sqrt(noise_variance / 2) * (
#                 np.random.randn(num_antennas, num_pilot) + i_complex * np.random.randn(num_antennas, num_pilot))
#         fc_noise_mat_d = np.sqrt(noise_variance / 2) * (
#                 np.random.randn(num_antennas, data_block_length) + i_complex * np.random.randn(num_antennas,
#                                                                                                data_block_length))
#         fc_noise_mat_p2 = np.sqrt(noise_variance / 2) * (
#                 np.random.randn(num_antennas, num_pilot) + i_complex * np.random.randn(num_antennas, num_pilot))
#         fc_rx_mat_p1 = fc_chan_mat @ sn_rx_mat_p1 + fc_noise_mat_p1
#         fc_rx_mat_d = fc_chan_mat @ sn_rx_mat_d + fc_noise_mat_d
#         fc_rx_mat_p2 = fc_chan_mat @ sn_rx_mat_p2 + fc_noise_mat_p2
#
#         # Least squares equalization of channel matrix
#         fc_chan_equalizer_mat = np.linalg.pinv(fc_chan_mat)
#         est_sn_rx_mat_p1 = fc_chan_equalizer_mat @ fc_rx_mat_p1
#         est_sn_rx_mat_d = fc_chan_equalizer_mat @ fc_rx_mat_d
#         est_sn_rx_mat_p2 = fc_chan_equalizer_mat @ fc_rx_mat_p2
#
#         # Compute null space for mod_pilot_symbols (treat mod_pilot_symbols as a 1 x num_pilot row vector)
#         pilots_perp_mat = null_space(mod_pilot_symbols)
#
#         # Estimated jammer signal (projecting est_sn_rx_mat_p1 onto the null space)
#         est_jn_signal_p1 = est_sn_rx_mat_p1 @ pilots_perp_mat
#         est_jn_signal_p2 = est_sn_rx_mat_p2 @ pilots_perp_mat
#
#         # Calculate jammer power (using Frobenius norm)
#         power_jam = np.linalg.norm(est_jn_signal_p1, 'fro') ** 2 / (np.prod(est_jn_signal_p1.shape))
#         power_jam_db = pow2db(power_jam)
#
#         # SVD to extract dominant components of estimated jammer channel
#         u_mat_p1, _, _ = svd(est_jn_signal_p1)
#         est_jn_chan_vec_p1 = u_mat_p1[:, 0]
#
#         u_mat_p2, _, _ = svd(est_jn_signal_p2)
#         est_jn_chan_p2 = u_mat_p2[:, 0]
#
#         if sec == 0:
#             correlation_vector[cc - 1] = np.vdot(est_jn_chan_vec_p1, est_jn_chan_p2)  # Adjusted index to be 0-based
#             new_num_data = new_num_data_main
#             data_block_length = data_block_length_main
#             continue
#         correlation_vector[cc - 1] = np.vdot(est_jn_chan_vec_p1, est_jn_chan_p2)  # Adjusted index to be 0-based
#
#         # Compute null space of the row vector of est_jn_chan_vec_p1
#
#         chan_perp_mat = np.conj(u_mat_p1[:, num_jn:].T)
#         if not mitigation:
#             chan_perp_mat = np.eye(num_sn)
#
#         rx_mat_canceled_jam = chan_perp_mat @ est_sn_rx_mat_p1
#         power_signal_db = pow2db(np.linalg.norm(rx_mat_canceled_jam, 'fro') ** 2 / (np.prod(rx_mat_canceled_jam.shape)))
#
#         est_tn_chan_vec = (rx_mat_canceled_jam @ np.conj(mod_pilot_symbols.T) / (
#                 np.linalg.norm(mod_pilot_symbols) ** 2)) / amp_tn_signal
#
#         est_data_symbol_vec = (est_tn_chan_vec.conj().T / (np.linalg.norm(est_tn_chan_vec) ** 2)) @ (
#                 chan_perp_mat @ est_sn_rx_mat_d)
#
#         demod_est_data_symbol_vec = pskdemod(est_data_symbol_vec, mod_order)
#
#         # Count correctly received symbols:
#         errors = np.sum(demod_est_data_symbol_vec != data_symbols)
#         block_reward = data_block_length - errors
#         total_reward += block_reward
#
#     return total_reward, correlation_vector, power_jam_db, power_signal_db
