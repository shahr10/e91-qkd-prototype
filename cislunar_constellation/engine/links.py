from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from cislunar_constellation.config import GroundConfig, LinkConfig

# Temporary bridge: reuse notebook code for geometry helpers
from constellation_notebook import (
    GROUND_STATIONS,
    get_gs_position_corrected,
    compute_elevation,
    check_occultation,
)


def compute_link_budget(distance_m: float, elevation_deg: float, link: LinkConfig) -> Tuple[float, float]:
    if distance_m <= 0 or elevation_deg <= 5:
        return 0.0, 0.5

    G_tx = (np.pi * link.tx_aperture_m / link.wavelength_m) ** 2 * link.tx_eff
    G_rx = (np.pi * link.rx_aperture_m / link.wavelength_m) ** 2 * link.rx_eff
    fspl = (link.wavelength_m / (4 * np.pi * distance_m)) ** 2

    airmass = 1 / np.sin(np.radians(max(elevation_deg, 5)))
    eta_atm = 10 ** (-link.atm_loss_zenith_db * airmass / 10)

    theta_beam = link.wavelength_m / (np.pi * link.tx_aperture_m)
    eta_point = np.exp(-(link.pointing_jitter_rad / theta_beam) ** 2)

    eta = float(np.clip(G_tx * G_rx * fspl * eta_atm * eta_point, 0.0, 1.0))

    n_signal = link.mu_sig * eta * link.detector_eff
    n_noise = link.dark_rate_cps * 1e-9 + 1e-7
    e_opt = 0.01

    if n_signal > 1e-15:
        qber = e_opt + 0.5 * n_noise / (n_signal + n_noise)
    else:
        qber = 0.5

    return eta, float(np.clip(qber, e_opt, 0.5))


def bb84_key_rate(eta: float, qber: float, link: LinkConfig) -> float:
    if eta <= 0 or qber >= 0.11:
        return 0.0

    Q = 1 - np.exp(-link.mu_sig * eta * link.detector_eff)
    Q1 = link.mu_sig * np.exp(-link.mu_sig) * eta * link.detector_eff

    def _binary_entropy(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    r = Q1 * (1 - _binary_entropy(qber)) - Q * 1.16 * _binary_entropy(qber)
    return max(r, 0.0) * link.rep_rate_hz


def build_visibility_matrix(
    orbit_trajectories: Dict[str, np.ndarray],
    ground: GroundConfig,
    link: LinkConfig,
    n_time: int = 36,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    orbit_names = list(orbit_trajectories.keys())
    num_j = len(orbit_names)

    gs_list = GROUND_STATIONS
    num_gs = len(gs_list)

    vis_M = np.zeros((num_j, n_time, num_gs), dtype=int)
    rate_M = np.zeros((num_j, n_time, num_gs))

    for j, name in enumerate(orbit_names):
        positions = orbit_trajectories[name]
        if positions is None:
            continue

        n_pts = positions.shape[1]

        for t in range(n_time):
            t_hours = t * 24.0 / n_time
            pos_idx = int(t * n_pts / n_time) % n_pts
            sat_pos = positions[:, pos_idx]

            for k, gs in enumerate(gs_list):
                gs_pos = get_gs_position_corrected(gs, t_hours)

                if check_occultation(sat_pos, gs_pos):
                    continue

                elev = compute_elevation(sat_pos, gs_pos)
                if elev < ground.elevation_mask_deg:
                    continue

                dist = np.linalg.norm(sat_pos - gs_pos)
                eta, qber = compute_link_budget(dist, elev, link)
                rate = bb84_key_rate(eta, qber, link)

                if rate > 10.0:
                    vis_M[j, t, k] = 1
                    rate_M[j, t, k] = rate

    return vis_M, rate_M, orbit_names
