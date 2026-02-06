from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from cislunar_constellation.config import NetworkConfig
from constellation_notebook import VALIDATED_ORBITS, GROUND_STATIONS


def assign_satellite_roles(
    selected_orbits: List[str],
    orbit_trajectories: Dict[str, np.ndarray],
    vis_M: np.ndarray,
    rate_M: np.ndarray,
    orbit_names: List[str],
    cfg: NetworkConfig,
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
    satellite_roles: Dict[str, str] = {}
    satellite_isl_partners: Dict[str, List[str]] = {}
    satellite_downlink_gs: Dict[str, List[str]] = {}

    for sat_name in selected_orbits:
        if sat_name not in orbit_names:
            continue
        j = orbit_names.index(sat_name)

        gs_vis_count = int(np.sum(vis_M[j, :, :] > 0))
        total_rate = float(np.sum(rate_M[j, :, :]))

        if gs_vis_count > 10 and total_rate > 1e5:
            satellite_roles[sat_name] = "DLS"
            reachable = []
            for k, gs in enumerate(GROUND_STATIONS):
                if np.any(vis_M[j, :, k] > 0):
                    reachable.append(gs["name"])
            satellite_downlink_gs[sat_name] = reachable
        elif total_rate > 5e5:
            satellite_roles[sat_name] = "EPS"
            satellite_downlink_gs[sat_name] = []
        else:
            satellite_roles[sat_name] = "RS"
            satellite_downlink_gs[sat_name] = []

    if cfg.allow_isl:
        for sat1 in selected_orbits:
            if sat1 not in orbit_trajectories or orbit_trajectories[sat1] is None:
                continue

            pos1 = orbit_trajectories[sat1]
            isl_partners = []

            for sat2 in selected_orbits:
                if sat2 == sat1:
                    continue
                if sat2 not in orbit_trajectories or orbit_trajectories[sat2] is None:
                    continue

                pos2 = orbit_trajectories[sat2]
                n_min = min(pos1.shape[1], pos2.shape[1])

                dist_min = np.inf
                for idx in range(n_min):
                    d = np.linalg.norm(pos1[:, idx] - pos2[:, idx])
                    if d < dist_min:
                        dist_min = d

                if dist_min < cfg.isl_distance_km * 1e3:
                    isl_partners.append(sat2)

            satellite_isl_partners[sat1] = isl_partners[: cfg.max_isl_partners]
    else:
        for sat in selected_orbits:
            satellite_isl_partners[sat] = []

    return satellite_roles, satellite_isl_partners, satellite_downlink_gs
