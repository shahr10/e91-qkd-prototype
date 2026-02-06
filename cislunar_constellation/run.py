from __future__ import annotations

from typing import Dict

import numpy as np

from cislunar_constellation.config import ConstellationConfig
from cislunar_constellation.models import DesignResults
from cislunar_constellation.engine.orbits import VALIDATED_ORBITS, propagate_orbit, list_orbits_by_family
from cislunar_constellation.engine.links import build_visibility_matrix
from cislunar_constellation.engine.optimize import genetic_algorithm_optimization
from cislunar_constellation.engine.network import assign_satellite_roles


def _select_orbits(cfg: ConstellationConfig) -> Dict[str, dict]:
    if cfg.orbits.orbit_names:
        return {k: VALIDATED_ORBITS[k] for k in cfg.orbits.orbit_names if k in VALIDATED_ORBITS}

    candidates = list_orbits_by_family(cfg.orbits.families)
    orbit_names = list(candidates.keys())

    if cfg.orbits.max_candidates and len(orbit_names) > cfg.orbits.max_candidates:
        orbit_names = orbit_names[: cfg.orbits.max_candidates]

    return {k: candidates[k] for k in orbit_names}


def run_design(cfg: ConstellationConfig) -> DesignResults:
    selected_catalog = _select_orbits(cfg)

    orbit_trajectories: Dict[str, np.ndarray] = {}
    for name in selected_catalog.keys():
        pos = propagate_orbit(name)
        if pos is not None:
            orbit_trajectories[name] = pos

    vis_M, rate_M, orbit_names = build_visibility_matrix(
        orbit_trajectories,
        cfg.ground,
        cfg.link,
        n_time=cfg.time.n_time,
        horizon_hours=cfg.time.horizon_hours,
    )

    selected, metrics = genetic_algorithm_optimization(
        vis_M,
        rate_M,
        orbit_names,
        cfg.optimize,
    )

    roles, isl_partners, downlink_map = assign_satellite_roles(
        selected,
        orbit_trajectories,
        vis_M,
        rate_M,
        orbit_names,
        cfg.network,
        cfg.ground,
    )

    return DesignResults(
        selected_orbits=selected,
        metrics=metrics,
        vis_M=vis_M,
        rate_M=rate_M,
        orbit_names=orbit_names,
        roles=roles,
        isl_partners=isl_partners,
        downlink_map=downlink_map,
        orbit_trajectories=orbit_trajectories,
        logbook=metrics.get("logbook"),
    )


if __name__ == "__main__":
    cfg = ConstellationConfig()
    results = run_design(cfg)
    print("Selected orbits:", results.selected_orbits)
    print("Metrics:", results.metrics)
