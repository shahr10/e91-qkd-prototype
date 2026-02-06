from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np


@dataclass
class DesignResults:
    selected_orbits: List[str]
    metrics: Dict[str, Any]
    vis_M: np.ndarray
    rate_M: np.ndarray
    orbit_names: List[str]
    roles: Dict[str, str]
    isl_partners: Dict[str, List[str]]
    downlink_map: Dict[str, List[str]]
    orbit_trajectories: Dict[str, np.ndarray]
    logbook: Optional[Any] = None
