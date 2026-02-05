from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


@dataclass(frozen=True)
class LiveRunConfig:
    clients_per_round: int
    round_period_s: int
    update_size_bits: float
    target_accuracy: float
    mode: str
    session_key_bits: int
    key_buffer_capacity_bits: float
    qkd_throughput_bps: float
    contact_window_s: int
    contact_gap_s: int
    sim_hours: float


def simulate(cfg: LiveRunConfig) -> Dict[str, Any]:
    """Run a lightweight in-memory simulation.

    This is a backend placeholder that keeps the Streamlit UI fast.
    Swap with the real backend when available.
    """

    total_seconds = int(cfg.sim_hours * 3600)
    total_rounds = max(1, total_seconds // cfg.round_period_s)

    rounds = np.arange(total_rounds)
    time_s = rounds * cfg.round_period_s

    # Simple accuracy curve (placeholder).
    growth_scale = max(1.0, total_rounds / 6.0)
    accuracy = 1.0 - np.exp(-rounds / growth_scale)
    accuracy = 0.5 + 0.5 * accuracy  # map to [0.5, 1.0)

    # Key buffer dynamics (placeholder model)
    key_demand_bits = cfg.update_size_bits if cfg.mode == "otp" else cfg.session_key_bits

    buffer = np.zeros_like(time_s, dtype=float)
    outages = np.zeros_like(time_s, dtype=int)

    cycle = cfg.contact_window_s + cfg.contact_gap_s
    for i in range(total_rounds):
        t = time_s[i]
        in_contact = (t % cycle) < cfg.contact_window_s
        supply = cfg.qkd_throughput_bps * (cfg.round_period_s if in_contact else 0)

        prev = buffer[i - 1] if i > 0 else cfg.key_buffer_capacity_bits / 2
        available = min(cfg.key_buffer_capacity_bits, prev + supply)

        if available >= key_demand_bits:
            buffer[i] = available - key_demand_bits
        else:
            buffer[i] = available
            outages[i] = 1

    time_to_target = None
    idx = np.where(accuracy >= cfg.target_accuracy)[0]
    if idx.size > 0:
        time_to_target = time_s[idx[0]]

    return {
        "time_s": time_s,
        "rounds": rounds,
        "accuracy": accuracy,
        "buffer": buffer,
        "outages": outages,
        "time_to_target": time_to_target,
        "key_demand_bits": key_demand_bits,
    }
