from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

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
    seed: int = 0


@dataclass(frozen=True)
class SweepConfig:
    clients_list: Tuple[int, ...]
    update_bits_list: Tuple[float, ...]
    base_cfg: LiveRunConfig


def _run_key_buffer(cfg: LiveRunConfig, total_rounds: int) -> Tuple[np.ndarray, np.ndarray]:
    buffer = np.zeros(total_rounds, dtype=float)
    outages = np.zeros(total_rounds, dtype=int)

    cycle = cfg.contact_window_s + cfg.contact_gap_s
    key_demand_bits = cfg.update_size_bits if cfg.mode == "otp" else cfg.session_key_bits

    for i in range(total_rounds):
        t = i * cfg.round_period_s
        in_contact = (t % cycle) < cfg.contact_window_s
        supply = cfg.qkd_throughput_bps * (cfg.round_period_s if in_contact else 0)

        prev = buffer[i - 1] if i > 0 else cfg.key_buffer_capacity_bits / 2
        available = min(cfg.key_buffer_capacity_bits, prev + supply)

        if available >= key_demand_bits:
            buffer[i] = available - key_demand_bits
        else:
            buffer[i] = available
            outages[i] = 1

    return buffer, outages


def _run_accuracy(cfg: LiveRunConfig, total_rounds: int, outages: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    accuracy = np.zeros(total_rounds, dtype=float)
    accuracy[0] = 0.55

    # Base learning rate scales with clients, but with diminishing returns.
    base_lr = 0.06 * np.log1p(cfg.clients_per_round)

    # Larger updates can help, but only if delivered reliably.
    update_gain = np.log1p(cfg.update_size_bits / 2e5) / 4.0
    update_gain = min(update_gain, 0.6)

    for i in range(1, total_rounds):
        if outages[i - 1] == 1:
            accuracy[i] = accuracy[i - 1]
            continue

        # Stochasticity for realism without heavy compute.
        noise = rng.normal(0.0, 0.002)
        lr = base_lr * (0.5 + update_gain) + noise
        lr = max(0.001, min(lr, 0.15))

        accuracy[i] = accuracy[i - 1] + lr * (1.0 - accuracy[i - 1])

    return np.clip(accuracy, 0.0, 1.0)


def simulate(cfg: LiveRunConfig) -> Dict[str, Any]:
    """Run a lightweight in-memory simulation with assumed dynamics."""

    total_seconds = int(cfg.sim_hours * 3600)
    total_rounds = max(2, total_seconds // cfg.round_period_s)

    rounds = np.arange(total_rounds)
    time_s = rounds * cfg.round_period_s

    buffer, outages = _run_key_buffer(cfg, total_rounds)
    accuracy = _run_accuracy(cfg, total_rounds, outages)

    time_to_target = None
    idx = np.where(accuracy >= cfg.target_accuracy)[0]
    if idx.size > 0:
        time_to_target = time_s[idx[0]]

    key_demand_bits = cfg.update_size_bits if cfg.mode == "otp" else cfg.session_key_bits

    return {
        "time_s": time_s,
        "rounds": rounds,
        "accuracy": accuracy,
        "buffer": buffer,
        "outages": outages,
        "time_to_target": time_to_target,
        "key_demand_bits": key_demand_bits,
    }


def run_sweep(cfg: SweepConfig, progress_cb=None) -> Dict[str, Any]:
    """Sweep clients_per_round and update_size_bits for a grid of outcomes."""

    clients = np.array(cfg.clients_list, dtype=int)
    updates = np.array(cfg.update_bits_list, dtype=float)

    time_to_target = np.zeros((clients.size, updates.size), dtype=float)
    outage_rate = np.zeros((clients.size, updates.size), dtype=float)
    final_accuracy = np.zeros((clients.size, updates.size), dtype=float)

    total = clients.size * updates.size
    done = 0
    for i, c in enumerate(clients):
        for j, u in enumerate(updates):
            run_cfg = LiveRunConfig(
                clients_per_round=int(c),
                round_period_s=cfg.base_cfg.round_period_s,
                update_size_bits=float(u),
                target_accuracy=cfg.base_cfg.target_accuracy,
                mode=cfg.base_cfg.mode,
                session_key_bits=cfg.base_cfg.session_key_bits,
                key_buffer_capacity_bits=cfg.base_cfg.key_buffer_capacity_bits,
                qkd_throughput_bps=cfg.base_cfg.qkd_throughput_bps,
                contact_window_s=cfg.base_cfg.contact_window_s,
                contact_gap_s=cfg.base_cfg.contact_gap_s,
                sim_hours=cfg.base_cfg.sim_hours,
                seed=cfg.base_cfg.seed,
            )

            res = simulate(run_cfg)
            ttt = res["time_to_target"]
            time_to_target[i, j] = float(ttt) if ttt is not None else np.nan
            outage_rate[i, j] = float(np.mean(res["outages"]))
            final_accuracy[i, j] = float(res["accuracy"][-1])
            done += 1
            if progress_cb is not None:
                progress_cb(done, total)

    return {
        "clients": clients,
        "updates": updates,
        "time_to_target": time_to_target,
        "outage_rate": outage_rate,
        "final_accuracy": final_accuracy,
    }
