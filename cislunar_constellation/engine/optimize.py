from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np

from deap import base, creator, tools, algorithms

from cislunar_constellation.config import OptimizeConfig
from constellation_notebook import VALIDATED_ORBITS


def setup_genetic_algorithm(vis_M, rate_M, orbit_names):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        len(orbit_names),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_constellation(individual, p_target=12):
        selected_count = np.sum(individual)
        size_penalty = abs(selected_count - p_target) / max(p_target, 1)

        if selected_count == 0:
            return (-1000.0, 0.0, 0.0)

        total_rate = 0.0
        coverage_count = 0
        selected_families = set()

        num_t = vis_M.shape[1]
        num_gs = vis_M.shape[2]

        for t in range(num_t):
            for k in range(num_gs):
                best_rate_tk = 0.0
                for j in np.where(individual)[0]:
                    if vis_M[j, t, k]:
                        best_rate_tk = max(best_rate_tk, rate_M[j, t, k])
                        selected_families.add(
                            VALIDATED_ORBITS[orbit_names[j]]["family"]
                        )
                if best_rate_tk > 0.0:
                    total_rate += best_rate_tk
                    coverage_count += 1

        coverage = coverage_count / (num_t * num_gs)
        diversity = len(selected_families)

        fitness_rate = total_rate / 1e6
        fitness_cov = coverage
        denom = len(set(VALIDATED_ORBITS[o]["family"] for o in orbit_names))
        fitness_div = diversity / max(denom, 1)

        penalty = 1.0 - 0.5 * size_penalty

        return (
            fitness_rate * penalty,
            fitness_cov * penalty,
            fitness_div * penalty,
        )

    toolbox.register("evaluate", evaluate_constellation)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    return toolbox


def genetic_algorithm_optimization(
    vis_M,
    rate_M,
    orbit_names,
    cfg: OptimizeConfig,
) -> Tuple[List[str], Dict[str, Any]]:
    toolbox = setup_genetic_algorithm(vis_M, rate_M, orbit_names)

    pop = toolbox.population(n=cfg.pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=cfg.cxpb,
        mutpb=cfg.mutpb,
        ngen=cfg.ngen,
        stats=stats,
        verbose=False,
    )

    best_individual = tools.selBest(pop, 1)[0]
    selected_idx = np.where(best_individual)[0]
    selected = [orbit_names[i] for i in selected_idx]

    fitness_values = toolbox.evaluate(best_individual, cfg.target_sats)

    total_rate = 0.0
    coverage_count = 0
    selected_families = set()
    family_breakdown = {}

    num_t, num_gs = vis_M.shape[1], vis_M.shape[2]

    for t in range(num_t):
        for k in range(num_gs):
            best_rate_tk = 0.0
            for j in selected_idx:
                if vis_M[j, t, k]:
                    best_rate_tk = max(best_rate_tk, rate_M[j, t, k])
                    fam = VALIDATED_ORBITS[orbit_names[j]]["family"]
                    selected_families.add(fam)
                    family_breakdown[fam] = family_breakdown.get(fam, 0) + 1
            if best_rate_tk > 0.0:
                total_rate += best_rate_tk
                coverage_count += 1

    coverage = coverage_count / (num_t * num_gs)
    diversity = len(selected_families)

    return selected, {
        "rate": total_rate,
        "coverage": coverage,
        "diversity": diversity,
        "families": selected_families,
        "family_breakdown": family_breakdown,
        "logbook": logbook,
        "fitness": fitness_values,
    }
