from collections import Counter
from dataclasses import asdict, dataclass
from math import log
from typing import Callable

import numpy as np
from pygad import GA

from map import Map, Tile

# TODO: avoid hardcoding map size
WIDTH = 40
HEIGHT = 30


def fitness_func(ga: GA, solution: np.ndarray, index: int) -> float:
    DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    # global_counter = Counter(solution)
    # desired_num = Counter({Tile.GRASS: 70, Tile.MOUNTAIN: 25, Tile.RIVER: 1080, Tile.RIVERSTONE: 20, Tile.ROCK: 5})

    fitness = 0
    # for tile in Tile:
    #     fitness -= abs(global_counter[tile] - desired_num[tile]) / log(desired_num[tile], 10)

    for i, tile in enumerate(solution):
        y, x = divmod(i, WIDTH)

        # TODO: optimize surrounding counting
        valids = (0 <= y + dy < HEIGHT and 0 <= x + dx < WIDTH for dx, dy in DIRECTIONS)
        valid_indices = (i + dy * WIDTH + dx for (dx, dy), valid in zip(DIRECTIONS, valids) if valid)
        surroundings = Counter(solution[j] for j in valid_indices)

        match tile:
            case Tile.GRASS:
                fitness += surroundings[Tile.GRASS] + surroundings[Tile.ROCK]
                fitness -= (
                    0
                    if 15 <= x < 25 and 10 <= y < 20
                    else min(abs(15 - x), abs(x - 24)) + min(abs(10 - y), abs(y - 19))
                )
            case Tile.MOUNTAIN:
                fitness += surroundings[Tile.MOUNTAIN]
                fitness -= (
                    0
                    if 18 <= x < 23 and 13 <= y < 18
                    else min(abs(18 - x), abs(x - 22)) + min(abs(13 - y), abs(y - 17))
                )
            case Tile.RIVER:
                fitness -= (
                    min(abs(15 - x), abs(x - 24)) + min(abs(10 - y), abs(y - 19))
                    if 15 <= x < 25 and 10 <= y < 20
                    else 0
                )
            case Tile.RIVERSTONE:
                fitness -= (
                    min(abs(15 - x), abs(x - 24)) + min(abs(10 - y), abs(y - 19))
                    if 15 <= x < 25 and 10 <= y < 20
                    else 0
                )
            case Tile.ROCK:
                fitness += surroundings[Tile.GRASS]
                fitness -= (
                    0
                    if 15 <= x < 25 and 10 <= y < 20
                    else min(abs(15 - x), abs(x - 24)) + min(abs(10 - y), abs(y - 19))
                )
            case _:
                continue

    return fitness


@dataclass
class GAConfig:
    num_generations: int
    num_parents_mating: int

    fitness_func: Callable = fitness_func
    parent_selection_type: str = "tournament"
    sol_per_pop: int = 32
    initial_population: np.ndarray | None = None
    crossover_probability: float = 0.7
    mutation_probability: float = 0.05
    mutation_by_replacement: bool = True

    keep_parents: int = -1
    keep_elitism: int = 1
    K_tournament: int = 3
    crossover_type: str = "single_point"
    mutation_type: str = "random"


def gen_map_ga(width: int, height: int, ga_config: GAConfig) -> Map:
    num_genes = width * height
    gene_space = np.arange(1, len(Tile) + 1)

    # TODO: support early stop and progress bar
    ga = GA(num_genes=num_genes, gene_type=int, gene_space=gene_space, suppress_warnings=True, **asdict(ga_config))
    ga.run()
    ga.plot_fitness()

    m = Map(width=width, height=height)
    best_solution, _, _ = ga.best_solution()
    m.tiles = np.asarray(best_solution, dtype=np.uint8)

    return m
