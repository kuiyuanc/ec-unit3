from collections import Counter

import numpy as np

from gen_map_ga import HEIGHT, WIDTH, GAConfig, gen_map_ga
from map import Tile


def main():
    sol_per_pop = 256

    # for i in range(10):
    #     ga_config = GAConfig(
    #         num_generations=1000, num_parents_mating=16, sol_per_pop=sol_per_pop, mutation_probability=0.1
    #     )

    #     m = gen_map_ga(WIDTH, HEIGHT, ga_config)
    #     m.save(path="output", filename=f"rand_{i}")

    desired_num = Counter({Tile.GRASS: 70, Tile.MOUNTAIN: 25, Tile.RIVER: 1080, Tile.RIVERSTONE: 20, Tile.ROCK: 5})
    population = tuple(desired_num.elements())

    for i in range(10):
        initial_population = np.asarray([np.random.permutation(population) for _ in range(sol_per_pop)])

        ga_config = GAConfig(
            num_generations=75,
            num_parents_mating=128,
            initial_population=initial_population,
            mutation_probability=0.1,
            mutation_type="swap",
            mutation_by_replacement=False
        )

        m = gen_map_ga(WIDTH, HEIGHT, ga_config)
        m.save(path="output", filename=f"rand_{i}")


if __name__ == "__main__":
    main()
