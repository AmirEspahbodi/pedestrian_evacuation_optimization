import argparse
import json

import numpy as np
from typing_extensions import Any

from src.config import load_ea_config, load_iea_config, load_simulation_config
from src.optimizer._01ea import ea_algorithm
from src.optimizer._02greedy import greedy_algorithm
from src.optimizer._03iea import iea_optimizer
from src.optimizer._04ql import q_learning_exit_optimizer
from src.optimizer._05cma_es_ma import (
    run_cma_es_optimization as run_cma_es_ma_optimization,
)
from src.optimizer._06GWO import integer_enhanced_gwo
from src.optimizer._07misc import GAConfig, MISOConfig, MISOIntegerOptimizer
from src.optimizer._08memetic import MemeticAlgorithm


def main():
    parser = argparse.ArgumentParser(description="Select an optimization algorithm.")
    parser.add_argument(
        "--opt",
        type=str,
        required=True,
        choices=[
            "EA",
            "IEA",
            "GREEDY",
            "QL",
            "MEMETIC",
            "CMA-ES-MA",
            "GWO",
            "MISC",
        ],
        help="The optimization method (EA, IEA, GREEDY, QL, MEMETIC)",
    )
    args = parser.parse_args()
    simulator_config = load_simulation_config("dataset/simulation.json")

    DATASET_FILES = [
        "dataset/p400/finall_grid_400_1.txt",
        "dataset/p400/finall_grid_400_2.txt",
        "dataset/p400/finall_grid_400_3.txt",
        "dataset/p400/finall_grid_400_4.txt",
        "dataset/p400/finall_grid_400_5.txt",
        "dataset/p400/finall_grid_400_6.txt",
        "dataset/p400/finall_grid_400_7.txt",
        "dataset/p400/finall_grid_400_8.txt",
        "dataset/p400/finall_grid_400_9.txt",
        "dataset/p400/finall_grid_400_10.txt",
    ]

    with open(DATASET_FILES[0], "r") as f:
        raw_gird = [[int(c) for c in line.strip()] for line in f]
        clean_gird = [[0 if cell == 3 else cell for cell in row] for row in raw_gird]

    pedestrian_confs = []
    for dataset_file in DATASET_FILES:
        with open(dataset_file, "r") as f:
            raw_grid = [[int(c) for c in line.strip()] for line in f]
            pedestrian = np.array(
                [[1 if cell == 3 else 0 for cell in row] for row in raw_grid],
                dtype=np.int8,
            )
            pedestrian_confs.append(pedestrian)

        def store_as_json(data, filename):
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                    print(f"Successfully saved results to {filename}")
            except IOError as e:
                print(f"Error writing to file {filename}: {e}")
            except TypeError as e:
                print(
                    f"Serialization Error: Ensure all data types are JSON compatible. {e}"
                )

        if args.opt == "EA":
            ea_config = load_ea_config("dataset/ea.json")
            best_overall_genes, best_overall_fitness, time_to_best, history = (
                ea_algorithm(pedestrian_confs, clean_gird, simulator_config, ea_config)
            )
            data = {
                "best_overall_genes": best_overall_genes,
                "best_overall_fitness": best_overall_fitness,
                "time_to_best": time_to_best,
                "history": history,
            }
            filename = "results/ea_result.json"
            store_as_json(data, filename)
        elif args.opt == "GREEDY":
            e_solutions, best_overall_fitness, time_of_best = greedy_algorithm(
                pedestrian_confs, clean_gird, simulator_config
            )
            data = {
                "e_solutions": e_solutions,
                "best_overall_fitness": best_overall_fitness,
                "time_of_best": time_of_best,
            }
            filename = "results/greedy_result.json"
            store_as_json(data, filename)
        elif args.opt == "IEA":
            iea_config = load_iea_config("dataset/iea.json")
            global_best_individual, global_best_fitness, history, time_to_best = (
                iea_optimizer(
                    pedestrian_confs, clean_gird, simulator_config, iea_config
                )
            )
            data: dict[str, Any] = {
                "global_best_individual": global_best_individual,
                "global_best_fitness": global_best_fitness,
                "history": history,
                "time_to_best": time_to_best,
            }
            filename = "results/iea_result.json"
            store_as_json(data, filename)
        elif args.opt == "MEMETIC":
            iea_config = load_iea_config("dataset/iea.json")
            memetic_algorithm = MemeticAlgorithm(
                pedestrian_confs, clean_gird, simulator_config, iea_config
            )
            best_overall_individual, best_overall_fitness, history, time_to_best = (
                memetic_algorithm.run()
            )
            data: dict[str, Any] = {
                "best_overall_individual": best_overall_individual,
                "best_overall_fitness": best_overall_fitness,
                "history": history,
                "time_to_best": time_to_best,
            }
            filename = "results/memetic_result.json"
            store_as_json(data, filename)
        elif args.opt == "QL":
            iea_config = load_iea_config("dataset/iea.json")
            best_solution, best_fitness, history, time_to_best = (
                q_learning_exit_optimizer(
                    pedestrian_confs, clean_gird, simulator_config, iea_config
                )
            )
            data: dict[str, Any] = {
                "best_solution": best_solution,
                "best_fitness": best_fitness,
                "history": history,
                "time_to_best": time_to_best,
            }
            filename = "results/ql_result.json"
            store_as_json(data, filename)
        elif args.opt == "CMA-ES-MA":
            iea_config = load_iea_config("dataset/iea.json")
            run_cma_es_ma_optimization(
                clean_gird, pedestrian_confs, simulator_config, iea_config
            )
        elif args.opt == "GWO":
            blocked = {}
            iea_config = load_iea_config("dataset/iea.json")

            def valid(x: np.ndarray) -> bool:
                return all(int(v) not in blocked for v in x)

            res = integer_enhanced_gwo(
                clean_gird,
                pedestrian_confs,
                simulator_config,
                iea_config,
                valid_fn=valid,
            )
            print("best_x:", res.best_x, "best_f:", res.best_f, "evals:", res.n_evals)
        elif args.opt == "MISC":
            iea_config = load_iea_config("dataset/iea.json")

            def is_valid(x: np.ndarray) -> bool:
                return True

            opt = MISOIntegerOptimizer(
                clean_gird,
                pedestrian_confs,
                simulator_config,
                iea_config,
                config=MISOConfig(nmax=250, enable_local_search=True),
                ga_config=GAConfig(pop_size=70, generations=70),
                seed=123,
                is_valid_fn=is_valid,
                repair_fn=None,
            )

            result = opt.optimize()
            print("best_x =", result["best_x"])
            print("best_f =", result["best_f"])
            print("evaluations =", result["evaluations"])


if __name__ == "__main__":
    main()
