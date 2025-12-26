import argparse
import json
import sys

import numpy as np

from src.config import load_ea_config, load_iea_config, load_simulation_config
from src.new_optimizer.ea import ea_algorithm


def main():
    parser = argparse.ArgumentParser(description="Select an optimization algorithm.")
    parser.add_argument(
        "--opt",
        type=str,
        required=True,
        choices=["EA", "IEA", "GREEDY", "QL", "MEMETIC"],
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
        elif args.opt == "IEA":
            optimizer_config = load_iea_config("dataset/iea.json")


if __name__ == "__main__":
    main()
