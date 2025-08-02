from src.simulator.environment import Environment
from src.optimizer.memetic import MemeticAlgorithm
import json


if __name__ == "__main__":
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    random_domain = [domain for domain in environment.domains if domain.id == 18][0]
    memetic_algo = MemeticAlgorithm(random_domain)
    best_overall_individual, best_overall_fitness, history, time_to_best = (
        memetic_algo.run()
    )

    print(f"best_overall_individual = {best_overall_individual}")
    print(f"best_overall_fitness = {best_overall_fitness}")
    print(f"history = {history}")
    print(f"time_to_best = {time_to_best}")

    data = {
        "best_overall_individual": best_overall_individual,
        "best_overall_fitness": best_overall_fitness,
        "history": history,
        "time_to_best": time_to_best
    }

    with open("memetic_results.json", "w") as f:
        json.dump(data, f, indent=4)


    print("MemeticAlgorithm optimizing completed!")
