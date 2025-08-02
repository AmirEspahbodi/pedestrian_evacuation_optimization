from src.simulator.environment import Environment
from src.optimizer.ql import q_learning_exit_optimizer
import json


if __name__ == "__main__":
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    random_domain = [domain for domain in environment.domains if domain.id == 10][0]
    best_overall_individual, best_overall_fitness, history, time_to_best = q_learning_exit_optimizer(
        random_domain
    )
    data = {
        "best_overall_individual": [int(i) for i in best_overall_individual],
        "best_overall_fitness": float(best_overall_fitness),
        "history": history,
        "time_to_best": time_to_best
    }

    with open("ql_optimizing_results.json", "w") as f:
        json.dump(data, f, indent=4)

    print("ql_optimizing completed!")
