from src.simulator.environment import Environment
from src.optimizer.ql import q_learning_exit_optimizer

if __name__ == "__main__":
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    random_domain = [domain for domain in environment.domains if domain.id == 18][0]
    best_solution, best_fitness, history, time_to_best = q_learning_exit_optimizer(
        random_domain
    )
    print(f"best_solution = {best_solution}")
    print(f"best_fitness = {best_fitness}")
    print(f"history = {history}")
    print(f"time_to_best = {time_to_best}")

    print("ql_optimizing completed!")
