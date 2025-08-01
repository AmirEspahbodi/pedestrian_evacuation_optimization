from src.simulator.environment import Environment
from src.optimizer.greedy import greedy_algorithm

if __name__ == "__main__":
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    random_domain = [domain for domain in environment.domains if domain.id == 10][0]
    emergency_accesses, fitness_value, time_of_best = greedy_algorithm(random_domain)
    print(f"time_of_best = {time_of_best}")
    print(f"fitness_value = {fitness_value}")
    print(f"emergency_accesses = {emergency_accesses}")
    with open("greedy_emergency_accesses.txt", "w") as fp:
        fp.write(
            f"emergency_accesses={emergency_accesses}, fitness_value={fitness_value}"
        )

    print("greedy optimizing completed!")
    print(emergency_accesses, fitness_value)
