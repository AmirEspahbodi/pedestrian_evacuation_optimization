from src.simulator.environment import Environment
from src.optimizer.iea import island_evolutionary_algorithm

if __name__ == "__main__":
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    random_domain = [domain for domain in environment.domains if domain.id == 10][0]
    emergency_accesses, fitness_value, history = island_evolutionary_algorithm(
        random_domain
    )
    print(fitness_value)
    print(emergency_accesses)
    print(history)
    with open("hustory.txt", "w") as fp:
        fp.write(str(history))
    with open("emergency_accesses.txt", "w") as fp:
        fp.write(
            f"emergency_accesses={emergency_accesses}, fitness_value={fitness_value}"
        )

    print("optimizing completed!")
    print(emergency_accesses, fitness_value)
