from src.simulator.environment import Environment
from src.optimizer.iea import iEA_optimizer

if __name__ == "__main__":
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    random_domain = [domain for domain in environment.domains if domain.id == 18][0]
    emergency_accesses, fitness_value, history = iEA_optimizer(
        random_domain
    )
    print(fitness_value)
    print(emergency_accesses)
    print(history)
    with open("iea_hustory.txt", "w") as fp:
        fp.write(str(history))
    with open("iea_emergency_accesses.txt", "w") as fp:
        fp.write(
            f"emergency_accesses={emergency_accesses}, fitness_value={fitness_value}"
        )

    print("iea optimizing completed!")
    print(emergency_accesses, fitness_value)
