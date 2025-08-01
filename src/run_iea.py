from src.simulator.environment import Environment
from src.optimizer.iea import iEA_optimizer

if __name__ == "__main__":
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    random_domain = [domain for domain in environment.domains if domain.id == 10][0]
    emergency_accesses, fitness_value, history, time_to_best = iEA_optimizer(
        random_domain
    )
    print(f"fitness_value = {fitness_value}")
    print(f"emergency_accesses = {emergency_accesses}")
    print(f"time_to_best = {time_to_best}")
    print(history)
    with open("iea_hustory.txt", "w") as fp:
        fp.write(str(history))
    with open("iea_emergency_accesses.txt", "w") as fp:
        fp.write(
            f"emergency_accesses={emergency_accesses}, fitness_value={fitness_value}"
        )

    print("iea optimizing completed!")
    print(emergency_accesses, fitness_value)
