from src.simulator.environment import Environment
from src.optimizer.ea import evolutionary_algorithm

if __name__ == "__main__":
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    random_domain = [domain for domain in environment.domains if domain.id == 18][0]
    emergency_accesses, fitness_value, history, time_to_best = evolutionary_algorithm(
        random_domain
    )
    print(f"fitness_value = {fitness_value}")
    print(f"emergency_accesses = {emergency_accesses}")
    print(f"time_to_best = {time_to_best}")
    print(history)
    
    with open("ea_hustory.txt", "w") as fp:
        fp.write(str(history))
    with open("ea_emergency_accesses.txt", "w") as fp:
        fp.write(
            f"emergency_accesses={emergency_accesses}, fitness_value={fitness_value}, time_to_best = {time_to_best}"
        )

    print("ea optimizing completed!")
    print(emergency_accesses, fitness_value)
