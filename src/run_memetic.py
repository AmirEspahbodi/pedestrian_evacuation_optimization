from src.simulator.environment import Environment
from src.optimizer.memetic import MemeticAlgorithm

if __name__ == "__main__":
    environment = Environment.from_json_file(
        "dataset/environments/environments_supermarket.json"
    )
    random_domain = [domain for domain in environment.domains if domain.id == 18][0]
    memetic_algo = MemeticAlgorithm(random_domain)
    best_solution, best_fitness, history = memetic_algo.run()

    print(best_solution)
    print(best_fitness)

    print(history)
    with open("ea_hustory.txt", "w") as fp:
        fp.write(str(history))
    with open("ea_emergency_accesses.txt", "w") as fp:
        fp.write(f"emergency_accesses={best_solution}, fitness_value={best_fitness}")

    print("ea optimizing completed!")
    print(best_solution, best_fitness)
