from .simulator import Environment, Domain, CAState

environment = Environment.from_json_file("dataset/environments/environment-example-supermarket.json")

count = 1
for cell1D in environment.domains[0].cells:
    for cell in cell1D:
        count += 1
        # if cell.state == CAState.ACCESS:
        #     print(f"Access at ({cell.x}, {cell.y})")
        # elif cell.state == CAState.OBSTACLE:
        #     print(f"Obstacle at (width={cell.x}, height={cell.y})")
print(f"Total cells processed: {count}")