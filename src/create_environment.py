from pydantic import BaseModel, ConfigDict

from .simulator.environment import Environment

environment = Environment.from_json_file("dataset/environments/environment-example-supermarket.json")
