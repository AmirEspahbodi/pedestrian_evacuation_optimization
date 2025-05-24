from pydantic import BaseModel, ConfigDict
from .domain import Domain


class Gateway(BaseModel):
    id:int
    domain1:int
    domain2:int
    model_config = ConfigDict()


class Environment(BaseModel):
    domains: list[Domain]
    gateways: list[Gateway]
    @classmethod
    def from_json_file(cls, file_path: str) -> "Environment":
        with open(file_path, 'r') as f:
            json_data_str = f.read()
        return cls.model_validate_json(json_data_str)
    model_config = ConfigDict()
