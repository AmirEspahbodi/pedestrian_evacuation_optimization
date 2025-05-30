from pathlib import Path
import json
from typing import TypeVar, Generic, Type

T = TypeVar("T")


class ConfigurationReader(Generic[T]):
    def __init__(self, file_path: str | Path, config_class: Type[T]):
        self.file_path = Path(file_path)
        self._config: T | None = None
        self._config_class = config_class

    @property
    def config(self) -> T:
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def load_config(self) -> T:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.file_path}")

        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
            return self._config_class(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Configuration validation error: {e}")
