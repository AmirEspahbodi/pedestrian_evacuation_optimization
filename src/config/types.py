from enum import StrEnum


class UpdateStrategy(StrEnum):
    RANDOM = "random"
    SEQUENTIAL = "sequential"
    REVERSE = "reverse"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class Neighborhood(StrEnum):
    MOORE = "Moore"
    VON_NEUMANN = "VonNeumann"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class OptimizerStrategy(StrEnum):
    EA = "ea"
    GREEDY = "greedy"
    IEA = "iea"
