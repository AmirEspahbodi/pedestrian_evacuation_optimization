from enum import StrEnum


class CAState(StrEnum):
    EMPTY = "empty"
    OBSTACLE = "obstacle"
    OCCUPIED = "occupied"
    ACCESS = "access"
    ACCESS_OCCUPIED = "access_occupied"
    EMERGENCY_ACCESS = "emergency_access"
    EMERGENCY_ACCESS_OCCUPIED = "emergency_access_occupied"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value
