from enum import StrEnum


class CAState(StrEnum):
    EMPTY = "empty"
    OBSTACLE = "obstacle"
    OCCUPIED = "occupied"
    ACCESS = "access"
    ACCESS_OCCUPIED = "access_occupied"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    @property
    def is_passable(self) -> bool:
        """Check if the state allows pedestrian movement."""
        return self in {CAState.EMPTY, CAState.ACCESS}

    @property
    def has_pedestrian(self) -> bool:
        """Check if the state contains a pedestrian."""
        return self in {CAState.OCCUPIED, CAState.ACCESS_OCCUPIED}

    @property
    def is_exit(self) -> bool:
        """Check if the state is an exit."""
        return self in {CAState.ACCESS, CAState.ACCESS_OCCUPIED}
