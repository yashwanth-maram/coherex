from enum import Enum


class TrackState(Enum):
    NEW = 0
    ACTIVE = 1
    DORMANT = 2
    TERMINATED = 3
