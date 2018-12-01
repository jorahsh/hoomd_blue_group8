from enum import Enum


class Stages(Enum):
    INIT = 0
    SHRINKING = 1
    MIXING = 2
    CURING = 3


class Integrators(Enum):
    NVE = 1
    NPT = 2
    LANGEVIN = 3
    NVT = 4

class NeighbourList(Enum):
    CELL = 1
    TREE = 2
