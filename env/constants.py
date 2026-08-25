# ============================================================
# Physical actions
# ============================================================

STAY = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

PHYSICAL_ACTIONS = (STAY, UP, RIGHT, DOWN, LEFT)

ACTION_TO_DELTA = {
    STAY: (0, 0),
    UP: (0, -1),
    RIGHT: (1, 0),
    DOWN: (0, 1),
    LEFT: (-1, 0),
}

ACTION_NAMES = {
    STAY: "STAY",
    UP: "UP",
    RIGHT: "RIGHT",
    DOWN: "DOWN",
    LEFT: "LEFT",
}


# ============================================================
# Communication actions
# ============================================================

NO_MESSAGE = 0
HELP = 1

COMMUNICATION_ACTIONS = (
    NO_MESSAGE,
    HELP,
)


# ============================================================
# Observation encoding
# ============================================================

EMPTY = 0
WALL = 1
OTHER_AGENT = 2
MOVABLE_OBJECT = 3
TARGET_OBJECT = 4
GOAL = 5