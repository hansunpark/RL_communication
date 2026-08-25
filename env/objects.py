from dataclasses import dataclass
from typing import Set, Tuple

from .constants import (
    UP,
    RIGHT,
    DOWN,
    LEFT,
    ACTION_TO_DELTA,
)


Position = Tuple[int, int]


@dataclass
class RectObject:
    object_id: int

    # left-top cell
    x: int
    y: int

    width: int
    height: int

    is_target: bool = False

    def cells(self) -> Set[Position]:
        return {
            (self.x + dx, self.y + dy)
            for dy in range(self.height)
            for dx in range(self.width)
        }

    def moved_cells(
        self,
        direction: int,
    ) -> Set[Position]:

        dx, dy = ACTION_TO_DELTA[direction]

        return {
            (x + dx, y + dy)
            for x, y in self.cells()
        }

    def contact_cells(
        self,
        direction: int,
    ) -> Set[Position]:

        # object moves RIGHT
        # agents must fill left face
        if direction == RIGHT:
            return {
                (self.x - 1, self.y + dy)
                for dy in range(self.height)
            }

        # object moves LEFT
        # agents must fill right face
        if direction == LEFT:
            return {
                (
                    self.x + self.width,
                    self.y + dy,
                )
                for dy in range(self.height)
            }

        # object moves DOWN
        # agents must fill upper face
        if direction == DOWN:
            return {
                (
                    self.x + dx,
                    self.y - 1,
                )
                for dx in range(self.width)
            }

        # object moves UP
        # agents must fill lower face
        if direction == UP:
            return {
                (
                    self.x + dx,
                    self.y + self.height,
                )
                for dx in range(self.width)
            }

        return set()

    def move(
        self,
        direction: int,
    ) -> None:

        dx, dy = ACTION_TO_DELTA[direction]

        self.x += dx
        self.y += dy