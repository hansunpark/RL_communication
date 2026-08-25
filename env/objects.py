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
    """
    직사각형 movable object.

    x, y:
        object의 왼쪽 위 cell 좌표

    width:
        가로 길이

    height:
        세로 길이
    """

    object_id: int
    x: int
    y: int
    width: int
    height: int
    is_target: bool = False

    def cells(self) -> Set[Position]:
        """
        현재 object가 점유하는 모든 cell을 반환.
        """

        return {
            (self.x + dx, self.y + dy)
            for dy in range(self.height)
            for dx in range(self.width)
        }

    def moved_cells(self, direction: int) -> Set[Position]:
        """
        direction 방향으로 한 칸 이동했다고 가정했을 때
        점유하게 될 cell을 반환.

        실제 object 위치는 변경하지 않는다.
        """

        dx, dy = ACTION_TO_DELTA[direction]

        return {
            (x + dx, y + dy)
            for x, y in self.cells()
        }

    def contact_cells(self, direction: int) -> Set[Position]:
        """
        object를 direction 방향으로 밀기 위해
        agent들이 반드시 점유해야 하는 접촉면 cell.

        예:
        RIGHT

        A -> ■ ■ ■
        A -> ■ ■ ■

        DOWN

        A A A
        ↓ ↓ ↓
        ■ ■ ■
        ■ ■ ■
        """

        if direction == RIGHT:
            return {
                (self.x - 1, self.y + dy)
                for dy in range(self.height)
            }

        if direction == LEFT:
            return {
                (self.x + self.width, self.y + dy)
                for dy in range(self.height)
            }

        if direction == DOWN:
            return {
                (self.x + dx, self.y - 1)
                for dx in range(self.width)
            }

        if direction == UP:
            return {
                (self.x + dx, self.y + self.height)
                for dx in range(self.width)
            }

        return set()

    def move(self, direction: int) -> None:
        """
        object의 실제 위치를 한 칸 이동.
        """

        dx, dy = ACTION_TO_DELTA[direction]

        self.x += dx
        self.y += dy