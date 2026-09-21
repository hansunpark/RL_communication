import random


# ============================================================
# Stage 8 Random Scenario Generator
#
# Stage 0-7은 사실상 "고정된 기하학 구조 + x축 lane shift"만 검증했다.
# 이 생성기는 target 크기, target/goal 위치, obstacle 개수/크기/위치를
# 독립적으로 랜덤화해서 env._configure_custom_scenario가 받는 dict
# 포맷의 시나리오를 만든다.
#
# 중요:
# env._check_success()는 target.cells() == goal_cells 완전 일치를
# 요구하므로, goal_cells은 반드시 target과 동일한 (width, height)로
# 생성해야 한다. 이 클래스는 항상 그렇게 만든다.
# ============================================================


class RandomScenarioGenerator:

    def __init__(
        self,
        map_width=12,
        map_height=12,
        randomize_size=False,
        randomize_position=False,
        position_radius=None,
        obstacle_count_range=(0, 0),
        target_size_range=((2, 4), (2, 3)),
        obstacle_size_range=((1, 3), (1, 2)),
        max_attempts=200,
    ):
        self.map_width = map_width
        self.map_height = map_height

        self.randomize_size = randomize_size
        self.randomize_position = randomize_position

        # target/goal 위치를 기존 기본 위치(anchor) 기준
        # +-position_radius 안에서만 흔든다. None이면 맵 전체에서
        # 완전 랜덤 (제약 없음).
        #
        # 8B를 한 번에 "완전 랜덤 위치"로 학습시켰더니, 학습이
        # 끝난 stage_7 policy가 저엔트로피 상태라 새로운 위치를
        # 전혀 탐색하지 못하고 200스텝 내내 실패했다 (성공률 0.0
        # 고정, target을 단 한 번도 건드리지 못함). Stage 3가
        # 스폰 반경을 3 -> 5 -> 완전 랜덤으로 점진적으로 늘렸던
        # 것과 같은 방식으로, 위치 랜덤화도 반경을 단계적으로
        # 늘려야 한다.
        self.position_radius = (
            position_radius
        )

        self.obstacle_count_range = (
            obstacle_count_range
        )

        self.target_size_range = (
            target_size_range
        )

        self.obstacle_size_range = (
            obstacle_size_range
        )

        self.max_attempts = max_attempts

        # Stage 0-6 기본 배치.
        # randomize_size / randomize_position이
        # 둘 다 False면 이 값 그대로 사용되어
        # 기존 stage_4~6 시나리오와 동일해진다.
        self.default_target = {
            "x": 4,
            "y": 2,
            "width": 3,
            "height": 2,
        }

        self.default_goal_offset_y = 7

    # --------------------------------------------------------
    # Target size
    # --------------------------------------------------------

    def _sample_target_size(self):

        if not self.randomize_size:
            return (
                self.default_target["width"],
                self.default_target["height"],
            )

        (w_min, w_max), (h_min, h_max) = (
            self.target_size_range
        )

        width = random.randint(w_min, w_max)
        height = random.randint(h_min, h_max)

        return width, height

    # --------------------------------------------------------
    # Rectangle placement
    # --------------------------------------------------------

    def _sample_rect_position(
        self,
        width,
        height,
        forbidden_cells,
    ):
        """
        맵 안에 완전히 들어가고 forbidden_cells와 겹치지
        않는 (x, y)를 찾는다. 실패하면 None.
        """

        if (
            width > self.map_width
            or height > self.map_height
        ):
            return None

        for _ in range(self.max_attempts):

            x = random.randint(
                0,
                self.map_width - width,
            )

            y = random.randint(
                0,
                self.map_height - height,
            )

            cells = {
                (x + dx, y + dy)
                for dy in range(height)
                for dx in range(width)
            }

            if cells.isdisjoint(
                forbidden_cells
            ):
                return x, y, cells

        return None

    def _sample_rect_position_near(
        self,
        anchor_x,
        anchor_y,
        width,
        height,
        forbidden_cells,
    ):
        """
        anchor(x, y) 기준 +-position_radius 범위 안에서
        맵 안에 들어가고 forbidden_cells와 겹치지 않는
        (x, y)를 찾는다.

        position_radius가 None이면 맵 전체에서
        _sample_rect_position와 동일하게 동작한다.
        """

        if self.position_radius is None:
            return self._sample_rect_position(
                width,
                height,
                forbidden_cells,
            )

        x_min = max(
            0,
            anchor_x - self.position_radius,
        )

        x_max = min(
            self.map_width - width,
            anchor_x + self.position_radius,
        )

        y_min = max(
            0,
            anchor_y - self.position_radius,
        )

        y_max = min(
            self.map_height - height,
            anchor_y + self.position_radius,
        )

        if (
            x_min > x_max
            or y_min > y_max
        ):
            return None

        for _ in range(self.max_attempts):

            x = random.randint(
                x_min,
                x_max,
            )

            y = random.randint(
                y_min,
                y_max,
            )

            cells = {
                (x + dx, y + dy)
                for dy in range(height)
                for dx in range(width)
            }

            if cells.isdisjoint(
                forbidden_cells
            ):
                return x, y, cells

        return None

    # --------------------------------------------------------
    # Scenario
    # --------------------------------------------------------

    def sample(self, name="stage8"):

        target_width, target_height = (
            self._sample_target_size()
        )

        if self.randomize_position:

            placed = (
                self._sample_rect_position_near(
                    self.default_target["x"],
                    self.default_target["y"],
                    target_width,
                    target_height,
                    forbidden_cells=set(),
                )
            )

            if placed is None:
                raise RuntimeError(
                    "Could not place target."
                )

            target_x, target_y, target_cells = (
                placed
            )

        else:
            target_x = self.default_target["x"]
            target_y = self.default_target["y"]

            target_cells = {
                (target_x + dx, target_y + dy)
                for dy in range(target_height)
                for dx in range(target_width)
            }

        target = {
            "x": target_x,
            "y": target_y,
            "width": target_width,
            "height": target_height,
        }

        # ----------------------------------------------------
        # Goal: target과 동일 shape, target과 겹치지 않는 위치
        # ----------------------------------------------------

        if self.randomize_position:

            # goal은 target의 (랜덤화된) 새 위치가 아니라
            # 기존 기본 goal 위치를 기준으로 흔든다. target
            # anchor 기준으로 흔들면 target과 goal이 항상
            # 붙어 있게 되어 "떨어진 곳으로 옮기기" 문제가
            # 사라진다.
            goal_anchor_x = (
                self.default_target["x"]
            )

            goal_anchor_y = min(
                self.default_target["y"]
                + self.default_goal_offset_y,

                self.map_height
                - target_height,
            )

            placed = (
                self._sample_rect_position_near(
                    goal_anchor_x,
                    goal_anchor_y,
                    target_width,
                    target_height,
                    forbidden_cells=target_cells,
                )
            )

            if placed is None:
                raise RuntimeError(
                    "Could not place goal "
                    "without overlapping "
                    "target."
                )

            _, _, goal_cells_set = placed

        else:
            goal_x = target_x

            goal_y = min(
                target_y
                + self.default_goal_offset_y,

                self.map_height
                - target_height,
            )

            goal_cells_set = {
                (goal_x + dx, goal_y + dy)
                for dy in range(target_height)
                for dx in range(target_width)
            }

        goal_cells = [
            list(cell)
            for cell in goal_cells_set
        ]

        occupied = (
            set(target_cells)
            | set(goal_cells_set)
        )

        # ----------------------------------------------------
        # Obstacles
        # ----------------------------------------------------

        obstacles = []

        obstacle_count = random.randint(
            *self.obstacle_count_range
        )

        (ow_min, ow_max), (oh_min, oh_max) = (
            self.obstacle_size_range
        )

        for _ in range(obstacle_count):

            width = random.randint(
                ow_min,
                ow_max,
            )

            height = random.randint(
                oh_min,
                oh_max,
            )

            placed = (
                self._sample_rect_position(
                    width,
                    height,
                    forbidden_cells=occupied,
                )
            )

            if placed is None:
                # 자리를 못 찾으면 이 obstacle은
                # 건너뛴다. 개수 정확성보다
                # 유효한 시나리오 생성이 우선이다.
                continue

            x, y, cells = placed

            obstacles.append(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            )

            occupied |= cells

        return {
            "name": name,
            "target": target,
            "goal_cells": goal_cells,
            "obstacles": obstacles,
            "walls": [],
        }
