import random


# ============================================================
# comm_scratch 시나리오 생성기
#
# Stage 0-7은 target/goal이 고정 좌표였고, 정책이 "탐색"이 아니라
# "그 좌표로 가는 법"을 배워버려서 Stage 8B에서 위치를 풀자마자
# 무너졌다. 이 생성기는 처음부터 target 위치를 맵 전체에서
# 랜덤으로 뽑는다. 난이도는 좌표를 고정해서가 아니라 아래 축들로
# 조절한다:
#
#   - target 크기 (고정 (w, h) 또는 범위에서 랜덤)
#   - target -> goal 거리 (좌상단 기준 맨해튼 거리 범위)
#   - 스폰 방식 (target 근처 반경 / 맵 전체 랜덤 / split)
#   - obstacle 개수
#
# target은 항상 벽에서 1칸 이상 떨어뜨린다. 미는 방향 반대쪽 면의
# 모든 칸에 에이전트가 서 있어야 밀 수 있으므로, 벽에 붙은 target은
# 벽에서 멀어지는 방향으로 영원히 못 민다 (Stage 8 생성기에서
# 8B1의 19%, 8B3 이후 32~34%가 이렇게 풀 수 없는 episode였다).
# ============================================================


TARGET_WALL_MARGIN = 1


class ScratchScenarioGenerator:

    def __init__(
        self,
        map_width=12,
        map_height=12,
        num_agents=4,
        target_size=None,
        target_size_range=((2, 4), (2, 3)),
        goal_distance_range=None,
        obstacle_count_range=(0, 0),
        obstacle_size_range=((1, 3), (1, 2)),
        spawn=None,
        max_attempts=200,
        goal_axis=None,
        allow_goal_overlap=False,
        obstacle_in_path=False,
    ):
        """
        target_size: (w, h) 고정. None이면 target_size_range
            에서 랜덤.
        goal_distance_range: (min, max) - target 좌상단과 goal
            좌상단 사이 맨해튼 거리. None이면 제한 없음.
        spawn:
            {"type": "near", "radius": r}
            {"type": "random"}
            {"type": "split", "near_count": k,
             "near_radius": r, "far_min_distance": d}
            {"type": "push_face", "radius": r}
                goal 반대편 옆면의 모든 칸에 에이전트를 세우고
                (goal_axis="horizontal" 필요), 나머지는 target
                근처 반경 r에 둔다. 여러 명이 같은 step에 같은
                방향을 골라야 하는 밀기를 처음 발견시키기 위한
                reverse curriculum 발판이다.
        goal_axis: None이면 goal 방향 제한 없음. "horizontal"이면
            goal을 target과 같은 행에만, "straight"면 같은 행 또는
            같은 열에만 둔다.
        obstacle_in_path: True면 obstacle_count_range 대신 1x1
            obstacle 하나를 target과 goal 사이 통로(target이 지나갈
            칸) 위에 둔다. goal_axis="straight"와 함께 쓴다. 경로를
            막은 obstacle을 치우는 동작을 가르치는 발판용.
        allow_goal_overlap: True면 goal이 target과 겹쳐도 된다.
            발판 레벨에서 "한 번만 밀면 성공"하는 배치(거리 1)를
            만들기 위함 - 겹치지 않게 하면 3x3은 최소 3칸 떨어져서
            3명이 연속 3번 동시에 밀어야 한다.
        """

        self.map_width = map_width
        self.map_height = map_height
        self.num_agents = num_agents

        self.target_size = target_size

        self.target_size_range = (
            target_size_range
        )

        self.goal_distance_range = (
            goal_distance_range
        )

        self.obstacle_count_range = (
            obstacle_count_range
        )

        self.obstacle_size_range = (
            obstacle_size_range
        )

        self.spawn = spawn or {
            "type": "random"
        }

        self.max_attempts = max_attempts

        self.goal_axis = goal_axis

        self.allow_goal_overlap = (
            allow_goal_overlap
        )

        self.obstacle_in_path = (
            obstacle_in_path
        )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _cells(x, y, width, height):

        return {
            (x + dx, y + dy)
            for dy in range(height)
            for dx in range(width)
        }

    def _sample_target_size(self):

        if self.target_size is not None:
            return self.target_size

        (w_min, w_max), (h_min, h_max) = (
            self.target_size_range
        )

        return (
            random.randint(w_min, w_max),
            random.randint(h_min, h_max),
        )

    def _goal_candidates(
        self,
        target_x,
        target_y,
        width,
        height,
    ):
        target_cells = self._cells(
            target_x,
            target_y,
            width,
            height,
        )

        candidates = []

        for y in range(
            self.map_height - height + 1
        ):
            for x in range(
                self.map_width - width + 1
            ):
                if (
                    self.goal_axis
                    == "horizontal"
                    and y != target_y
                ):
                    continue

                if (
                    self.goal_axis
                    == "straight"
                    and y != target_y
                    and x != target_x
                ):
                    continue

                distance = (
                    abs(x - target_x)
                    + abs(y - target_y)
                )

                if (
                    self.goal_distance_range
                    is not None
                ):
                    low, high = (
                        self.goal_distance_range
                    )

                    if not (
                        low
                        <= distance
                        <= high
                    ):
                        continue

                if (x, y) == (
                    target_x,
                    target_y,
                ):
                    continue

                if (
                    not self.allow_goal_overlap
                    and not self._cells(
                        x,
                        y,
                        width,
                        height,
                    ).isdisjoint(
                        target_cells
                    )
                ):
                    continue

                candidates.append((x, y))

        return candidates

    def _place_obstacles(
        self,
        occupied,
    ):
        obstacles = []

        count = random.randint(
            *self.obstacle_count_range
        )

        (ow_min, ow_max), (oh_min, oh_max) = (
            self.obstacle_size_range
        )

        for _ in range(count):

            width = random.randint(
                ow_min,
                ow_max,
            )

            height = random.randint(
                oh_min,
                oh_max,
            )

            for _ in range(
                self.max_attempts
            ):
                x = random.randint(
                    0,
                    self.map_width - width,
                )

                y = random.randint(
                    0,
                    self.map_height - height,
                )

                cells = self._cells(
                    x,
                    y,
                    width,
                    height,
                )

                if cells.isdisjoint(
                    occupied
                ):
                    obstacles.append(
                        {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                        }
                    )

                    occupied |= cells

                    break

            # 자리를 못 찾은 obstacle은 건너뛴다 (Stage 8
            # 생성기와 같은 규칙).

        return obstacles

    def _near_positions(
        self,
        target_cells,
        blocked,
        radius,
    ):
        candidates = [
            (x, y)

            for y in range(self.map_height)
            for x in range(self.map_width)

            if (x, y) not in blocked
            and min(
                abs(x - tx) + abs(y - ty)
                for tx, ty in target_cells
            )
            <= radius
        ]

        if len(candidates) < self.num_agents:
            return None

        chosen = random.sample(
            candidates,
            self.num_agents,
        )

        return {
            f"agent_{i}": list(position)
            for i, position
            in enumerate(chosen)
        }

    @staticmethod
    def _corridor_cells(
        target_x,
        target_y,
        width,
        height,
        goal_x,
        goal_y,
    ):
        """
        target이 goal까지 똑바로 갈 때 지나가는 칸 중 target과
        goal이 처음 차지하지 않는 칸 (같은 행/열 goal 전용).
        """

        if goal_y == target_y:
            xs = (
                range(target_x + width, goal_x)
                if goal_x > target_x
                else range(goal_x + width, target_x)
            )

            return {
                (x, target_y + dy)
                for x in xs
                for dy in range(height)
            }

        if goal_x == target_x:
            ys = (
                range(target_y + height, goal_y)
                if goal_y > target_y
                else range(goal_y + height, target_y)
            )

            return {
                (target_x + dx, y)
                for y in ys
                for dx in range(width)
            }

        return set()

    def _push_face_positions(
        self,
        target_x,
        target_y,
        width,
        height,
        goal_x,
        target_cells,
        radius,
    ):
        if (
            goal_x == target_x
            or height > self.num_agents
        ):
            return None

        push_x = (
            target_x - 1
            if goal_x > target_x
            else target_x + width
        )

        pushers = [
            (push_x, target_y + dy)
            for dy in range(height)
        ]

        others = self._near_positions(
            target_cells,
            target_cells | set(pushers),
            radius,
        )

        if others is None:
            return None

        positions = (
            pushers
            + [
                tuple(position)
                for position
                in others.values()
            ][
                : self.num_agents
                - len(pushers)
            ]
        )

        random.shuffle(positions)

        return {
            f"agent_{i}": list(position)
            for i, position
            in enumerate(positions)
        }

    # --------------------------------------------------------
    # Scenario
    # --------------------------------------------------------

    def sample(self, name="comm_scratch"):

        for _ in range(self.max_attempts):

            width, height = (
                self._sample_target_size()
            )

            margin = TARGET_WALL_MARGIN

            target_x = random.randint(
                margin,
                self.map_width
                - width
                - margin,
            )

            target_y = random.randint(
                margin,
                self.map_height
                - height
                - margin,
            )

            goal_candidates = (
                self._goal_candidates(
                    target_x,
                    target_y,
                    width,
                    height,
                )
            )

            if not goal_candidates:
                continue

            goal_x, goal_y = random.choice(
                goal_candidates
            )

            target_cells = self._cells(
                target_x,
                target_y,
                width,
                height,
            )

            goal_cells = self._cells(
                goal_x,
                goal_y,
                width,
                height,
            )

            occupied = (
                set(target_cells)
                | goal_cells
            )

            if self.obstacle_in_path:

                corridor = self._corridor_cells(
                    target_x,
                    target_y,
                    width,
                    height,
                    goal_x,
                    goal_y,
                ) - occupied

                if not corridor:
                    continue

                ox, oy = random.choice(
                    sorted(corridor)
                )

                obstacles = [
                    {
                        "x": ox,
                        "y": oy,
                        "width": 1,
                        "height": 1,
                    }
                ]

            else:
                obstacles = (
                    self._place_obstacles(
                        occupied
                    )
                )

            scenario = {
                "name": name,
                "target": {
                    "x": target_x,
                    "y": target_y,
                    "width": width,
                    "height": height,
                },
                "goal_cells": [
                    list(cell)
                    for cell in sorted(
                        goal_cells
                    )
                ],
                "obstacles": obstacles,
                "walls": [],
            }

            # 같은 행 goal이면 밀기 방향(좌/우)을 기록한다 (버킷용).
            if self.goal_axis == "horizontal":
                scenario["variant"] = (
                    "push_right"
                    if goal_x > target_x
                    else "push_left"
                )

            spawn_type = self.spawn["type"]

            if spawn_type == "near":

                obstacle_cells = set()

                for obstacle in obstacles:
                    obstacle_cells |= (
                        self._cells(
                            obstacle["x"],
                            obstacle["y"],
                            obstacle["width"],
                            obstacle["height"],
                        )
                    )

                positions = (
                    self._near_positions(
                        target_cells,
                        target_cells
                        | obstacle_cells,
                        self.spawn["radius"],
                    )
                )

                if positions is None:
                    continue

                scenario[
                    "agent_positions"
                ] = positions

            elif spawn_type == "push_face":

                positions = (
                    self._push_face_positions(
                        target_x,
                        target_y,
                        width,
                        height,
                        goal_x,
                        target_cells,
                        self.spawn["radius"],
                    )
                )

                if positions is None:
                    continue

                scenario[
                    "agent_positions"
                ] = positions

            elif spawn_type == "split":

                scenario["spawn_mode"] = {
                    "type": "split",
                    "near_count":
                        self.spawn[
                            "near_count"
                        ],
                    "near_radius":
                        self.spawn[
                            "near_radius"
                        ],
                    "far_min_distance":
                        self.spawn[
                            "far_min_distance"
                        ],
                }

            # "random"은 env 기본 동작(_randomize_agents)을
            # 그대로 쓴다.

            return scenario

        raise RuntimeError(
            "Could not generate a valid "
            "comm_scratch scenario."
        )
