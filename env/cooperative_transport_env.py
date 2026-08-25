from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from .constants import (
    STAY,
    UP,
    RIGHT,
    DOWN,
    LEFT,
    ACTION_TO_DELTA,
    NO_MESSAGE,
    HELP,
    EMPTY,
    WALL,
    OTHER_AGENT,
    MOVABLE_OBJECT,
    TARGET_OBJECT,
    GOAL,
)

from .objects import RectObject


Position = Tuple[int, int]


class CooperativeTransportEnv(ParallelEnv):

    metadata = {
        "name": "cooperative_transport_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        width: int = 12,
        height: int = 12,
        num_agents: int = 4,
        vision_size: int = 5,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if vision_size % 2 == 0:
            raise ValueError("vision_size must be odd.")

        self.width = width
        self.height = height

        self.num_agents = num_agents

        self.vision_size = vision_size
        self.vision_radius = vision_size // 2

        self.max_steps = max_steps
        self.render_mode = render_mode

        self.possible_agents = [
            f"agent_{i}"
            for i in range(num_agents)
        ]

        self.agents: List[str] = []

        # ====================================================
        # Global environment state
        # ====================================================

        self.agent_positions: Dict[str, Position] = {}

        self.objects: List[RectObject] = []

        self.goal_cells: Set[Position] = set()

        # 고정 벽.
        # 현재 기본 환경에서는 바깥 경계만 벽 역할을 하고,
        # 내부 wall은 비워둔다.
        self.walls: Set[Position] = set()

        self.step_count = 0

        self.rng = np.random.default_rng()

        # receiver -> [(sender_id, dx, dy), ...]
        self.received_messages: Dict[
            str,
            List[Tuple[int, int, int]]
        ] = {}

        # ====================================================
        # Action spaces
        # ====================================================

        # action[0] = physical action
        #
        # 0 STAY
        # 1 UP
        # 2 RIGHT
        # 3 DOWN
        # 4 LEFT
        #
        # action[1] = communication
        #
        # 0 NONE
        # 1 HELP

        self.action_spaces = {
            agent: spaces.MultiDiscrete([5, 2])
            for agent in self.possible_agents
        }

        # ====================================================
        # Observation spaces
        # ====================================================

        self.max_objects = 10

        self.observation_spaces = {}

        for agent in self.possible_agents:

            self.observation_spaces[agent] = spaces.Dict(
                {
                    # 5x5 local observation
                    "local_map": spaces.Box(
                        low=0,
                        high=5,
                        shape=(
                            self.vision_size,
                            self.vision_size,
                        ),
                        dtype=np.int8,
                    ),

                    # 어느 cell이 같은 object인지 구분
                    "object_ids": spaces.Box(
                        low=0,
                        high=self.max_objects,
                        shape=(
                            self.vision_size,
                            self.vision_size,
                        ),
                        dtype=np.int16,
                    ),

                    # goal이 object 아래 가려져도 정보 유지
                    "goal_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(
                            self.vision_size,
                            self.vision_size,
                        ),
                        dtype=np.int8,
                    ),

                    # agent 자신의 절대좌표
                    "position": spaces.Box(
                        low=np.array(
                            [0, 0],
                            dtype=np.int16,
                        ),
                        high=np.array(
                            [
                                self.width - 1,
                                self.height - 1,
                            ],
                            dtype=np.int16,
                        ),
                        dtype=np.int16,
                    ),

                    # object 접촉 시 전체 크기 공개
                    #
                    # 각 row:
                    #
                    # [
                    #   touching,
                    #   width,
                    #   height,
                    #   is_target
                    # ]
                    #
                    "touching_objects": spaces.Box(
                        low=0,
                        high=max(
                            self.width,
                            self.height,
                        ),
                        shape=(
                            self.max_objects,
                            4,
                        ),
                        dtype=np.int16,
                    ),

                    # HELP messages
                    #
                    # [
                    #   active,
                    #   relative_dx,
                    #   relative_dy,
                    #   sender_id
                    # ]
                    #
                    "messages": spaces.Box(
                        low=-max(
                            self.width,
                            self.height,
                        ),
                        high=max(
                            self.width,
                            self.height,
                        ),
                        shape=(
                            self.num_agents - 1,
                            4,
                        ),
                        dtype=np.int16,
                    ),
                }
            )

    # ========================================================
    # PettingZoo spaces
    # ========================================================

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]

        self.step_count = 0

        # ====================================================
        # Target object
        # ====================================================

        target = RectObject(
            object_id=0,
            x=4,
            y=2,
            width=3,
            height=2,
            is_target=True,
        )

        # ====================================================
        # Movable obstacle
        # ====================================================

        obstacle = RectObject(
            object_id=1,
            x=4,
            y=6,
            width=3,
            height=1,
            is_target=False,
        )

        self.objects = [
            target,
            obstacle,
        ]

        # ====================================================
        # Goal
        # ====================================================

        self.goal_cells = {
            (x, y)
            for y in range(9, 11)
            for x in range(4, 7)
        }

        # ====================================================
        # Random agent positions
        # ====================================================

        forbidden = set()

        for obj in self.objects:
            forbidden |= obj.cells()

        forbidden |= self.walls

        available_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in forbidden
        ]

        selected_indices = self.rng.choice(
            len(available_cells),
            size=self.num_agents,
            replace=False,
        )

        self.agent_positions = {
            agent: available_cells[index]
            for agent, index
            in zip(
                self.agents,
                selected_indices,
            )
        }

        self.received_messages = {
            agent: []
            for agent in self.agents
        }

        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }

        infos = {
            agent: {}
            for agent in self.agents
        }

        if self.render_mode == "human":
            self.render()

        return observations, infos

    # ========================================================
    # Step
    # ========================================================

    def step(self, actions):

        if not self.agents:
            return {}, {}, {}, {}, {}

        current_agents = self.agents[:]

        self.step_count += 1

        physical_actions = {}
        communication_actions = {}

        for agent in current_agents:

            action = actions[agent]

            physical_actions[agent] = int(
                action[0]
            )

            communication_actions[agent] = int(
                action[1]
            )

        # ----------------------------------------------------
        # Physical transition
        # ----------------------------------------------------

        self._resolve_movement(
            physical_actions
        )

        # ----------------------------------------------------
        # Communication
        #
        # action at t
        # -> included in observation t+1
        # ----------------------------------------------------

        self._update_messages(
            communication_actions
        )

        success = self._check_success()

        # ----------------------------------------------------
        # Team reward
        # ----------------------------------------------------

        if success:
            reward = 100.0
        else:
            reward = -1.0

        rewards = {
            agent: reward
            for agent in current_agents
        }

        terminations = {
            agent: success
            for agent in current_agents
        }

        timeout = (
            self.step_count >= self.max_steps
        )

        truncations = {
            agent: (
                timeout
                and not success
            )
            for agent in current_agents
        }

        infos = {
            agent: {
                "success": success,
                "step_count": self.step_count,
            }
            for agent in current_agents
        }

        observations = {
            agent: self._get_observation(agent)
            for agent in current_agents
        }

        if self.render_mode == "human":
            self.render()

        if success or timeout:
            self.agents = []

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

    # ========================================================
    # Movement resolution
    # ========================================================

    def _resolve_movement(
        self,
        actions: Dict[str, int],
    ):

        # ====================================================
        # Current occupancy
        # ====================================================

        agent_at = {
            position: agent
            for agent, position
            in self.agent_positions.items()
        }

        # ====================================================
        # Detect possible pushes
        # ====================================================

        push_candidates = {}

        push_agents = {}

        for obj in self.objects:

            possible_pushes = []

            for direction in (
                UP,
                RIGHT,
                DOWN,
                LEFT,
            ):

                contact_cells = (
                    obj.contact_cells(direction)
                )

                participants = []

                valid = True

                # 한쪽 접촉면 전체에 agent가 있어야 한다.
                for cell in contact_cells:

                    if cell not in agent_at:
                        valid = False
                        break

                    agent = agent_at[cell]

                    # 붙어 있는 것만으로 push가 아니다.
                    # 실제 object 방향으로 움직여야 한다.
                    if actions[agent] != direction:
                        valid = False
                        break

                    participants.append(agent)

                if valid:
                    possible_pushes.append(
                        (
                            direction,
                            participants,
                        )
                    )

            # 동시에 서로 다른 방향의 완전한 push가
            # 성립하면 모호하므로 모두 취소.
            if len(possible_pushes) == 1:

                direction, participants = (
                    possible_pushes[0]
                )

                push_candidates[
                    obj.object_id
                ] = direction

                push_agents[
                    obj.object_id
                ] = set(participants)

        valid_pushes = dict(
            push_candidates
        )

        # ====================================================
        # Push cannot leave map / hit wall
        # ====================================================

        for obj in self.objects:

            oid = obj.object_id

            if oid not in valid_pushes:
                continue

            direction = valid_pushes[oid]

            destination = obj.moved_cells(
                direction
            )

            if any(
                (
                    not self._inside(cell)
                    or cell in self.walls
                )
                for cell in destination
            ):
                valid_pushes.pop(
                    oid,
                    None,
                )

        # ====================================================
        # Object-object conflict resolution
        #
        # Object cannot push another object.
        # ====================================================

        self._remove_object_conflicts(
            valid_pushes
        )

        # ====================================================
        # Agent/object movement dependencies
        # ====================================================

        while True:

            final_object_cells = (
                self._calculate_final_object_cells(
                    valid_pushes
                )
            )

            occupied_by_final_objects = set()

            for cells in (
                final_object_cells.values()
            ):
                occupied_by_final_objects |= cells

            raw_agent_targets = {}

            for agent in self.agents:

                current = (
                    self.agent_positions[agent]
                )

                direction = actions[agent]

                dx, dy = (
                    ACTION_TO_DELTA[
                        direction
                    ]
                )

                target = (
                    current[0] + dx,
                    current[1] + dy,
                )

                if (
                    not self._inside(target)
                    or target in self.walls
                ):
                    target = current

                raw_agent_targets[
                    agent
                ] = target

            final_agent_positions = (
                self._resolve_agent_conflicts(
                    raw_targets=raw_agent_targets,
                    final_object_cells=(
                        occupied_by_final_objects
                    ),
                )
            )

            # Push를 수행하는 agent가 실제로 object가
            # 비운 cell로 이동하지 못한다면 push도 취소.
            pushes_to_cancel = set()

            for obj in self.objects:

                oid = obj.object_id

                if oid not in valid_pushes:
                    continue

                for agent in push_agents[oid]:

                    if (
                        final_agent_positions[
                            agent
                        ]
                        ==
                        self.agent_positions[
                            agent
                        ]
                    ):
                        pushes_to_cancel.add(
                            oid
                        )
                        break

            if not pushes_to_cancel:
                break

            for oid in pushes_to_cancel:
                valid_pushes.pop(
                    oid,
                    None,
                )

            self._remove_object_conflicts(
                valid_pushes
            )

        # ====================================================
        # Commit simultaneously
        # ====================================================

        for obj in self.objects:

            oid = obj.object_id

            if oid in valid_pushes:

                obj.move(
                    valid_pushes[oid]
                )

        self.agent_positions = (
            final_agent_positions
        )

    # ========================================================
    # Object collision resolution
    # ========================================================

    def _remove_object_conflicts(
        self,
        valid_pushes,
    ):

        changed = True

        while changed:

            changed = False

            final_cells = (
                self._calculate_final_object_cells(
                    valid_pushes
                )
            )

            for i in range(
                len(self.objects)
            ):

                for j in range(
                    i + 1,
                    len(self.objects),
                ):

                    a = self.objects[i]
                    b = self.objects[j]

                    if (
                        final_cells[a.object_id]
                        &
                        final_cells[b.object_id]
                    ):

                        a_moving = (
                            a.object_id
                            in valid_pushes
                        )

                        b_moving = (
                            b.object_id
                            in valid_pushes
                        )

                        if a_moving:
                            valid_pushes.pop(
                                a.object_id,
                                None,
                            )

                        if b_moving:
                            valid_pushes.pop(
                                b.object_id,
                                None,
                            )

                        if a_moving or b_moving:
                            changed = True
                            break

                if changed:
                    break

    def _calculate_final_object_cells(
        self,
        valid_pushes,
    ):

        result = {}

        for obj in self.objects:

            if obj.object_id in valid_pushes:

                result[obj.object_id] = (
                    obj.moved_cells(
                        valid_pushes[
                            obj.object_id
                        ]
                    )
                )

            else:
                result[obj.object_id] = (
                    obj.cells()
                )

        return result

    # ========================================================
    # Agent conflict resolution
    # ========================================================

    def _resolve_agent_conflicts(
        self,
        raw_targets,
        final_object_cells,
    ):

        current = dict(
            self.agent_positions
        )

        moving = {
            agent: (
                raw_targets[agent]
                != current[agent]
            )
            for agent in self.agents
        }

        while True:

            changed = False

            proposed = {
                agent: (
                    raw_targets[agent]
                    if moving[agent]
                    else current[agent]
                )
                for agent in self.agents
            }

            # -----------------------------------------------
            # Agent-object collision
            # -----------------------------------------------

            for agent in self.agents:

                if (
                    moving[agent]
                    and proposed[agent]
                    in final_object_cells
                ):

                    moving[agent] = False
                    changed = True

            if changed:
                continue

            # -----------------------------------------------
            # Two or more agents end in same cell
            # -----------------------------------------------

            occupancy = {}

            for agent, position in (
                proposed.items()
            ):

                occupancy.setdefault(
                    position,
                    [],
                ).append(agent)

            for agents in occupancy.values():

                if len(agents) <= 1:
                    continue

                for agent in agents:

                    if moving[agent]:

                        moving[agent] = False
                        changed = True

            if changed:
                continue

            # -----------------------------------------------
            # Direct swaps prohibited
            #
            # A -> B
            # B -> A
            # -----------------------------------------------

            for i in range(
                len(self.agents)
            ):

                for j in range(
                    i + 1,
                    len(self.agents),
                ):

                    a = self.agents[i]
                    b = self.agents[j]

                    if not (
                        moving[a]
                        and moving[b]
                    ):
                        continue

                    if (
                        raw_targets[a]
                        == current[b]
                        and raw_targets[b]
                        == current[a]
                    ):

                        moving[a] = False
                        moving[b] = False

                        changed = True

            if not changed:
                break

        return {
            agent: (
                raw_targets[agent]
                if moving[agent]
                else current[agent]
            )
            for agent in self.agents
        }

    # ========================================================
    # Communication
    # ========================================================

    def _update_messages(
        self,
        communication_actions,
    ):

        next_messages = {
            agent: []
            for agent in self.agents
        }

        for sender in self.agents:

            if (
                communication_actions[
                    sender
                ]
                != HELP
            ):
                continue

            sx, sy = (
                self.agent_positions[
                    sender
                ]
            )

            sender_id = int(
                sender.split("_")[1]
            )

            # HELP is broadcast
            for receiver in self.agents:

                if receiver == sender:
                    continue

                rx, ry = (
                    self.agent_positions[
                        receiver
                    ]
                )

                relative_dx = sx - rx
                relative_dy = sy - ry

                next_messages[
                    receiver
                ].append(
                    (
                        sender_id,
                        relative_dx,
                        relative_dy,
                    )
                )

        self.received_messages = (
            next_messages
        )

    # ========================================================
    # Observation
    # ========================================================

    def _get_observation(
        self,
        agent,
    ):

        ax, ay = (
            self.agent_positions[agent]
        )

        local_map = np.zeros(
            (
                self.vision_size,
                self.vision_size,
            ),
            dtype=np.int8,
        )

        object_ids = np.zeros(
            (
                self.vision_size,
                self.vision_size,
            ),
            dtype=np.int16,
        )

        goal_mask = np.zeros(
            (
                self.vision_size,
                self.vision_size,
            ),
            dtype=np.int8,
        )

        object_by_cell = {}

        for obj in self.objects:

            for cell in obj.cells():

                object_by_cell[
                    cell
                ] = obj

        other_agents = {
            position
            for name, position
            in self.agent_positions.items()
            if name != agent
        }

        # ====================================================
        # 5x5 local view
        # ====================================================

        for local_y in range(
            self.vision_size
        ):

            for local_x in range(
                self.vision_size
            ):

                world_x = (
                    ax
                    + local_x
                    - self.vision_radius
                )

                world_y = (
                    ay
                    + local_y
                    - self.vision_radius
                )

                position = (
                    world_x,
                    world_y,
                )

                if not self._inside(position):

                    local_map[
                        local_y,
                        local_x,
                    ] = WALL

                    continue

                if position in self.walls:

                    local_map[
                        local_y,
                        local_x,
                    ] = WALL

                    continue

                if position in self.goal_cells:

                    local_map[
                        local_y,
                        local_x,
                    ] = GOAL

                    goal_mask[
                        local_y,
                        local_x,
                    ] = 1

                if position in object_by_cell:

                    obj = (
                        object_by_cell[
                            position
                        ]
                    )

                    if obj.is_target:
                        code = TARGET_OBJECT
                    else:
                        code = MOVABLE_OBJECT

                    local_map[
                        local_y,
                        local_x,
                    ] = code

                    object_ids[
                        local_y,
                        local_x,
                    ] = (
                        obj.object_id + 1
                    )

                if position in other_agents:

                    local_map[
                        local_y,
                        local_x,
                    ] = OTHER_AGENT

        # ====================================================
        # Object size knowledge after contact
        # ====================================================

        touching_objects = np.zeros(
            (
                self.max_objects,
                4,
            ),
            dtype=np.int16,
        )

        for obj in self.objects:

            if (
                obj.object_id
                >= self.max_objects
            ):
                continue

            if self._agent_touching_object(
                agent,
                obj,
            ):

                touching_objects[
                    obj.object_id
                ] = np.array(
                    [
                        1,
                        obj.width,
                        obj.height,
                        int(obj.is_target),
                    ],
                    dtype=np.int16,
                )

        # ====================================================
        # Communication observation
        # ====================================================

        messages = np.zeros(
            (
                self.num_agents - 1,
                4,
            ),
            dtype=np.int16,
        )

        received = (
            self.received_messages[
                agent
            ]
        )

        for index, (
            sender_id,
            dx,
            dy,
        ) in enumerate(
            received[
                : self.num_agents - 1
            ]
        ):

            messages[index] = np.array(
                [
                    1,
                    dx,
                    dy,
                    sender_id,
                ],
                dtype=np.int16,
            )

        return {
            "local_map": local_map,
            "object_ids": object_ids,
            "goal_mask": goal_mask,

            "position": np.array(
                [ax, ay],
                dtype=np.int16,
            ),

            "touching_objects":
                touching_objects,

            "messages":
                messages,
        }

    # ========================================================
    # Helpers
    # ========================================================

    def _inside(
        self,
        position: Position,
    ):

        x, y = position

        return (
            0 <= x < self.width
            and
            0 <= y < self.height
        )

    def _agent_touching_object(
        self,
        agent,
        obj,
    ):

        ax, ay = (
            self.agent_positions[
                agent
            ]
        )

        # Manhattan adjacency only.
        # Diagonal contact is not contact.

        for ox, oy in obj.cells():

            if (
                abs(ax - ox)
                +
                abs(ay - oy)
                == 1
            ):
                return True

        return False

    # ========================================================
    # Success
    # ========================================================

    def _check_success(self):

        target_objects = [
            obj
            for obj in self.objects
            if obj.is_target
        ]

        if len(target_objects) != 1:
            raise RuntimeError(
                "Exactly one target object is required."
            )

        target = target_objects[0]

        return (
            target.cells()
            ==
            self.goal_cells
        )

    # ========================================================
    # Global state
    #
    # MAPPO centralized critic 등에서 나중에 사용 가능.
    # ========================================================

    def state(self):

        grid = np.zeros(
            (
                self.height,
                self.width,
            ),
            dtype=np.int16,
        )

        for x, y in self.goal_cells:
            grid[y, x] = GOAL

        for x, y in self.walls:
            grid[y, x] = WALL

        for obj in self.objects:

            value = (
                TARGET_OBJECT
                if obj.is_target
                else MOVABLE_OBJECT
            )

            for x, y in obj.cells():
                grid[y, x] = value

        for index, agent in enumerate(
            self.possible_agents
        ):

            if agent not in self.agent_positions:
                continue

            x, y = (
                self.agent_positions[
                    agent
                ]
            )

            # agent identity 보존을 위해 10 이상 사용
            grid[y, x] = 10 + index

        return grid.flatten()

    # ========================================================
    # Render
    # ========================================================

    def render(self):

        grid = [
            [
                "."
                for _ in range(self.width)
            ]
            for _ in range(self.height)
        ]

        for x, y in self.goal_cells:
            grid[y][x] = "G"

        for x, y in self.walls:
            grid[y][x] = "#"

        for obj in self.objects:

            symbol = (
                "T"
                if obj.is_target
                else "O"
            )

            for x, y in obj.cells():
                grid[y][x] = symbol

        for agent, (
            x,
            y,
        ) in self.agent_positions.items():

            number = agent.split("_")[1]

            grid[y][x] = number

        result = (
            f"\nStep {self.step_count}\n"
            +
            "\n".join(
                " ".join(row)
                for row in grid
            )
            +
            "\n"
        )

        if self.render_mode == "ansi":
            return result

        print(result)

    def close(self):
        pass