from __future__ import annotations

from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

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
    HELP,
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
        "name": "cooperative_transport_v1",
        "render_modes": [
            "human",
            "ansi",
        ],
        "is_parallelizable": True,
    }
    @property
    def num_agents(self):
        return self._num_agents
    @num_agents.setter
    def num_agents(self,num):
        self._num_agents=num
        return num

    def __init__(
        self,
        width=12,
        height=12,
        num_agents=4,
        vision_size=5,
        max_steps=200,
        curriculum_stage=0,
        step_penalty=-0.01,
        distance_reward_coef=0.2,
        success_reward=10.0,
        render_mode=None,
    ):
        super().__init__()

        if vision_size % 2 == 0:
            raise ValueError(
                "vision_size must be odd."
            )

        self.width = width
        self.height = height

        self._num_agents = num_agents

        self.vision_size = vision_size
        self.vision_radius = (
            vision_size // 2
        )

        self.max_steps = max_steps

        self.render_mode = render_mode

        self.curriculum_stage = (
            curriculum_stage
        )

        self.step_penalty = step_penalty

        self.distance_reward_coef = (
            distance_reward_coef
        )

        self.success_reward = (
            success_reward
        )

        self.possible_agents = [
            f"agent_{i}"
            for i in range(num_agents)
        ]

        self.agents = []

        self.agent_positions = {}

        self.objects: List[
            RectObject
        ] = []

        self.goal_cells = set()

        self.walls = set()

        self.received_messages = {}

        self.step_count = 0

        self.rng = (
            np.random.default_rng()
        )

        self.max_objects = 10

        self.action_spaces = {
            agent:
                spaces.MultiDiscrete(
                    [5, 2]
                )
            for agent
            in self.possible_agents
        }

        self.observation_spaces = {}

        for agent in self.possible_agents:

            self.observation_spaces[
                agent
            ] = spaces.Dict(
                {
                    "local_map":
                        spaces.Box(
                            low=0,
                            high=5,
                            shape=(
                                vision_size,
                                vision_size,
                            ),
                            dtype=np.int8,
                        ),

                    "object_ids":
                        spaces.Box(
                            low=0,
                            high=self.max_objects,
                            shape=(
                                vision_size,
                                vision_size,
                            ),
                            dtype=np.int16,
                        ),

                    "goal_mask":
                        spaces.Box(
                            low=0,
                            high=1,
                            shape=(
                                vision_size,
                                vision_size,
                            ),
                            dtype=np.int8,
                        ),

                    "position":
                        spaces.Box(
                            low=np.array(
                                [0, 0],
                                dtype=np.int16,
                            ),
                            high=np.array(
                                [
                                    width - 1,
                                    height - 1,
                                ],
                                dtype=np.int16,
                            ),
                            dtype=np.int16,
                        ),

                    "touching_objects":
                        spaces.Box(
                            low=0,
                            high=max(
                                width,
                                height,
                            ),
                            shape=(
                                self.max_objects,
                                4,
                            ),
                            dtype=np.int16,
                        ),

                    "messages":
                        spaces.Box(
                            low=-max(
                                width,
                                height,
                            ),
                            high=max(
                                width,
                                height,
                            ),
                            shape=(
                                num_agents - 1,
                                4,
                            ),
                            dtype=np.int16,
                        ),
                }
            )

    def action_space(
        self,
        agent,
    ):
        return self.action_spaces[
            agent
        ]

    def observation_space(
        self,
        agent,
    ):
        return self.observation_spaces[
            agent
        ]

    # ========================================================
    # Curriculum
    # ========================================================

    def set_curriculum_stage(
        self,
        stage: int,
    ):
        self.curriculum_stage = max(
            0,
            min(4, stage),
        )

    def _configure_stage(self):
        """
        Stage 0:
            push itself

        Stage 1:
            agents must approach + align

        Stage 2:
            larger target requiring 3 agents

        Stage 3:
            movable obstacle

        Stage 4:
            full task
        """

        stage = self.curriculum_stage

        self.walls = set()

        # ---------------------------------------------
        # Stage 0
        # ---------------------------------------------

        if stage == 0:

            target = RectObject(
                object_id=0,
                x=5,
                y=4,
                width=2,
                height=1,
                is_target=True,
            )

            self.objects = [target]

            self.goal_cells = {
                (5, 7),
                (6, 7),
            }

            self.agent_positions = {
                "agent_0": (5, 3),
                "agent_1": (6, 3),
                "agent_2": (2, 2),
                "agent_3": (9, 2),
            }

            return

        # ---------------------------------------------
        # Stage 1
        # ---------------------------------------------

        if stage == 1:

            target = RectObject(
                object_id=0,
                x=5,
                y=4,
                width=2,
                height=1,
                is_target=True,
            )

            self.objects = [target]

            self.goal_cells = {
                (5, 8),
                (6, 8),
            }

            candidate_positions = [
                (3, 2),
                (4, 2),
                (5, 2),
                (6, 2),
                (7, 2),
                (8, 2),

                (3, 3),
                (4, 3),
                (7, 3),
                (8, 3),
            ]

            selected = self.rng.choice(
                len(candidate_positions),
                size=self.num_agents,
                replace=False,
            )

            self.agent_positions = {
                agent:
                    candidate_positions[idx]
                for agent, idx
                in zip(
                    self.possible_agents,
                    selected,
                )
            }

            return

        # ---------------------------------------------
        # Stage 2
        # ---------------------------------------------

        if stage == 2:

            target = RectObject(
                object_id=0,
                x=4,
                y=4,
                width=3,
                height=2,
                is_target=True,
            )

            self.objects = [target]

            self.goal_cells = {
                (x, y)
                for y in range(8, 10)
                for x in range(4, 7)
            }

            candidate_positions = [
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 1),
                (6, 1),
                (7, 1),
                (8, 1),

                (2, 2),
                (3, 2),
                (4, 2),
                (5, 2),
                (6, 2),
                (7, 2),
                (8, 2),

                (2, 3),
                (3, 3),
                (7, 3),
                (8, 3),
            ]

            selected = self.rng.choice(
                len(candidate_positions),
                size=self.num_agents,
                replace=False,
            )

            self.agent_positions = {
                agent:
                    candidate_positions[idx]
                for agent, idx
                in zip(
                    self.possible_agents,
                    selected,
                )
            }

            return

        # ---------------------------------------------
        # Stage 3
        # ---------------------------------------------

        if stage == 3:

            target = RectObject(
                object_id=0,
                x=4,
                y=2,
                width=3,
                height=2,
                is_target=True,
            )

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

            self.goal_cells = {
                (x, y)
                for y in range(9, 11)
                for x in range(4, 7)
            }

            self._randomize_agents()

            return

        # ---------------------------------------------
        # Stage 4
        # ---------------------------------------------

        target = RectObject(
            object_id=0,
            x=4,
            y=2,
            width=3,
            height=2,
            is_target=True,
        )

        obstacle_1 = RectObject(
            object_id=1,
            x=4,
            y=6,
            width=3,
            height=1,
            is_target=False,
        )

        obstacle_2 = RectObject(
            object_id=2,
            x=8,
            y=5,
            width=1,
            height=2,
            is_target=False,
        )

        self.objects = [
            target,
            obstacle_1,
            obstacle_2,
        ]

        self.goal_cells = {
            (x, y)
            for y in range(9, 11)
            for x in range(4, 7)
        }

        self._randomize_agents()

    def _randomize_agents(self):

        forbidden = set()

        for obj in self.objects:
            forbidden |= obj.cells()

        forbidden |= self.walls

        available = [
            (x, y)
            for y in range(
                self.height
            )
            for x in range(
                self.width
            )
            if (x, y)
            not in forbidden
        ]

        selected = self.rng.choice(
            len(available),
            size=self.num_agents,
            replace=False,
        )

        self.agent_positions = {
            agent: available[index]
            for agent, index
            in zip(
                self.possible_agents,
                selected,
            )
        }

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        if seed is not None:
            self.rng = (
                np.random.default_rng(
                    seed
                )
            )

        self.agents = (
            self.possible_agents[:]
        )

        self.step_count = 0

        self._configure_stage()

        self.received_messages = {
            agent: []
            for agent in self.agents
        }

        observations = {
            agent:
                self._get_observation(
                    agent
                )
            for agent in self.agents
        }

        infos = {
            agent: {
                "stage":
                    self.curriculum_stage
            }
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

        current_agents = (
            self.agents[:]
        )

        self.step_count += 1

        physical_actions = {}
        communication_actions = {}

        for agent in current_agents:

            physical_actions[
                agent
            ] = int(
                actions[agent][0]
            )

            communication_actions[
                agent
            ] = int(
                actions[agent][1]
            )

        distance_before = (
            self._target_goal_distance()
        )

        self._resolve_movement(
            physical_actions
        )

        distance_after = (
            self._target_goal_distance()
        )

        self._update_messages(
            communication_actions
        )

        success = (
            self._check_success()
        )

        distance_improvement = (
            distance_before
            - distance_after
        )

        reward = (
            self.step_penalty
            + self.distance_reward_coef
            * distance_improvement
        )

        if success:
            reward += (
                self.success_reward
            )

        rewards = {
            agent: reward
            for agent in current_agents
        }

        terminations = {
            agent: success
            for agent in current_agents
        }

        timeout = (
            self.step_count
            >= self.max_steps
        )

        truncations = {
            agent:
                timeout
                and not success
            for agent in current_agents
        }

        infos = {
            agent: {
                "success": success,
                "step_count":
                    self.step_count,
                "stage":
                    self.curriculum_stage,
                "distance":
                    distance_after,
                "distance_improvement":
                    distance_improvement,
            }
            for agent in current_agents
        }

        observations = {
            agent:
                self._get_observation(
                    agent
                )
            for agent
            in current_agents
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
    # Reward helper
    # ========================================================

    def _target_goal_distance(self):

        target = next(
            obj
            for obj in self.objects
            if obj.is_target
        )

        goal_x = min(
            x
            for x, _
            in self.goal_cells
        )

        goal_y = min(
            y
            for _, y
            in self.goal_cells
        )

        return (
            abs(target.x - goal_x)
            +
            abs(target.y - goal_y)
        )

    # ========================================================
    # Movement
    # ========================================================

    def _resolve_movement(
        self,
        actions: Dict[str, int],
    ):

        agent_at = {
            position: agent
            for agent, position
            in self.agent_positions.items()
        }

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
                    obj.contact_cells(
                        direction
                    )
                )

                participants = []

                valid = True

                for cell in (
                    contact_cells
                ):

                    if cell not in agent_at:
                        valid = False
                        break

                    agent = (
                        agent_at[cell]
                    )

                    if (
                        actions[agent]
                        != direction
                    ):
                        valid = False
                        break

                    participants.append(
                        agent
                    )

                if valid:
                    possible_pushes.append(
                        (
                            direction,
                            participants,
                        )
                    )

            if len(
                possible_pushes
            ) == 1:

                direction, participants = (
                    possible_pushes[0]
                )

                push_candidates[
                    obj.object_id
                ] = direction

                push_agents[
                    obj.object_id
                ] = set(
                    participants
                )

        valid_pushes = dict(
            push_candidates
        )

        # Map / wall validation
        for obj in self.objects:

            oid = obj.object_id

            if oid not in valid_pushes:
                continue

            destination = (
                obj.moved_cells(
                    valid_pushes[oid]
                )
            )

            if any(
                (
                    not self._inside(
                        cell
                    )
                    or cell in self.walls
                )
                for cell in destination
            ):
                valid_pushes.pop(
                    oid,
                    None,
                )

        # Object cannot push object
        self._remove_object_conflicts(
            valid_pushes
        )

        while True:

            final_object_cells = (
                self
                ._calculate_final_object_cells(
                    valid_pushes
                )
            )

            occupied = set()

            for cells in (
                final_object_cells
                .values()
            ):
                occupied |= cells

            raw_targets = {}

            for agent in self.agents:

                current = (
                    self.agent_positions[
                        agent
                    ]
                )

                dx, dy = (
                    ACTION_TO_DELTA[
                        actions[agent]
                    ]
                )

                target = (
                    current[0] + dx,
                    current[1] + dy,
                )

                if (
                    not self._inside(
                        target
                    )
                    or target in self.walls
                ):
                    target = current

                raw_targets[
                    agent
                ] = target

            final_agent_positions = (
                self._resolve_agent_conflicts(
                    raw_targets,
                    occupied,
                )
            )

            pushes_to_cancel = set()

            for obj in self.objects:

                oid = obj.object_id

                if oid not in valid_pushes:
                    continue

                for agent in (
                    push_agents[oid]
                ):

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

            for oid in (
                pushes_to_cancel
            ):
                valid_pushes.pop(
                    oid,
                    None,
                )

            self._remove_object_conflicts(
                valid_pushes
            )

        # Commit objects
        for obj in self.objects:

            if (
                obj.object_id
                in valid_pushes
            ):
                obj.move(
                    valid_pushes[
                        obj.object_id
                    ]
                )

        # Commit agents
        self.agent_positions = (
            final_agent_positions
        )

    def _calculate_final_object_cells(
        self,
        valid_pushes,
    ):

        result = {}

        for obj in self.objects:

            if (
                obj.object_id
                in valid_pushes
            ):

                result[
                    obj.object_id
                ] = obj.moved_cells(
                    valid_pushes[
                        obj.object_id
                    ]
                )

            else:

                result[
                    obj.object_id
                ] = obj.cells()

        return result

    def _remove_object_conflicts(
        self,
        valid_pushes,
    ):

        changed = True

        while changed:

            changed = False

            final_cells = (
                self
                ._calculate_final_object_cells(
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

                    overlap = (
                        final_cells[
                            a.object_id
                        ]
                        &
                        final_cells[
                            b.object_id
                        ]
                    )

                    if not overlap:
                        continue

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

                    if (
                        a_moving
                        or b_moving
                    ):
                        changed = True
                        break

                if changed:
                    break

    def _resolve_agent_conflicts(
        self,
        raw_targets,
        final_object_cells,
    ):

        current = dict(
            self.agent_positions
        )

        moving = {
            agent:
                raw_targets[agent]
                != current[agent]
            for agent in self.agents
        }

        while True:

            changed = False

            proposed = {
                agent:
                    (
                        raw_targets[
                            agent
                        ]
                        if moving[agent]
                        else current[
                            agent
                        ]
                    )
                for agent in self.agents
            }

            # Agent-object collision
            for agent in self.agents:

                if (
                    moving[agent]
                    and proposed[
                        agent
                    ]
                    in final_object_cells
                ):
                    moving[
                        agent
                    ] = False

                    changed = True

            if changed:
                continue

            # Agent-agent same final cell
            occupancy = {}

            for agent, pos in (
                proposed.items()
            ):

                occupancy.setdefault(
                    pos,
                    [],
                ).append(agent)

            for agents in (
                occupancy.values()
            ):

                if len(agents) <= 1:
                    continue

                for agent in agents:

                    if moving[agent]:

                        moving[
                            agent
                        ] = False

                        changed = True

            if changed:
                continue

            # Direct swap forbidden
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
            agent:
                (
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

            for receiver in self.agents:

                if (
                    receiver
                    == sender
                ):
                    continue

                rx, ry = (
                    self.agent_positions[
                        receiver
                    ]
                )

                next_messages[
                    receiver
                ].append(
                    (
                        sender_id,
                        sx - rx,
                        sy - ry,
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
            self.agent_positions[
                agent
            ]
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

        for ly in range(
            self.vision_size
        ):

            for lx in range(
                self.vision_size
            ):

                wx = (
                    ax
                    + lx
                    - self.vision_radius
                )

                wy = (
                    ay
                    + ly
                    - self.vision_radius
                )

                pos = (wx, wy)

                if not self._inside(
                    pos
                ):
                    local_map[
                        ly,
                        lx,
                    ] = WALL
                    continue

                if pos in self.walls:
                    local_map[
                        ly,
                        lx,
                    ] = WALL
                    continue

                if (
                    pos
                    in self.goal_cells
                ):

                    local_map[
                        ly,
                        lx,
                    ] = GOAL

                    goal_mask[
                        ly,
                        lx,
                    ] = 1

                if (
                    pos
                    in object_by_cell
                ):

                    obj = (
                        object_by_cell[
                            pos
                        ]
                    )

                    local_map[
                        ly,
                        lx,
                    ] = (
                        TARGET_OBJECT
                        if obj.is_target
                        else MOVABLE_OBJECT
                    )

                    object_ids[
                        ly,
                        lx,
                    ] = (
                        obj.object_id + 1
                    )

                if pos in other_agents:

                    local_map[
                        ly,
                        lx,
                    ] = OTHER_AGENT

        touching_objects = (
            np.zeros(
                (
                    self.max_objects,
                    4,
                ),
                dtype=np.int16,
            )
        )

        for obj in self.objects:

            if (
                obj.object_id
                >= self.max_objects
            ):
                continue

            if (
                self
                ._agent_touching_object(
                    agent,
                    obj,
                )
            ):

                touching_objects[
                    obj.object_id
                ] = np.array(
                    [
                        1,
                        obj.width,
                        obj.height,
                        int(
                            obj.is_target
                        ),
                    ],
                    dtype=np.int16,
                )

        messages = np.zeros(
            (
                self.num_agents - 1,
                4,
            ),
            dtype=np.int16,
        )

        for index, (
            sender_id,
            dx,
            dy,
        ) in enumerate(
            self.received_messages[
                agent
            ][
                :
                self.num_agents - 1
            ]
        ):

            messages[
                index
            ] = np.array(
                [
                    1,
                    dx,
                    dy,
                    sender_id,
                ],
                dtype=np.int16,
            )

        return {
            "local_map":
                local_map,

            "object_ids":
                object_ids,

            "goal_mask":
                goal_mask,

            "position":
                np.array(
                    [ax, ay],
                    dtype=np.int16,
                ),

            "touching_objects":
                touching_objects,

            "messages":
                messages,
        }

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

        for ox, oy in obj.cells():

            if (
                abs(ax - ox)
                + abs(ay - oy)
                == 1
            ):
                return True

        return False

    def _inside(
        self,
        pos,
    ):

        x, y = pos

        return (
            0 <= x < self.width
            and
            0 <= y < self.height
        )

    def _check_success(self):

        target = next(
            obj
            for obj in self.objects
            if obj.is_target
        )

        return (
            target.cells()
            == self.goal_cells
        )

    # ========================================================
    # Render
    # ========================================================

    def render(self):

        grid = [
            [
                "."
                for _ in range(
                    self.width
                )
            ]
            for _ in range(
                self.height
            )
        ]

        for x, y in (
            self.goal_cells
        ):
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
                grid[y][x] = (
                    symbol
                )

        for agent, (
            x,
            y,
        ) in (
            self.agent_positions
            .items()
        ):

            grid[y][x] = (
                agent.split("_")[1]
            )

        result = (
            f"\nStage "
            f"{self.curriculum_stage}"
            f" | Step "
            f"{self.step_count}\n"
            +
            "\n".join(
                " ".join(row)
                for row in grid
            )
            +
            "\n"
        )

        if (
            self.render_mode
            == "ansi"
        ):
            return result

        print(result)

    def close(self):
        pass