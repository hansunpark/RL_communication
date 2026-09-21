from __future__ import annotations

from typing import Dict, List, Tuple

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

from .object import RectObject


Position = Tuple[int, int]


class CooperativeTransportEnv(ParallelEnv):

    metadata = {
        "name": "cooperative_transport_v3",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": True,
    }
    @property
    def num_agents(self):
        return self._num_agents
    @num_agents.setter
    def num_agents(self, value):
        self._num_agents=value
        return value


    def __init__(
        self,
        width=12,
        height=12,
        num_agents=4,
        vision_size=5,
        max_steps=200,
        curriculum_stage=0,
        spawn_level=0,
        reward_mode="target_only",
        step_penalty=-0.01,
        distance_reward_coef=0.2,
        obstacle_reward_coef=0.1,
        success_reward=10.0,
        render_mode=None,
    ):
        super().__init__()

        if vision_size % 2 == 0:
            raise ValueError("vision_size must be odd.")

        self.width = width
        self.height = height
        self._num_agents = num_agents
        self.vision_size = vision_size
        self.vision_radius = vision_size // 2
        self.max_steps = max_steps

        self.render_mode = render_mode

        self.curriculum_stage = curriculum_stage
        self.spawn_level = spawn_level
        self.reward_mode = reward_mode

        self.step_penalty = step_penalty
        self.distance_reward_coef = distance_reward_coef
        self.obstacle_reward_coef = obstacle_reward_coef
        self.success_reward = success_reward

        self.possible_agents = [
            f"agent_{i}"
            for i in range(num_agents)
        ]

        self.agents = []

        self.agent_positions = {}

        self.objects: List[RectObject] = []

        self.goal_cells = set()
        self.walls = set()

        self.received_messages = {}

        self.step_count = 0

        self.rng = np.random.default_rng()

        self.max_objects = 10

        self.current_scenario_name = None

        self.action_spaces = {
            agent: spaces.MultiDiscrete([5, 2])
            for agent in self.possible_agents
        }

        self.observation_spaces = {}

        for agent in self.possible_agents:

            self.observation_spaces[agent] = spaces.Dict(
                {
                    "local_map": spaces.Box(
                        low=0,
                        high=5,
                        shape=(
                            vision_size,
                            vision_size,
                        ),
                        dtype=np.int8,
                    ),

                    "object_ids": spaces.Box(
                        low=0,
                        high=self.max_objects,
                        shape=(
                            vision_size,
                            vision_size,
                        ),
                        dtype=np.int16,
                    ),

                    "goal_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(
                            vision_size,
                            vision_size,
                        ),
                        dtype=np.int8,
                    ),

                    "position": spaces.Box(
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

                    "touching_objects": spaces.Box(
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

                    "messages": spaces.Box(
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

    # =====================================================
    # Spaces
    # =====================================================

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # =====================================================
    # Curriculum
    # =====================================================

    def set_curriculum(
        self,
        stage: int,
        spawn_level: int = 0,
        reward_mode: str = "target_only",
    ):
        self.curriculum_stage = max(
            0,
            min(6, stage),
        )

        self.spawn_level = max(
            0,
            min(2, spawn_level),
        )

        if reward_mode not in (
            "target_only",
            "obstacle_shaping",
        ):
            raise ValueError(
                "Unknown reward mode."
            )

        self.reward_mode = reward_mode

    # =====================================================
    # Reset
    # =====================================================

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

        self.current_scenario_name = None

        # ---------------------------------------------
        # Benchmark / custom scenario
        # ---------------------------------------------

        if (
            options is not None
            and
            "scenario" in options
        ):
            self._configure_custom_scenario(
                options["scenario"]
            )

        else:
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
                    self.curriculum_stage,

                "spawn_level":
                    self.spawn_level,

                "reward_mode":
                    self.reward_mode,

                "scenario":
                    self.current_scenario_name,
            }

            for agent in self.agents
        }

        if self.render_mode == "human":
            self.render()

        return observations, infos

    # =====================================================
    # Custom benchmark scenario
    # =====================================================

    def _configure_custom_scenario(
        self,
        scenario,
    ):
        """
        scenario must be a dictionary.

        Example:

        {
            "name": "left_obstacle",
            "target": {
                "x": 4,
                "y": 2,
                "width": 3,
                "height": 2,
            },
            "goal_cells": [(4,9), ...],
            "obstacles": [
                {
                    "x": 3,
                    "y": 6,
                    "width": 3,
                    "height": 1,
                }
            ],
            "walls": [],
            "random_agents": True,
            "agent_positions": None,
        }
        """

        self.current_scenario_name = (
            scenario.get(
                "name",
                "custom",
            )
        )

        target_cfg = (
            scenario["target"]
        )

        target = RectObject(
            object_id=0,
            x=int(
                target_cfg["x"]
            ),
            y=int(
                target_cfg["y"]
            ),
            width=int(
                target_cfg["width"]
            ),
            height=int(
                target_cfg["height"]
            ),
            is_target=True,
        )

        self.objects = [target]

        for i, obstacle_cfg in enumerate(
            scenario.get(
                "obstacles",
                [],
            ),
            start=1,
        ):
            obstacle = RectObject(
                object_id=i,
                x=int(
                    obstacle_cfg["x"]
                ),
                y=int(
                    obstacle_cfg["y"]
                ),
                width=int(
                    obstacle_cfg["width"]
                ),
                height=int(
                    obstacle_cfg["height"]
                ),
                is_target=False,
            )

            self.objects.append(
                obstacle
            )

        self.goal_cells = {
            tuple(cell)
            for cell in scenario[
                "goal_cells"
            ]
        }

        self.walls = {
            tuple(cell)
            for cell in scenario.get(
                "walls",
                [],
            )
        }

        if (
            "reward_mode"
            in scenario
        ):
            self.reward_mode = (
                scenario[
                    "reward_mode"
                ]
            )

        agent_positions = (
            scenario.get(
                "agent_positions"
            )
        )

        spawn_mode = (
            scenario.get(
                "spawn_mode"
            )
        )

        if agent_positions:

            self.agent_positions = {
                agent: tuple(
                    agent_positions[
                        agent
                    ]
                )

                for agent
                in self.possible_agents
            }

            self._validate_agent_positions()

        elif (
            spawn_mode is not None
            and
            spawn_mode.get("type")
            == "split"
        ):
            self._spawn_split(
                near_count=
                    spawn_mode.get(
                        "near_count",
                        1,
                    ),

                near_radius=
                    spawn_mode.get(
                        "near_radius",
                        self.vision_radius,
                    ),

                far_min_distance=
                    spawn_mode.get(
                        "far_min_distance",
                        6,
                    ),
            )

        else:
            self._randomize_agents()

    def _validate_agent_positions(
        self,
    ):
        positions = list(
            self.agent_positions.values()
        )

        if (
            len(set(positions))
            != len(positions)
        ):
            raise ValueError(
                "Agent positions overlap."
            )

        forbidden = (
            self._forbidden_cells()
        )

        for position in positions:

            if not self._inside(position):
                raise ValueError(
                    f"Agent outside map: "
                    f"{position}"
                )

            if position in forbidden:
                raise ValueError(
                    f"Agent spawned on "
                    f"occupied cell: "
                    f"{position}"
                )

    # =====================================================
    # Curriculum stage configuration
    # =====================================================

    def _configure_stage(self):

        self.current_scenario_name = (
            f"stage_{self.curriculum_stage}"
        )

        self.walls = set()

        stage = (
            self.curriculum_stage
        )

        # Stage 0
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

            fixed = [
                (5, 3),
                (6, 3),
                (1, 1),
                (10, 1),
            ]

            self.agent_positions = {
                agent: fixed[i]

                for i, agent
                in enumerate(
                    self.possible_agents
                )
            }

            return

        # Stage 1
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

            self._spawn_agents_near(
                target.cells(),
                radius=4,
            )

            return

        # Stage 2
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

                for y in range(
                    8,
                    10,
                )

                for x in range(
                    4,
                    7,
                )
            }

            self._spawn_agents_near(
                target.cells(),
                radius=4,
            )

            return

        # Stage 3
        if stage == 3:

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

                for y in range(
                    8,
                    10,
                )

                for x in range(
                    4,
                    7,
                )
            }

            if self.spawn_level == 0:

                self._spawn_agents_near(
                    target.cells(),
                    radius=3,
                )

            elif self.spawn_level == 1:

                self._spawn_agents_near(
                    target.cells(),
                    radius=5,
                )

            else:
                self._randomize_agents()

            return

        # Stage 4
        if stage == 4:

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

                for y in range(
                    9,
                    11,
                )

                for x in range(
                    4,
                    7,
                )
            }

            reference = (
                target.cells()
                |
                obstacle.cells()
            )

            self._spawn_agents_near(
                reference,
                radius=4,
            )

            return

        # Stage 5
        if stage == 5:

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

                for y in range(
                    9,
                    11,
                )

                for x in range(
                    4,
                    7,
                )
            }

            self._randomize_agents()

            return

        # Stage 6

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

            for y in range(
                9,
                11,
            )

            for x in range(
                4,
                7,
            )
        }

        self._randomize_agents()

    # =====================================================
    # Spawning
    # =====================================================

    def _spawn_agents_near(
        self,
        reference_cells,
        radius,
    ):
        forbidden = (
            self._forbidden_cells()
        )

        candidates = []

        for y in range(
            self.height
        ):
            for x in range(
                self.width
            ):
                pos = (x, y)

                if pos in forbidden:
                    continue

                distance = min(
                    abs(x - rx)
                    +
                    abs(y - ry)

                    for rx, ry
                    in reference_cells
                )

                if distance <= radius:
                    candidates.append(
                        pos
                    )

        if (
            len(candidates)
            < self.num_agents
        ):
            raise RuntimeError(
                "Not enough spawn cells."
            )

        selected = (
            self.rng.choice(
                len(candidates),
                size=self.num_agents,
                replace=False,
            )
        )

        self.agent_positions = {
            agent:
                candidates[index]

            for agent, index
            in zip(
                self.possible_agents,
                selected,
            )
        }

    def _randomize_agents(self):

        forbidden = (
            self._forbidden_cells()
        )

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

        if (
            len(available)
            < self.num_agents
        ):
            raise RuntimeError(
                "Not enough free cells."
            )

        selected = (
            self.rng.choice(
                len(available),
                size=self.num_agents,
                replace=False,
            )
        )

        self.agent_positions = {
            agent:
                available[index]

            for agent, index
            in zip(
                self.possible_agents,
                selected,
            )
        }

    def _spawn_split(
        self,
        near_count,
        near_radius,
        far_min_distance,
    ):
        """
        near_count agents spawn within near_radius of the target
        object; the remaining agents spawn at least
        far_min_distance away from it (Manhattan distance).

        Used to create information asymmetry: only the "near"
        agents can see the target/goal without moving, so the
        rest must rely on a HELP broadcast from a near agent to
        know where to go.
        """

        reference_cells = (
            self._target_object().cells()
        )

        forbidden = (
            self._forbidden_cells()
        )

        def distance_to_reference(pos):

            x, y = pos

            return min(
                abs(x - rx)
                + abs(y - ry)

                for rx, ry
                in reference_cells
            )

        all_cells = [
            (x, y)

            for y in range(self.height)
            for x in range(self.width)
        ]

        near_candidates = [
            pos
            for pos in all_cells

            if pos not in forbidden
            and distance_to_reference(pos)
            <= near_radius
        ]

        if (
            len(near_candidates)
            < near_count
        ):
            raise RuntimeError(
                "Not enough near "
                "spawn cells."
            )

        near_selected = (
            self.rng.choice(
                len(near_candidates),
                size=near_count,
                replace=False,
            )
        )

        near_positions = [
            near_candidates[i]
            for i in near_selected
        ]

        used = set(near_positions)

        far_count = (
            self.num_agents
            - near_count
        )

        far_candidates = [
            pos
            for pos in all_cells

            if pos not in forbidden
            and pos not in used
            and distance_to_reference(pos)
            >= far_min_distance
        ]

        if (
            len(far_candidates)
            < far_count
        ):
            raise RuntimeError(
                "Not enough far "
                "spawn cells. Reduce "
                "far_min_distance or "
                "near_count."
            )

        far_selected = (
            self.rng.choice(
                len(far_candidates),
                size=far_count,
                replace=False,
            )
        )

        far_positions = [
            far_candidates[i]
            for i in far_selected
        ]

        positions = (
            near_positions
            + far_positions
        )

        self.agent_positions = {
            agent: positions[i]

            for i, agent
            in enumerate(
                self.possible_agents
            )
        }

    def _forbidden_cells(self):

        forbidden = set(
            self.walls
        )

        for obj in self.objects:
            forbidden |= (
                obj.cells()
            )

        return forbidden

    # =====================================================
    # Step
    # =====================================================

    def step(self, actions):

        if not self.agents:
            return (
                {},
                {},
                {},
                {},
                {},
            )

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

        blocking_before = (
            self._obstacle_blocking_score()
        )

        movement_info = (
            self._resolve_movement(
                physical_actions
            )
        )

        distance_after = (
            self._target_goal_distance()
        )

        blocking_after = (
            self._obstacle_blocking_score()
        )

        self._update_messages(
            communication_actions
        )

        success = (
            self._check_success()
        )

        target_progress = (
            distance_before
            -
            distance_after
        )

        obstacle_progress = (
            blocking_before
            -
            blocking_after
        )

        reward = (
            self.step_penalty
            +
            self.distance_reward_coef
            * target_progress
        )

        if (
            self.reward_mode
            == "obstacle_shaping"
        ):
            reward += (
                self.obstacle_reward_coef
                * obstacle_progress
            )

        if success:
            reward += (
                self.success_reward
            )

        rewards = {
            agent: reward
            for agent
            in current_agents
        }

        terminations = {
            agent: success
            for agent
            in current_agents
        }

        timeout = (
            self.step_count
            >= self.max_steps
        )

        truncations = {
            agent:
                timeout
                and not success

            for agent
            in current_agents
        }

        infos = {
            agent: {
                "success":
                    success,

                "stage":
                    self.curriculum_stage,

                "scenario":
                    self.current_scenario_name,

                "step_count":
                    self.step_count,

                "target_distance":
                    distance_after,

                "target_progress":
                    target_progress,

                "obstacle_blocking":
                    blocking_after,

                "obstacle_progress":
                    obstacle_progress,

                "target_moved":
                    movement_info[
                        "target_moved"
                    ],

                "obstacle_moves":
                    movement_info[
                        "obstacle_moves"
                    ],

                "successful_pushes":
                    movement_info[
                        "successful_pushes"
                    ],
            }

            for agent
            in current_agents
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

    # =====================================================
    # Reward / diagnostics
    # =====================================================

    def _target_goal_distance(self):

        target = (
            self._target_object()
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
            abs(
                target.x
                -
                goal_x
            )
            +
            abs(
                target.y
                -
                goal_y
            )
        )

    def _obstacle_blocking_score(
        self,
    ):
        target = (
            self._target_object()
        )

        if len(self.objects) <= 1:
            return 0

        target_cells = (
            target.cells()
        )

        target_x_min = min(
            x
            for x, _
            in target_cells
        )

        target_x_max = max(
            x
            for x, _
            in target_cells
        )

        goal_x_min = min(
            x
            for x, _
            in self.goal_cells
        )

        goal_x_max = max(
            x
            for x, _
            in self.goal_cells
        )

        corridor_x_min = min(
            target_x_min,
            goal_x_min,
        )

        corridor_x_max = max(
            target_x_max,
            goal_x_max,
        )

        target_y_min = min(
            y
            for _, y
            in target_cells
        )

        target_y_max = max(
            y
            for _, y
            in target_cells
        )

        goal_y_min = min(
            y
            for _, y
            in self.goal_cells
        )

        goal_y_max = max(
            y
            for _, y
            in self.goal_cells
        )

        corridor_y_min = min(
            target_y_min,
            goal_y_min,
        )

        corridor_y_max = max(
            target_y_max,
            goal_y_max,
        )

        score = 0

        for obj in self.objects:

            if obj.is_target:
                continue

            for x, y in obj.cells():

                if (
                    corridor_x_min
                    <= x
                    <= corridor_x_max
                    and
                    corridor_y_min
                    <= y
                    <= corridor_y_max
                ):
                    score += 1

        return score

    def _target_object(self):

        return next(
            obj
            for obj in self.objects
            if obj.is_target
        )

    # =====================================================
    # Movement
    # =====================================================

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

                for cell in contact_cells:

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

        for obj in self.objects:

            oid = obj.object_id

            if oid not in valid_pushes:
                continue

            destination = (
                obj.moved_cells(
                    valid_pushes[oid]
                )
            )

            invalid = any(
                (
                    not self._inside(cell)
                    or
                    cell in self.walls
                )

                for cell
                in destination
            )

            if invalid:
                valid_pushes.pop(
                    oid,
                    None,
                )

        self._remove_object_conflicts(
            valid_pushes
        )

        while True:

            final_objects = (
                self._calculate_final_object_cells(
                    valid_pushes
                )
            )

            occupied = set()

            for cells in (
                final_objects.values()
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
                    or
                    target in self.walls
                ):
                    target = current

                raw_targets[
                    agent
                ] = target

            final_agents = (
                self._resolve_agent_conflicts(
                    raw_targets,
                    occupied,
                )
            )

            cancel_pushes = set()

            for obj in self.objects:

                oid = obj.object_id

                if oid not in valid_pushes:
                    continue

                for agent in (
                    push_agents[oid]
                ):

                    if (
                        final_agents[agent]
                        ==
                        self.agent_positions[
                            agent
                        ]
                    ):
                        cancel_pushes.add(
                            oid
                        )

                        break

            if not cancel_pushes:
                break

            for oid in cancel_pushes:
                valid_pushes.pop(
                    oid,
                    None,
                )

            self._remove_object_conflicts(
                valid_pushes
            )

        target_moved = False
        obstacle_moves = 0
        successful_pushes = 0

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

                successful_pushes += 1

                if obj.is_target:
                    target_moved = True
                else:
                    obstacle_moves += 1

        self.agent_positions = (
            final_agents
        )

        return {
            "target_moved":
                target_moved,

            "obstacle_moves":
                obstacle_moves,

            "successful_pushes":
                successful_pushes,
        }

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
                        or
                        b_moving
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

            for agent
            in self.agents
        }

        while True:

            changed = False

            proposed = {
                agent:
                    (
                        raw_targets[agent]
                        if moving[agent]
                        else
                        current[agent]
                    )

                for agent
                in self.agents
            }

            for agent in self.agents:

                if (
                    moving[agent]
                    and
                    proposed[agent]
                    in final_object_cells
                ):
                    moving[
                        agent
                    ] = False

                    changed = True

            if changed:
                continue

            occupancy = {}

            for agent, pos in (
                proposed.items()
            ):
                occupancy.setdefault(
                    pos,
                    [],
                ).append(
                    agent
                )

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
                        and
                        moving[b]
                    ):
                        continue

                    if (
                        raw_targets[a]
                        == current[b]
                        and
                        raw_targets[b]
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
                    else
                    current[agent]
                )

            for agent
            in self.agents
        }

    # =====================================================
    # Communication
    # =====================================================

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

                if receiver == sender:
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

    # =====================================================
    # Observation
    # =====================================================

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

                pos = (
                    wx,
                    wy,
                )

                if not self._inside(pos):

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

                if pos in self.goal_cells:

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
                        else
                        MOVABLE_OBJECT
                    )

                    object_ids[
                        ly,
                        lx,
                    ] = (
                        obj.object_id
                        + 1
                    )

                if pos in other_agents:

                    local_map[
                        ly,
                        lx,
                    ] = (
                        OTHER_AGENT
                    )

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
                self._agent_touching_object(
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

        received = (
            self.received_messages.get(
                agent,
                [],
            )
        )

        for index, (
            sender_id,
            dx,
            dy,
        ) in enumerate(
            received[
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
                +
                abs(ay - oy)
                == 1
            ):
                return True

        return False

    # =====================================================
    # Utility
    # =====================================================

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

        target = (
            self._target_object()
        )

        return (
            target.cells()
            ==
            self.goal_cells
        )

    def state(self):

        grid = np.zeros(
            (
                self.height,
                self.width,
            ),
            dtype=np.float32,
        )

        for x, y in self.walls:
            grid[y, x] = 1

        for x, y in self.goal_cells:
            grid[y, x] = 2

        for obj in self.objects:

            value = (
                4
                if obj.is_target
                else 3
            )

            for x, y in obj.cells():
                grid[y, x] = value

        for _, (
            x,
            y,
        ) in (
            self.agent_positions.items()
        ):
            grid[y, x] = 5

        return grid.flatten()

    # =====================================================
    # Render
    # =====================================================

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
        ) in (
            self.agent_positions.items()
        ):
            grid[y][x] = (
                agent.split("_")[1]
            )

        header = (
            self.current_scenario_name
            if self.current_scenario_name
            is not None
            else
            f"Stage "
            f"{self.curriculum_stage}"
        )

        result = (
            "\n"
            +
            str(header)
            +
            f" | Step "
            f"{self.step_count}"
            +
            "\n"
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