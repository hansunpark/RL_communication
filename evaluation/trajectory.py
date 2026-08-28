import json
import os

from copy import deepcopy


class TrajectoryRecorder:

    def __init__(
        self,
        scenario_name,
        seed,
        width,
        height,
    ):
        self.data = {
            "scenario":
                scenario_name,

            "seed":
                seed,

            "width":
                width,

            "height":
                height,

            "success":
                False,

            "episode_length":
                0,

            "return":
                0.0,

            "frames": [],
        }

    def record_initial(
        self,
        env,
    ):
        self.data[
            "frames"
        ].append(
            self._make_frame(
                env=env,
                step=0,
                actions=None,
                reward=0.0,
                info=None,
            )
        )

    def record_step(
        self,
        env,
        step,
        actions,
        reward,
        info,
    ):
        self.data[
            "frames"
        ].append(
            self._make_frame(
                env=env,
                step=step,
                actions=actions,
                reward=reward,
                info=info,
            )
        )

    def finish(
        self,
        success,
        episode_length,
        episode_return,
    ):
        self.data[
            "success"
        ] = bool(success)

        self.data[
            "episode_length"
        ] = int(
            episode_length
        )

        self.data[
            "return"
        ] = float(
            episode_return
        )

    def _make_frame(
        self,
        env,
        step,
        actions,
        reward,
        info,
    ):
        objects = []

        for obj in env.objects:

            objects.append(
                {
                    "id":
                        obj.object_id,

                    "x":
                        obj.x,

                    "y":
                        obj.y,

                    "width":
                        obj.width,

                    "height":
                        obj.height,

                    "is_target":
                        obj.is_target,
                }
            )

        formatted_actions = None

        if actions is not None:

            formatted_actions = {
                agent: [
                    int(
                        action[0]
                    ),
                    int(
                        action[1]
                    ),
                ]

                for agent, action
                in actions.items()
            }

        frame = {
            "step":
                int(step),

            "agents": {
                agent: [
                    int(pos[0]),
                    int(pos[1]),
                ]

                for agent, pos
                in env.agent_positions.items()
            },

            "objects":
                objects,

            "goal_cells": [
                [
                    int(x),
                    int(y),
                ]

                for x, y
                in sorted(
                    env.goal_cells
                )
            ],

            "walls": [
                [
                    int(x),
                    int(y),
                ]

                for x, y
                in sorted(
                    env.walls
                )
            ],

            "actions":
                formatted_actions,

            "reward":
                float(reward),
        }

        if info is not None:

            frame.update(
                {
                    "success":
                        bool(
                            info[
                                "success"
                            ]
                        ),

                    "target_distance":
                        int(
                            info[
                                "target_distance"
                            ]
                        ),

                    "obstacle_blocking":
                        int(
                            info[
                                "obstacle_blocking"
                            ]
                        ),

                    "target_moved":
                        bool(
                            info[
                                "target_moved"
                            ]
                        ),

                    "obstacle_moves":
                        int(
                            info[
                                "obstacle_moves"
                            ]
                        ),
                }
            )

        return frame

    def save(
        self,
        path,
    ):
        directory = os.path.dirname(
            path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.data,
                f,
                indent=2,
                ensure_ascii=False,
            )