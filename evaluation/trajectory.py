import json
import os


class TrajectoryRecorder:

    def __init__(
        self,
        scenario_name,
        seed,
        width,
        height,
        action_mode,
    ):
        self.scenario_name = (
            scenario_name
        )

        self.seed = seed

        self.width = width

        self.height = height

        self.action_mode = action_mode

        self.frames = []

        self.summary = {}

    def record_frame(
        self,
        env,
        actions,
        reward,
        info,
        action_probabilities=None,
    ):
        agents = {}

        agent_positions = getattr(
            env,
            "agent_positions",
            {},
        )

        for agent, pos in (
            agent_positions.items()
        ):
            agents[agent] = [
                int(pos[0]),
                int(pos[1]),
            ]

        objects = []

        for obj in getattr(
            env,
            "objects",
            [],
        ):
            objects.append(
                {
                    "id":
                        int(
                            obj.object_id
                        ),

                    "x":
                        int(obj.x),

                    "y":
                        int(obj.y),

                    "width":
                        int(
                            obj.width
                        ),

                    "height":
                        int(
                            obj.height
                        ),

                    "is_target":
                        bool(
                            obj.is_target
                        ),
                }
            )

        goal_cells = []

        for cell in getattr(
            env,
            "goal_cells",
            [],
        ):
            goal_cells.append(
                [
                    int(cell[0]),
                    int(cell[1]),
                ]
            )

        walls = []

        for cell in getattr(
            env,
            "walls",
            [],
        ):
            walls.append(
                [
                    int(cell[0]),
                    int(cell[1]),
                ]
            )

        converted_actions = {}

        for agent, action in (
            actions.items()
        ):
            if hasattr(
                action,
                "tolist",
            ):
                converted_actions[
                    agent
                ] = action.tolist()

            else:
                converted_actions[
                    agent
                ] = action

        frame = {
            "step":
                len(self.frames),

            "agents":
                agents,

            "objects":
                objects,

            "goal":
                goal_cells,

            "walls":
                walls,

            "actions":
                converted_actions,

            "action_probabilities":
                (
                    action_probabilities
                    if action_probabilities
                    is not None
                    else {}
                ),

            "reward":
                float(reward),

            "target_distance":
                info.get(
                    "target_distance",
                    None,
                ),

            "obstacle_blocking":
                info.get(
                    "obstacle_blocking",
                    None,
                ),

            "target_moved":
                info.get(
                    "target_moved",
                    0,
                ),

            "obstacle_moves":
                info.get(
                    "obstacle_moves",
                    0,
                ),

            "success":
                info.get(
                    "success",
                    False,
                ),
        }

        self.frames.append(
            frame
        )

    def finish(
        self,
        success,
        episode_length,
        episode_return,
    ):
        self.summary = {
            "success":
                bool(success),

            "episode_length":
                int(
                    episode_length
                ),

            "episode_return":
                float(
                    episode_return
                ),
        }

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

        data = {
            "scenario":
                self.scenario_name,

            "seed":
                self.seed,

            "width":
                self.width,

            "height":
                self.height,

            "action_mode":
                self.action_mode,

            "summary":
                self.summary,

            "frames":
                self.frames,
        }

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )