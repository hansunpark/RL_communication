from dataclasses import dataclass
from typing import List, Dict, Tuple


Position = Tuple[int, int]


@dataclass
class BenchmarkScenario:

    name: str
    category: str

    target_x: int
    target_y: int
    target_width: int
    target_height: int

    goal_x: int
    goal_y: int

    obstacles: List[Dict]

    description: str = ""

    def goal_cells(self):

        return [
            (
                self.goal_x + dx,
                self.goal_y + dy,
            )

            for dy in range(
                self.target_height
            )

            for dx in range(
                self.target_width
            )
        ]

    def to_env_dict(self):

        return {
            "name":
                self.name,

            "target": {
                "x":
                    self.target_x,

                "y":
                    self.target_y,

                "width":
                    self.target_width,

                "height":
                    self.target_height,
            },

            "goal_cells":
                self.goal_cells(),

            "obstacles":
                self.obstacles,

            "walls": [],

            "random_agents":
                True,

            "agent_positions":
                None,

            "reward_mode":
                "obstacle_shaping",
        }


def obstacle(
    x,
    y,
    width,
    height,
):
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


# ==========================================================
# A. IID Stage 6
# ==========================================================

IID_SCENARIOS = [

    BenchmarkScenario(
        name="iid_stage6",
        category="iid",

        target_x=4,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=9,

        obstacles=[
            obstacle(
                4,
                6,
                3,
                1,
            ),

            obstacle(
                8,
                5,
                1,
                2,
            ),
        ],

        description=(
            "Exactly the Stage 6 "
            "training layout with "
            "new agent spawn seeds."
        ),
    ),
]


# ==========================================================
# B. Obstacle position generalization
# ==========================================================

OBSTACLE_SCENARIOS = [

    BenchmarkScenario(
        name="obstacle_left",
        category="obstacle_ood",

        target_x=4,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=9,

        obstacles=[
            obstacle(
                3,
                6,
                3,
                1,
            ),
        ],

        description=(
            "Blocking obstacle shifted "
            "one cell left."
        ),
    ),

    BenchmarkScenario(
        name="obstacle_right",
        category="obstacle_ood",

        target_x=4,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=9,

        obstacles=[
            obstacle(
                5,
                6,
                3,
                1,
            ),
        ],

        description=(
            "Blocking obstacle shifted "
            "one cell right."
        ),
    ),

    BenchmarkScenario(
        name="obstacle_far_left",
        category="obstacle_ood",

        target_x=4,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=9,

        obstacles=[
            obstacle(
                2,
                6,
                3,
                1,
            ),
        ],

        description=(
            "Obstacle partially blocks "
            "left side of corridor."
        ),
    ),

    BenchmarkScenario(
        name="obstacle_far_right",
        category="obstacle_ood",

        target_x=4,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=9,

        obstacles=[
            obstacle(
                6,
                6,
                3,
                1,
            ),
        ],

        description=(
            "Obstacle partially blocks "
            "right side of corridor."
        ),
    ),

    BenchmarkScenario(
        name="obstacle_near_target",
        category="obstacle_ood",

        target_x=4,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=9,

        obstacles=[
            obstacle(
                4,
                5,
                3,
                1,
            ),
        ],

        description=(
            "Blocking obstacle is closer "
            "to target than during training."
        ),
    ),

    BenchmarkScenario(
        name="obstacle_near_goal",
        category="obstacle_ood",

        target_x=4,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=9,

        obstacles=[
            obstacle(
                4,
                7,
                3,
                1,
            ),
        ],

        description=(
            "Blocking obstacle is closer "
            "to goal."
        ),
    ),

    BenchmarkScenario(
        name="obstacle_off_path",
        category="obstacle_ood",

        target_x=4,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=9,

        obstacles=[
            obstacle(
                8,
                6,
                3,
                1,
            ),
        ],

        description=(
            "Obstacle does not block "
            "the direct path."
        ),
    ),
]


# ==========================================================
# C. Target / Goal positional generalization
# ==========================================================

TARGET_GOAL_SCENARIOS = [

    BenchmarkScenario(
        name="lane_left",
        category="target_goal_ood",

        target_x=2,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=2,
        goal_y=9,

        obstacles=[
            obstacle(
                2,
                6,
                3,
                1,
            ),
        ],

        description=(
            "Entire transport lane shifted "
            "left."
        ),
    ),

    BenchmarkScenario(
        name="lane_right",
        category="target_goal_ood",

        target_x=6,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=6,
        goal_y=9,

        obstacles=[
            obstacle(
                6,
                6,
                3,
                1,
            ),
        ],

        description=(
            "Entire transport lane shifted "
            "right."
        ),
    ),

    BenchmarkScenario(
        name="target_lower",
        category="target_goal_ood",

        target_x=4,
        target_y=3,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=9,

        obstacles=[
            obstacle(
                4,
                6,
                3,
                1,
            ),
        ],

        description=(
            "Target starts one row lower."
        ),
    ),

    BenchmarkScenario(
        name="goal_higher",
        category="target_goal_ood",

        target_x=4,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=8,

        obstacles=[
            obstacle(
                4,
                6,
                3,
                1,
            ),
        ],

        description=(
            "Goal starts one row higher."
        ),
    ),
]


# ==========================================================
# D. Combined generalization
# ==========================================================

COMBINED_SCENARIOS = [

    BenchmarkScenario(
        name="combined_left_shift",
        category="combined_ood",

        target_x=2,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=2,
        goal_y=9,

        obstacles=[
            obstacle(
                3,
                6,
                3,
                1,
            ),
        ],

        description=(
            "New lane and shifted obstacle."
        ),
    ),

    BenchmarkScenario(
        name="combined_right_shift",
        category="combined_ood",

        target_x=6,
        target_y=2,
        target_width=3,
        target_height=2,

        goal_x=6,
        goal_y=9,

        obstacles=[
            obstacle(
                5,
                6,
                3,
                1,
            ),
        ],

        description=(
            "New right lane and shifted "
            "blocking obstacle."
        ),
    ),

    BenchmarkScenario(
        name="combined_shorter",
        category="combined_ood",

        target_x=2,
        target_y=3,
        target_width=3,
        target_height=2,

        goal_x=2,
        goal_y=8,

        obstacles=[
            obstacle(
                2,
                6,
                3,
                1,
            ),
        ],

        description=(
            "Shifted lane and different "
            "transport distance."
        ),
    ),
]


# ==========================================================
# E. Stress tests
# ==========================================================

STRESS_SCENARIOS = [

    BenchmarkScenario(
        name="stress_two_blocks",
        category="stress",

        target_x=4,
        target_y=1,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=9,

        obstacles=[
            obstacle(
                4,
                4,
                3,
                1,
            ),

            obstacle(
                4,
                7,
                3,
                1,
            ),
        ],

        description=(
            "Two sequential blocking "
            "obstacles."
        ),
    ),

    BenchmarkScenario(
        name="stress_partial_double",
        category="stress",

        target_x=4,
        target_y=1,
        target_width=3,
        target_height=2,

        goal_x=4,
        goal_y=9,

        obstacles=[
            obstacle(
                3,
                5,
                3,
                1,
            ),

            obstacle(
                6,
                7,
                3,
                1,
            ),
        ],

        description=(
            "Two offset partial blockers."
        ),
    ),
]


ALL_SCENARIOS = (
    IID_SCENARIOS
    +
    OBSTACLE_SCENARIOS
    +
    TARGET_GOAL_SCENARIOS
    +
    COMBINED_SCENARIOS
    +
    STRESS_SCENARIOS
)