from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkScenario:
    name: str
    config: Optional[dict] = None

    def to_env_dict(self):
        return self.config


# ============================================================
# IID
#
# config=None:
# custom scenario를 만들지 않고,
# 원래 Stage 6 환경을 그대로 사용한다.
# ============================================================

IID_SCENARIOS = [
    BenchmarkScenario(
        name="iid_stage6",
        config=None,
    )
]


# ============================================================
# Obstacle OOD
#
# Stage 6:
# target = (4, 2), 3x2
# goal   = x=4~6, y=9~10
#
# obstacle 위치만 변경
# ============================================================

OBSTACLE_SCENARIOS = [

    BenchmarkScenario(
        name="obstacle_left",
        config={
            "target": {
                "x": 4,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 3,
                    "y": 6,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 8,
                    "y": 5,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [4, 9],
                [5, 9],
                [6, 9],
                [4, 10],
                [5, 10],
                [6, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="obstacle_right",
        config={
            "target": {
                "x": 4,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 5,
                    "y": 6,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 8,
                    "y": 5,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [4, 9],
                [5, 9],
                [6, 9],
                [4, 10],
                [5, 10],
                [6, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="obstacle_far_left",
        config={
            "target": {
                "x": 4,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 1,
                    "y": 6,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 8,
                    "y": 5,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [4, 9],
                [5, 9],
                [6, 9],
                [4, 10],
                [5, 10],
                [6, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="obstacle_far_right",
        config={
            "target": {
                "x": 4,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 8,
                    "y": 6,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 2,
                    "y": 5,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [4, 9],
                [5, 9],
                [6, 9],
                [4, 10],
                [5, 10],
                [6, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="obstacle_near_target",
        config={
            "target": {
                "x": 4,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 4,
                    "y": 5,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 8,
                    "y": 5,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [4, 9],
                [5, 9],
                [6, 9],
                [4, 10],
                [5, 10],
                [6, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="obstacle_near_goal",
        config={
            "target": {
                "x": 4,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 4,
                    "y": 8,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 8,
                    "y": 5,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [4, 9],
                [5, 9],
                [6, 9],
                [4, 10],
                [5, 10],
                [6, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="obstacle_off_path",
        config={
            "target": {
                "x": 4,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 8,
                    "y": 6,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 1,
                    "y": 5,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [4, 9],
                [5, 9],
                [6, 9],
                [4, 10],
                [5, 10],
                [6, 10],
            ],
        },
    ),
]


# ============================================================
# Target / Goal OOD
# ============================================================

TARGET_GOAL_SCENARIOS = [

    BenchmarkScenario(
        name="lane_left",
        config={
            "target": {
                "x": 2,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 2,
                    "y": 6,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 8,
                    "y": 5,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [2, 9],
                [3, 9],
                [4, 9],
                [2, 10],
                [3, 10],
                [4, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="lane_right",
        config={
            "target": {
                "x": 6,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 6,
                    "y": 6,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 2,
                    "y": 5,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [6, 9],
                [7, 9],
                [8, 9],
                [6, 10],
                [7, 10],
                [8, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="target_lower",
        config={
            "target": {
                "x": 4,
                "y": 3,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 4,
                    "y": 6,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 8,
                    "y": 5,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [4, 9],
                [5, 9],
                [6, 9],
                [4, 10],
                [5, 10],
                [6, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="goal_higher",
        config={
            "target": {
                "x": 4,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 4,
                    "y": 5,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 8,
                    "y": 5,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [4, 8],
                [5, 8],
                [6, 8],
                [4, 9],
                [5, 9],
                [6, 9],
            ],
        },
    ),
]


# ============================================================
# Combined OOD
# ============================================================

COMBINED_SCENARIOS = [

    BenchmarkScenario(
        name="combined_left_shift",
        config={
            "target": {
                "x": 2,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 2,
                    "y": 5,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 7,
                    "y": 6,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [2, 9],
                [3, 9],
                [4, 9],
                [2, 10],
                [3, 10],
                [4, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="combined_right_shift",
        config={
            "target": {
                "x": 6,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 6,
                    "y": 5,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 2,
                    "y": 6,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [6, 9],
                [7, 9],
                [8, 9],
                [6, 10],
                [7, 10],
                [8, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="combined_shorter",
        config={
            "target": {
                "x": 4,
                "y": 3,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 4,
                    "y": 6,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 9,
                    "y": 4,
                    "width": 1,
                    "height": 2,
                },
            ],

            "goal_cells": [
                [4, 8],
                [5, 8],
                [6, 8],
                [4, 9],
                [5, 9],
                [6, 9],
            ],
        },
    ),
]


# ============================================================
# Stress
# ============================================================

STRESS_SCENARIOS = [

    BenchmarkScenario(
        name="stress_two_blocks",
        config={
            "target": {
                "x": 4,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 4,
                    "y": 5,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 4,
                    "y": 7,
                    "width": 3,
                    "height": 1,
                },
            ],

            "goal_cells": [
                [4, 9],
                [5, 9],
                [6, 9],
                [4, 10],
                [5, 10],
                [6, 10],
            ],
        },
    ),

    BenchmarkScenario(
        name="stress_partial_double",
        config={
            "target": {
                "x": 4,
                "y": 2,
                "width": 3,
                "height": 2,
            },

            "obstacles": [
                {
                    "x": 3,
                    "y": 5,
                    "width": 3,
                    "height": 1,
                },
                {
                    "x": 6,
                    "y": 7,
                    "width": 3,
                    "height": 1,
                },
            ],

            "goal_cells": [
                [4, 9],
                [5, 9],
                [6, 9],
                [4, 10],
                [5, 10],
                [6, 10],
            ],
        },
    ),
]