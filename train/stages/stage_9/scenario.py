# ============================================================
# Stage 9 시나리오: 파이프라인 검증용.
#
# target/goal 위치는 Stage 0-6 기본값으로 고정한다 (Stage 8의
# 위치 일반화가 아직 끝나지 않았으므로, 여기서 위치까지 같이
# 랜덤화하면 "한 번에 새 축 하나" 원칙을 또 어기게 된다).
#
# 유일하게 새로 도입하는 축은 스폰 방식(_spawn_split)이다:
# 일부 에이전트만 target 근처에 확정 스폰하고 나머지는 멀리
# 스폰해서, 통신 head/PPO 결합 액션 파이프라인이 실제로 굴러
# 가는지 확인한다.
#
# 주의:
# target 위치가 고정이라 멀리 스폰된 에이전트도 학습을 통해
# "어디로 가면 되는지" 이미 알고 있을 수 있다. 즉 이 스테이지는
# "통신이 실제로 필요해지는지"를 검증하지 못한다 - 그건 Stage 8의
# 위치 일반화(8B1~)가 끝난 뒤에 이 스폰 방식을 다시 얹어서
# 검증해야 한다. docs/curriculum_plan.md 참고.
# ============================================================


DEFAULT_TARGET = {
    "x": 4,
    "y": 2,
    "width": 3,
    "height": 2,
}

DEFAULT_GOAL_CELLS = [
    [4, 9],
    [5, 9],
    [6, 9],
    [4, 10],
    [5, 10],
    [6, 10],
]


def make_stage9_scenario(
    near_count,
    near_radius,
    far_min_distance,
    name,
):

    return {
        "name": name,

        "target": dict(
            DEFAULT_TARGET
        ),

        "goal_cells": [
            list(cell)
            for cell in DEFAULT_GOAL_CELLS
        ],

        "obstacles": [],

        "walls": [],

        "spawn_mode": {
            "type": "split",

            "near_count":
                near_count,

            "near_radius":
                near_radius,

            "far_min_distance":
                far_min_distance,
        },
    }
