import random

from ..stage_8.curriculum import (
    Stage8ScenarioSampler,
)

from ..stage_8.scenario import (
    RandomScenarioGenerator,
)

from ..comm_scratch.scenario import (
    ScratchScenarioGenerator,
)


# ============================================================
# Stage 8-comm: 8B1부터 통신을 켜고 Stage 8을 이어가는 트랙
#
# stage_8A_size_mastered.pt(크기 랜덤, 위치 고정, 통신 없음)에서
# 출발해서, 위치 랜덤화(8B1~)가 시작되는 시점부터 comm_action을
# 실제로 env에 보낸다. 위치가 랜덤해지는 순간부터 target/goal을
# 먼저 본 에이전트와 못 본 에이전트 사이에 정보 비대칭이 생기므로,
# 통신이 처음으로 쓸모 있어지는 시점과 켜는 시점을 맞춘 것이다.
#
# 기하학은 Stage 8 레벨을 그대로 재사용하고(8B1~8D), 그 뒤에
# docs/curriculum_plan.md Phase 2(Stage 9)의 split spawn을 랜덤
# 기하학 위에 얹은 레벨 3개를 붙인다. split은 쉬운 쪽부터
# (3near/1far -> 2/2 -> 1/3).
#
# Stage 8 no-comm 트랙과 기하학/보상/버킷 규칙을 똑같이 맞춰서,
# 나중에 두 트랙의 체크포인트를 같은 조건에서 비교할 수 있게 한다.
# ============================================================



# ------------------------------------------------------------
# 8B0: 가로 밀기 도입 (8B1 앞에 추가, 2026-09-29)
#
# 1차/2차 실행에서 8B1이 20% 근처에 멈춰 있어서 실패를 분해해
# 보니, goal이 target 바로 아래(dx=0)면 88% 성공, 한 칸이라도
# 옆으로 어긋나면(dx!=0) 230판 중 0판이었다. Stage 0-7은 goal이
# 항상 target 바로 아래였고(Stage 7 lane shift도 target과 goal을
# 같이 옮김), 정책은 "옆으로 밀기"를 한 번도 배운 적이 없다.
# 8B1의 "위치 랜덤화"는 사실 "가로 밀기"라는 새 기술을 같이
# 요구하고 있었다.
#
# 그래서 target은 기본 위치에 고정한 채 goal만 옆으로 옮기는
# 레벨을 먼저 둔다 (새 축 = goal의 가로 어긋남 하나).
#
# 3차 실행에서 8B0a(아래로 7 + 옆으로 +-1)도 44,000 episode 동안
# dx!=0 버킷이 0%였다. "아래로 밀기"만으로 거리 보상 대부분을
# 받을 수 있어서, 옆면에 2명이 붙어 동시에 미는 동작을 끝내
# 발견하지 못했다. 그래서 그 앞에 "옆으로만 밀기" 레벨을 둔다:
# target과 goal이 같은 행에 있고 옆으로만 1~2칸 떨어져 있어서,
# 아래로 밀면 오히려 거리 보상을 잃는다.
#
# 4차 실행에서 8B0(target (4,5), 같은 행 옆으로)도 4,000 episode
# 동안 0%였다. 8A 정책을 직접 돌려보니 어떤 배치에서든 target을
# 아래로만 민다 (3,000번 넘는 target 이동 중 옆 방향 0번). 정책이
# "위에서 접근해 아래로 밀기"에 거의 결정적으로 굳어 있어서 두
# 명이 옆면에 서는 상황 자체가 생기지 않는다. 그래서 맨 앞에
# reverse curriculum 발판을 둔다: 2명을 밀어야 할 옆면에 미리
# 세워두면 둘이 같은 step에 같은 방향만 고르면 되므로, 지금
# entropy로도 우연히 성공할 수 있다. target은 정책이 아는 기본
# 위치 (4,2)에 둔다.
#
# 5차 실행: 8B0s(0.03 entropy)와 8B0은 통과했지만 8B0a에서 dx=0까지
# 전부 0%가 됐다. 8B0 통과 체크포인트를 직접 돌려보니 target을 옆으로만
# 밀고(396번) 아래로는 한 번도 안 민다 — 가로 레벨에서 아래로 밀 일이
# 전혀 없어서 "아래로 밀기"를 통째로 잊었다 (catastrophic forgetting).
# rehearsal은 이 실행에서 통과한 레벨만 섞으므로 8A 과제는 들어가지
# 않는다. 그래서 가로 전용 레벨에는 8A 과제(goal이 바로 아래)를
# below_mix 비율만큼 섞고, 그 버킷("goal_below")도 mastery 조건에 넣는다.
#
# 6차 실행: below 혼합 덕분에 아래로 밀기(95%)와 오른쪽 밀기는 유지/학습
# 됐지만, 30,000 episode 동안 왼쪽 밀기(dx<0)는 0%. 오른쪽 면에 세운
# 2명이 LEFT를 고를 확률이 각 3~4%라 둘이 동시에 밀 확률이 step당 약
# 0.1%였고, 오히려 RIGHT(52%)를 눌렀다 — 먼저 배운 "옆면에 붙으면 RIGHT"가
# 반대쪽 면에도 그대로 나왔다. 그래서 그 앞에 왼쪽 편향 발판을 둔다:
# 가로 episode의 75%가 왼쪽이고(오른쪽도 25% 남겨 잊지 않게 함), 이
# 레벨만 max_steps를 60으로 줄여 실패 episode를 빨리 끝낸다 — 2명이
# 옆면에 서 있는 시작 상태에서의 시도 횟수가 3배 이상 늘어난다.
#
# 8차 실행 (위로 밀기): 8B2(반경 5)에서 먼 위치 버킷이 45,000 episode
# 동안 ~50%에 멈췄다. goal 세로 위치별로 보니 goal이 target 아래면 88~96%,
# 같은 행 66%, **위면 0/58**이고, target 이동 3,742번 중 위 방향은 0번.
# Stage 0-8 계열은 위로 미는 법을 배운 적이 없다 (반경 5부터 goal이 target
# 위에 올 수 있다). 가로 밀기와 같은 처방으로 8B0b 뒤에 위로 밀기 발판
# (3명을 아래 면에 미리 세움) -> 미리 세우지 않는 단계를 둔다.
#
# 8차 실행 계속: 3x2 발판(8B0uS)에서 15,000 episode 동안 위로 밀기 0%.
# 아래 면에 선 에이전트의 5x5 시야는 y=2~6이라, target(y=2~3)보다 2칸
# 위의 goal(y=0~1)은 미는 3명 모두에게 안 보인다 (1칸 위는 goal_mask로만
# 보임). 게다가 3명이 동시에 UP을 고를 확률이 step당 0.08%였다. comm_scratch가
# 2x1부터 시작했듯 위로 밀기도 2칸 폭 target + goal 1칸 위(보임) + 2명
# 발판(8B0uS2)부터 시작한다 (현재 체크포인트로 3/212 성공 - 신호는 있다).
#
# 8B3(완전 랜덤 위치)에서 goal이 target 위면 0/63, 위+옆 7%로 정체.
# target이 처음에 보였는지는 무관(둘 다 40%) - 찾기가 아니라 밀 방향이
# 문제였다. (4,7)에 둔 target을 위로 2-5칸 밀기도 0/200 - 위로 2칸조차
# (4,2)에서는 통과했는데 다른 위치에선 0이었다. 위로 밀기가 (4,2)라는
# 좌표에 묶여 학습됐다 (Stage 0-7의 좌표 암기와 같은 문제). 그래서 target
# 위치를 랜덤으로 두는 긴 위로 밀기 발판(8B0uLs) -> 발판 없음(8B0uL)을
# 추가한다.
#
# 키: target_y(정수 또는 후보 튜플), target_x(후보 튜플, 기본 (4,)),
#     goal_y(정수 또는 후보 튜플) 또는 goal_dy(target 기준 세로 오프셋 후보),
#     shifts(가로 이동 후보), side_spawn(밀 면에 미리 세우기),
#     below_mix(8A 과제 비율), max_steps(None이면 기본값),
#     target_width(기본 3), target_sizes((w, h) 후보; 주면 target_width
#     대신 사용), random_position(True면 target을 벽에서 1칸 이상 떨어진
#     임의 위치에 둔다; target_x/target_y 무시)
#
# 8B4(크기 + 위치 랜덤) 5,000 episode 시점: 2x2/3x2/4x2는 61-68%인데
# 세로 3칸짜리가 2x3 11%, 3x3 21%, 4x3 18%. comm_scratch S3a와 같은
# 병목 — 3명이 옆면에 세로로 줄 서서 동시에 밀어야 한다. comm_scratch에서
# 통한 처방(S3s 발판 -> S3t 근처 스폰)을 그대로 8B0tS -> 8B0t로 둔다.
# ------------------------------------------------------------

LATERAL_LEVELS = [
    {
        "name": "stage_8B0sL_lateral_side_spawn_leftbias_comm",
        "target_y": 2,
        "goal_y": 2,
        "shifts": (-2, -1, -2, -1, -2, -1, 1, 2),
        "side_spawn": True,
        "below_mix": 0.3,
        "max_steps": 60,
    },
    {
        "name": "stage_8B0s_lateral_side_spawn_comm",
        "target_y": 2,
        "goal_y": 2,
        "shifts": (-2, -1, 1, 2),
        "side_spawn": True,
        "below_mix": 0.5,
        "max_steps": None,
    },
    {
        "name": "stage_8B0_lateral_only_comm",
        "target_y": 2,
        "goal_y": 2,
        "shifts": (-2, -1, 1, 2),
        "side_spawn": False,
        "below_mix": 0.5,
        "max_steps": None,
    },
    {
        "name": "stage_8B0a_goal_lateral1_comm",
        "target_y": 2,
        "goal_y": 9,
        "shifts": (-1, 0, 1),
        "side_spawn": False,
        "below_mix": 0.0,
        "max_steps": None,
    },
    {
        "name": "stage_8B0b_goal_lateral2_comm",
        "target_y": 2,
        "goal_y": 9,
        "shifts": (-2, -1, 0, 1, 2),
        "side_spawn": False,
        "below_mix": 0.0,
        "max_steps": None,
    },
    {
        "name": "stage_8B0uS2_up_narrow_side_spawn_comm",
        "target_y": 2,
        "goal_y": 1,
        "shifts": (0,),
        "side_spawn": True,
        "below_mix": 0.3,
        # 30으로 하면 섞어 넣은 8A 과제(평균 ~26 step)가 자주 시간
        # 초과돼서 goal_below 버킷이 0.85를 못 넘는다.
        "max_steps": 60,
        "target_width": 2,
    },
    {
        "name": "stage_8B0uS_up_side_spawn_comm",
        "target_y": 2,
        "goal_y": (0, 1),
        "shifts": (0,),
        "side_spawn": True,
        "below_mix": 0.3,
        "max_steps": 60,
    },
    {
        "name": "stage_8B0u_up_comm",
        "target_y": 2,
        "goal_y": (0, 1),
        "shifts": (0,),
        "side_spawn": False,
        "below_mix": 0.3,
        "max_steps": 100,
    },
    {
        "name": "stage_8B0uLs_up_long_side_spawn_comm",
        "target_x": tuple(range(1, 9)),
        "target_y": tuple(range(3, 10)),
        "goal_dy": (-1, -2, -3),
        "shifts": (0,),
        "side_spawn": True,
        "below_mix": 0.3,
        "max_steps": 60,
    },
    {
        "name": "stage_8B0uL_up_long_comm",
        "target_x": tuple(range(1, 9)),
        "target_y": tuple(range(5, 10)),
        "goal_dy": (-2, -3, -4, -5),
        "shifts": (0,),
        "side_spawn": False,
        "below_mix": 0.3,
        "max_steps": None,
    },
    {
        "name": "stage_8B0tS_tall_side_spawn_comm",
        "target_sizes": ((2, 3), (3, 3), (4, 3)),
        "random_position": True,
        "target_y": 0,
        "goal_dy": (0,),
        "shifts": (-2, -1, 1, 2),
        "side_spawn": True,
        "below_mix": 0.3,
        "max_steps": 60,
    },
    {
        "name": "stage_8B0t_tall_lateral_comm",
        "target_sizes": ((2, 3), (3, 3), (4, 3)),
        "random_position": True,
        "target_y": 0,
        "goal_dy": (0,),
        "shifts": (-2, -1, 1, 2),
        "side_spawn": False,
        "below_mix": 0.3,
        "max_steps": 100,
    },
]

DEFAULT_TARGET = {
    "x": 4,
    "y": 2,
    "width": 3,
    "height": 2,
}

# 가로 밀기를 새로 발견해야 하는 8B0* 레벨에서만 쓰는 이동
# entropy 계수. scratchpad 실험(8B0s, 8A 체크포인트 출발)에서
# 0.01은 2,000 episode 동안 0%, 0.03은 3,364 episode 만에 4개
# 버킷 모두 0.85 이상으로 통과했다. 다만 0.03은 Stage 8 8B1에서
# 정책을 무너뜨렸던 값이라, 8B1부터는 기본값(train.py의
# entropy_coef)으로 돌아간다.
LATERAL_ENTROPY_COEF = 0.03

SPLIT_NEAR_COUNTS = [3, 2, 1]

SPLIT_NEAR_RADIUS = 2  # = vision_radius (근처 에이전트는 target을 바로 본다)

SPLIT_FAR_MIN_DISTANCE = 6

# ------------------------------------------------------------
# 가로/위로 밀기 레벨(LATERAL_LEVELS) 뒤의 레벨들.
#
# stage8_level: Stage8ScenarioSampler의 레벨 인덱스(8B1=1 ... 8D=6)를
#     그대로 쓴다.
# obstacle_intro: Stage 8 생성기를 직접 만들어 obstacle 개수/크기를
#     지정한다 (크기 + 위치는 8B4처럼 완전 랜덤).
# near_count: split spawn을 얹는다.
# entropy_coef: 이 레벨에서만 쓰는 이동 entropy 계수.
#
# 8C0a/8C0b (2026-09-29): 8C(obstacle 0-3개)에서 5,000 episode 동안
# obstacle 3개 버킷이 0-3%. comm_scratch S10에서도 obstacle을 한 번에
# 0-2개 넣었다가 막혔고, 쉬운 조건에서 obstacle 1개부터 들이는 것으로
# 풀었다. 같은 처방: 작은 obstacle 1개(8C0a) -> 보통 obstacle 1개(8C0b)
# -> 8C.
#
# 참고: 8C0a를 entropy 0.03으로 돌렸다가 붕괴해서 기본값으로 되돌렸다
# (docs/curriculum_plan.md, 11차 실행).
#
# 12차: entropy 0.01로도 8C0a가 15,000 episode 동안 23/35/39%에 정체.
# obstacle이 경로 위면 18%, 경로 밖이어도 48% (obstacle 없을 때 89%).
# comm_scratch S10a는 쉬운 기반(goal 2-5칸) 위에 obstacle을 얹어서 경로
# 밖 81-86%였는데, 8C0a는 가장 어려운 기하학(goal 어디든) 위에 얹었다.
# 그래서 8C0a는 comm_scratch 생성기로 goal 거리를 2-5칸으로 제한한다
# (스폰은 이 트랙의 기본인 맵 전체 랜덤 유지).
#
# 그 다음 8C0b가 goal 거리(2-5 -> 무제한)와 obstacle 크기(작음 -> 보통)를
# 동시에 바꾸므로, 사이에 goal 거리만 푸는 8C0a2를 둔다 (comm_scratch
# S10b에서 세 축을 한꺼번에 바꿨다가 18/28/41%로 막힌 것과 같은 실수 방지).
#
# 8C0s: 8C0a가 60,000 episode에 버킷 73-83%로 상한에 걸렸다. 실패는 대부분
# obstacle이 target-goal 경로 위에 있을 때(1/15)였다. comm_scratch S10a2와
# 같은 발판 — 같은 행/열 goal 3-5칸, 그 통로 위 1x1 obstacle, 근처 스폰,
# max_steps 100, entropy 0.03 — 을 8C0a 앞에 둔다.
#
# 8C0s1: 8C0s가 30,000 episode에 최저 버킷 26/27/31%로 느렸다. comm_scratch
# S10a1과 같은 이유로 작은 target(2x1, 2x2) 전용 발판을 앞에 둔다.
# ------------------------------------------------------------

MAIN_LEVELS = [
    {
        "name": "stage_8B1_position_r2_comm",
        "stage8_level": 1,
    },
    {
        "name": "stage_8B2_position_r5_comm",
        "stage8_level": 2,
    },
    {
        "name": "stage_8B3_position_full_comm",
        "stage8_level": 3,
    },
    {
        "name": "stage_8B4_size_and_position_comm",
        "stage8_level": 4,
    },
    {
        "name": "stage_8C0s1_clear_obstacle_small_target_comm",
        "obstacle_intro": {
            "count": (1, 1),
            "size": ((1, 1), (1, 1)),
            "goal_distance_range": (3, 5),
            "goal_axis": "straight",
            "obstacle_in_path": True,
            "target_size_range": ((2, 2), (1, 2)),
            "spawn": {
                "type": "near",
                "radius": SPLIT_NEAR_RADIUS,
            },
        },
        "max_steps": 100,
        "entropy_coef": LATERAL_ENTROPY_COEF,
    },
    {
        "name": "stage_8C0s_clear_obstacle_scaffold_comm",
        "obstacle_intro": {
            "count": (1, 1),
            "size": ((1, 1), (1, 1)),
            "goal_distance_range": (3, 5),
            "goal_axis": "straight",
            "obstacle_in_path": True,
            "spawn": {
                "type": "near",
                "radius": SPLIT_NEAR_RADIUS,
            },
        },
        "max_steps": 100,
        "entropy_coef": LATERAL_ENTROPY_COEF,
    },
    {
        "name": "stage_8C0a_one_small_obstacle_comm",
        "obstacle_intro": {
            "count": (1, 1),
            "size": ((1, 2), (1, 1)),
            "goal_distance_range": (2, 5),
        },
        # entropy 0.03을 쓰다가 붕괴했다 (8B4 과제 214/240 -> 48/240,
        # 이동 entropy 1.2). 짧은 발판 레벨과 달리 이 레벨은 episode가
        # 길어서 0.03이 정책을 무작위 쪽으로 밀었다. 기본값(0.01)을 쓴다.
    },
    {
        "name": "stage_8C0a2_one_small_obstacle_goal_any_comm",
        "obstacle_intro": {
            "count": (1, 1),
            "size": ((1, 2), (1, 1)),
        },
    },
    {
        "name": "stage_8C0b_one_obstacle_comm",
        "obstacle_intro": {
            "count": (1, 1),
            "size": ((1, 3), (1, 2)),
        },
    },
    {
        "name": "stage_8C_obstacles_comm",
        "stage8_level": 5,
    },
    {
        "name": "stage_8D_full_comm",
        "stage8_level": 6,
    },
] + [
    {
        "name": f"stage_9_split_{near}near_comm",
        # split 레벨은 Stage 8 최종 기하학(8D) 위에 얹는다.
        "stage8_level": 6,
        "near_count": near,
    }
    for near in SPLIT_NEAR_COUNTS
]


class Stage8CommSampler:

    LEVEL_NAMES = [
        level["name"]
        for level in LATERAL_LEVELS
    ] + [
        level["name"]
        for level in MAIN_LEVELS
    ]

    def __init__(
        self,
        map_width=12,
        map_height=12,
        help_penalty=0.002,
        max_steps=200,
    ):
        self.stage8 = Stage8ScenarioSampler(
            map_width=map_width,
            map_height=map_height,
        )

        self.help_penalty = help_penalty

        self.max_steps = max_steps

        self.level = 0

    def _main_level(self):

        return MAIN_LEVELS[
            self.level
            - len(LATERAL_LEVELS)
        ]

    @property
    def num_lateral_levels(self):

        return len(LATERAL_LEVELS)

    def _lateral_scenario(self):

        level = LATERAL_LEVELS[self.level]

        if "target_sizes" in level:
            target_width, target_height = (
                random.choice(
                    level["target_sizes"]
                )
            )
        else:
            target_width = level.get(
                "target_width",
                DEFAULT_TARGET["width"],
            )

            target_height = (
                DEFAULT_TARGET["height"]
            )

        target_y = level["target_y"]

        if isinstance(target_y, tuple):
            target_y = random.choice(target_y)

        target_x = random.choice(
            level.get(
                "target_x",
                (DEFAULT_TARGET["x"],),
            )
        )

        if level.get("random_position"):

            # 벽에서 1칸 이상 (벽에 붙으면 못 민다), 옆으로
            # 밀 여유(최대 2칸)를 남긴다.
            target_x = random.randint(
                3,
                self.stage8.map_width
                - target_width
                - 3,
            )

            target_y = random.randint(
                1,
                self.stage8.map_height
                - target_height
                - 1,
            )

        if "goal_dy" in level:
            goal_y = target_y + random.choice(
                level["goal_dy"]
            )
        else:
            goal_y = level["goal_y"]

        if random.random() < level["below_mix"]:

            # 8A 과제: 기본 위치, 크기 랜덤, goal은 바로 아래.
            self.stage8.level = 0

            scenario = self.stage8.sample()

            scenario["name"] = self.name

            scenario["variant"] = "below"

            return scenario

        dx = random.choice(level["shifts"])

        goal_x = target_x + dx

        if isinstance(goal_y, tuple):
            goal_y = random.choice(goal_y)

        target = dict(DEFAULT_TARGET)

        target["x"] = target_x

        target["y"] = target_y

        target["width"] = target_width

        target["height"] = target_height

        scenario = {
            "name": self.name,
            "target": target,
            "goal_cells": [
                [goal_x + cx, goal_y + cy]
                for cy in range(
                    target["height"]
                )
                for cx in range(
                    target["width"]
                )
            ],
            "obstacles": [],
            "walls": [],
        }

        if level["side_spawn"]:
            scenario["agent_positions"] = (
                self._side_spawn_positions(
                    target,
                    dx,
                    goal_y - target_y,
                )
            )

        return scenario

    def _side_spawn_positions(
        self,
        target,
        dx,
        dy=0,
    ):
        """
        goal 쪽으로 밀 수 있도록 반대편 면의 모든 칸에 에이전트를
        세우고(옆으로 밀면 2명, 위로 밀면 3명), 나머지는 맵의 빈
        칸에 랜덤으로 둔다.
        """

        if dx == 0 and dy < 0:
            # 위로 밀기: target 아래 면
            push_y = (
                target["y"]
                + target["height"]
            )

            pushers = [
                (target["x"] + cx, push_y)
                for cx in range(
                    target["width"]
                )
            ]

        else:
            push_x = (
                target["x"] - 1
                if dx > 0
                else target["x"]
                + target["width"]
            )

            pushers = [
                (push_x, target["y"] + cy)
                for cy in range(
                    target["height"]
                )
            ]

        blocked = {
            (target["x"] + cx, target["y"] + cy)
            for cy in range(target["height"])
            for cx in range(target["width"])
        } | set(pushers)

        free = [
            (x, y)
            for y in range(
                self.stage8.map_height
            )
            for x in range(
                self.stage8.map_width
            )
            if (x, y) not in blocked
        ]

        others = random.sample(
            free,
            4 - len(pushers),
        )

        positions = pushers + others

        random.shuffle(positions)

        return {
            f"agent_{i}": list(position)
            for i, position
            in enumerate(positions)
        }

    @property
    def name(self):

        return self.LEVEL_NAMES[
            self.level
        ]

    def sample(self):

        if (
            self.level
            < self.num_lateral_levels
        ):
            return self._lateral_scenario()

        level = self._main_level()

        if (
            "obstacle_intro" in level
            and "goal_distance_range"
            in level["obstacle_intro"]
        ):
            intro = level["obstacle_intro"]

            generator = ScratchScenarioGenerator(
                map_width=self.stage8.map_width,
                map_height=self.stage8.map_height,
                goal_distance_range=
                    intro["goal_distance_range"],
                obstacle_count_range=
                    intro["count"],
                obstacle_size_range=
                    intro["size"],
                spawn=intro.get(
                    "spawn",
                    {"type": "random"},
                ),
                goal_axis=intro.get(
                    "goal_axis"
                ),
                obstacle_in_path=intro.get(
                    "obstacle_in_path",
                    False,
                ),
                target_size_range=intro.get(
                    "target_size_range",
                    ((2, 4), (2, 3)),
                ),
            )

            scenario = generator.sample(
                name=self.name
            )

        elif "obstacle_intro" in level:

            generator = RandomScenarioGenerator(
                map_width=self.stage8.map_width,
                map_height=self.stage8.map_height,
                randomize_size=True,
                randomize_position=True,
                position_radius=None,
                obstacle_count_range=level[
                    "obstacle_intro"
                ]["count"],
                obstacle_size_range=level[
                    "obstacle_intro"
                ]["size"],
            )

            scenario = generator.sample(
                name=self.name
            )

        else:
            self.stage8.level = level[
                "stage8_level"
            ]

            scenario = self.stage8.sample()

        scenario["name"] = self.name

        if "near_count" in level:
            scenario["spawn_mode"] = {
                "type": "split",
                "near_count":
                    level["near_count"],
                "near_radius":
                    SPLIT_NEAR_RADIUS,
                "far_min_distance":
                    SPLIT_FAR_MIN_DISTANCE,
            }

        return scenario

    def bucket_for(self, scenario):

        if (
            self.level
            < self.num_lateral_levels
        ):
            if (
                scenario.get("variant")
                == "below"
            ):
                return "goal_below"

            goal_y = min(
                cell[1]
                for cell in scenario[
                    "goal_cells"
                ]
            )

            if (
                goal_y
                < scenario["target"]["y"]
            ):
                return (
                    "goal_up"
                    f"{scenario['target']['y'] - goal_y}"
                )

            if (
                "target_sizes"
                in LATERAL_LEVELS[self.level]
            ):
                # 세로로 긴 target 레벨은 모양 x 좌우 방향별로
                # 본다 (한 모양/한 방향만 뒤처지는 것이 평균에
                # 가려지지 않게).
                goal_x = min(
                    cell[0]
                    for cell in scenario[
                        "goal_cells"
                    ]
                )

                direction = (
                    "right"
                    if goal_x
                    > scenario["target"]["x"]
                    else "left"
                )

                return (
                    f"{scenario['target']['width']}x"
                    f"{scenario['target']['height']}"
                    f"_{direction}"
                )

            goal_x = min(
                cell[0]
                for cell in scenario[
                    "goal_cells"
                ]
            )

            return (
                f"goal_dx"
                f"{goal_x - scenario['target']['x']:+d}"
            )

        bucket = self.stage8.bucket_for(
            scenario
        )

        spawn_mode = scenario.get(
            "spawn_mode"
        )

        if spawn_mode:
            bucket += (
                f"_near"
                f"{spawn_mode['near_count']}"
            )

        return bucket

    def train_settings(self):

        if (
            self.level
            < self.num_lateral_levels
        ):
            return {
                "entropy_coef":
                    LATERAL_ENTROPY_COEF,
            }

        level = self._main_level()

        if "entropy_coef" in level:
            return {
                "entropy_coef":
                    level["entropy_coef"],
            }

        return {}

    def env_settings(self):

        max_steps = self.max_steps

        if (
            self.level
            < self.num_lateral_levels
        ):
            max_steps = (
                LATERAL_LEVELS[self.level][
                    "max_steps"
                ]
                or self.max_steps
            )

        else:
            max_steps = self._main_level().get(
                "max_steps",
                self.max_steps,
            )

        return {
            "help_penalty":
                self.help_penalty,
            "max_steps": max_steps,
        }

    def advance(self):

        if (
            self.level
            < len(self.LEVEL_NAMES) - 1
        ):
            self.level += 1

            return True

        return False
