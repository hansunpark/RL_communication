from .scenario import (
    ScratchScenarioGenerator,
)


# ============================================================
# comm_scratch 커리큘럼: 처음부터 통신을 켠 별도 모델
#
# 기존 Stage 0-8에서 배운 교훈을 반영한 설계 원칙:
#
# 1. target 위치는 첫 레벨부터 맵 전체에서 랜덤이다. 난이도는
#    좌표 고정이 아니라 크기 / goal 거리 / 스폰 방식 / obstacle로
#    조절한다. (Stage 0-7이 고정 좌표에 과적합해서 8B에서 무너진
#    것을 반복하지 않기 위함.)
# 2. 한 레벨에 새 축은 하나만 바꾼다 (Stage 8B 2차 시도 실패 교훈).
# 3. 통신은 처음부터 켠다. 다만 help_penalty는 정보 비대칭이 없는
#    동안(모든 에이전트가 target 근처에 스폰되고 goal도 가까워서
#    다 보이는 레벨)에는 0으로 둔다. 쓸모가 없는 동안 페널티만
#    받으면 HELP가 0%로 굳어서 나중에 쓸모가 생겨도 다시 탐색하기
#    어렵기 때문이다. goal이 시야(5x5) 밖으로 나가기 시작하는
#    레벨부터 페널티를 켠다.
# 4. 정보 비대칭(스폰 분리)은 위치가 이미 랜덤인 상태 위에 바로
#    얹으므로, 통신이 실제로 쓸모 있는 환경에서 학습된다.
#
# S3 분할 (2026-09-29): 처음에는 S2(3x2) 다음에 바로 2-4 x 2-3
# 전체로 넘어갔는데, 4칸 폭 target은 4명 전원이 한 면에 붙어야
# 밀 수 있어서 난이도가 급격히 뛰었다. 28,000 episode 동안 large
# 버킷은 0%, S2에서 85~97%였던 small까지 25~39%로 무너졌다.
# 한 면 최대 3칸(S3a)을 먼저 거치게 쪼갰다.
#
# S3a/S3b entropy (2026-09-29): S3a의 병목은 세로로 긴 target(2x3,
# 3x3)을 옆으로 미는 동작이었다 — 3명이 옆면에 세로로 줄 서서
# 동시에 밀어야 한다 (2x3 22%, 가로 방향 goal 33%, 세로 86%). 같은
# 체크포인트에서 이 레벨만 비교했을 때 entropy_coef 0.01은 19,400
# episode에서 최저 버킷 9/20/29%로 하락, 0.03은 12,000 episode에서
# 40/52/65%로 상승했다. 그래서 새 협동 동작이 필요한 S3a/S3b에만
# 0.03을 쓴다 (stage_8_comm의 가로 밀기 레벨과 같은 처방).
#
# S3s 발판 (2026-09-29): 위 비교를 이어서 보니 0.03 쪽도 10,000 episode에
# 13/31/60%였다가 12,000에서 40/52/65%로 튀는 등 두 설정 모두 20~50%를
# 오르내리는 정체였고(100-episode 창의 노이즈가 컸다), 0.03을 계속 쓰면
# 이동 entropy가 1.18까지 올라갔다. stage_8_comm에서 가로 밀기를 푼 것은
# entropy 단독이 아니라 "밀 자리에 미리 세우기 + 짧은 episode" 발판이었으므로
# 같은 처방을 쓴다: S3s는 세로로 긴 target(2x3, 3x3)을 같은 행에서 옆으로
# 2~3칸 밀되, 3명을 goal 반대편 옆면에 미리 세우고 max_steps를 60으로
# 줄인다. entropy 0.03은 S3s에만 쓰고 S3a/S3b는 기본값으로 돌린다.
# goal이 target과 겹치지 않게 하면 3x3은 최소 3칸 떨어져서 3명이 연속
# 3번 동시에 밀어야 했고, 시작 체크포인트가 120판 중 0판이었다. 그래서
# S3s만 겹침을 허용하고 거리를 1~2칸으로 둔다 (stage_8_comm 발판도 1칸부터).
#
# S3t (2026-09-29): S3s는 통과했지만 S3a에서 세로로 긴 target의 가로 밀기가
# 그대로 0%였다 (2x3 가로 0/21, 3x3 가로 0/13). 미리 세워두지 않으면 옆면에
# 스스로 대형을 짜지 못한다. stage_8_comm은 발판(8B0s) 다음에 "미리 세우지
# 않고 옆으로만"(8B0) 단계를 거쳤는데 여기선 그 단계를 건너뛰었었다.
# S3t = S3s와 같은 배치(세로로 긴 target, 같은 행 goal)에서 근처 랜덤 스폰.
#
# S10a/S10b (2026-09-29): S10(가장 어려운 조건인 split 1near + goal 거리
# 무제한 위에 크기가 제각각인 obstacle 0-2개)에서 5,000 episode 동안
# obstacle 버킷이 7-17%. S9 통과 체크포인트로 보면 obstacle 0개 88%,
# 1개 34%, 2개 9% — obstacle 자체가 큰 새 축이었다. 실패가 대부분이
# 되면서 obstacle 없는 과제도 88% -> 67%로 떨어지기 시작했다. Stage 4-6이
# 쉬운 배치에서 obstacle을 들였던 것처럼, 쉬운 기반(근처 스폰, 가까운 goal)
# 위에 작은 obstacle 1개(S10a) -> 어려운 기반 위에 보통 obstacle 1개(S10b)
# -> 기존 S10 순서로 쪼갠다.
#
# S10a2 (2026-09-29): 두 트랙 모두 obstacle이 target-goal 경로 위에 있을
# 때 거의 못 풀었다 (comm_scratch S10b 경로 위 0-38%, stage8_comm 8C0a
# 1/15). S10b가 45,000 episode에 최저 버킷 42/53/64%로 정체해서, 경로를
# 막은 1x1 obstacle을 치우는 발판(같은 행/열 goal 3-5칸, 근처 스폰,
# max_steps 100, entropy 0.03)을 S10a와 S10b 사이에 둔다.
#
# S10a1: S10a2가 15,000 episode 동안 큰 target 0-4%, 작은 target 15-35%.
# 큰 target은 통로가 3칸 높이라 obstacle을 여러 칸 밀어내야 한다. 2x1부터
# 시작한 S0, 2칸 폭부터 시작한 위로 밀기처럼 작은 target(2x1, 2x2) 전용
# 발판을 앞에 둔다 — 통로가 2칸 이하라 한 번 옆으로 밀면 비워진다.
#
# S10b 분할 (2026-09-29): S10a 다음 S10b에서 goal 거리(2-5 -> 무제한), 스폰
# (근처 -> 랜덤), obstacle 크기(작음 -> 보통) 세 축을 한꺼번에 바꿨더니
# 5,000 episode에 최저 버킷 18/28/41%. 한 레벨에 한 축씩으로 다시 쪼갠다.
#
# 레벨 표 (각 행에서 바뀐 축만 표시):
#
#   S0  2x1 target, goal 거리 2-3, 근처 스폰(r2)       [기본 밀기]
#   S1  goal 거리 2-5                                   (goal 거리)
#   S2  target 3x2                                      (크기)
#   S3s 발판: 2x3/3x3을 옆으로, 3명을 옆면에 미리 세움  (새 협동 동작)
#   S3t 같은 배치, 근처 랜덤 스폰 (미리 세우지 않음)      (스폰)
#   S3a target 크기 랜덤 (2-3 x 2-3, 한 면 최대 3칸)    (크기)
#   S3b target 크기 랜덤 (2-4 x 2-3)                    (크기)
#   S4  goal 거리 2-8, help_penalty 켬                  (goal 거리)
#   S5  goal 거리 제한 없음                             (goal 거리)
#   S6  스폰 반경 5                                     (스폰)
#   S7  스폰 완전 랜덤                                  (스폰)
#   S8  split 2near/2far                                (스폰)
#   S9  split 1near/3far                                (스폰)
#   S10a 근처 스폰, goal 2-5, 작은 obstacle 정확히 1개   (obstacle)
#   S10a1 발판: 작은 target(2x1, 2x2)으로 경로 위 obstacle 치우기
#   S10a2 발판: 경로 위 1x1 obstacle 치우기 (모든 크기)  (새 협동 동작)
#   S10b goal 무제한 (근처 스폰, 작은 obstacle 1개)     (goal 거리)
#   S10c 스폰 랜덤                                        (스폰)
#   S10d obstacle 보통 크기 (1-3 x 1-2)                   (obstacle 크기)
#   S10 split 1near + obstacle 0-2개                     (obstacle 수)
#   S11 obstacle 0-4개                                  (obstacle)
# ============================================================


HELP_PENALTY = 0.002

NEAR_RADIUS = 2  # = vision_radius

FAR_MIN_DISTANCE = 6

RANDOM_SIZE = None

FULL_SIZE_RANGE = ((2, 4), (2, 3))

DEFAULT_OBSTACLE_SIZE_RANGE = ((1, 3), (1, 2))

NEW_COORDINATION_ENTROPY_COEF = 0.03

LEVELS = [
    {
        "name": "S0_push_2x1_d2-3",
        "target_size": (2, 1),
        "goal_distance_range": (2, 3),
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": 0.0,
    },
    {
        "name": "S1_goal_d2-5",
        "target_size": (2, 1),
        "goal_distance_range": (2, 5),
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": 0.0,
    },
    {
        "name": "S2_size_3x2",
        "target_size": (3, 2),
        "goal_distance_range": (2, 5),
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": 0.0,
    },
    {
        "name": "S3s_tall_side_push_scaffold",
        "entropy_coef": NEW_COORDINATION_ENTROPY_COEF,
        "target_size": RANDOM_SIZE,
        "target_size_range": ((2, 3), (3, 3)),
        "goal_distance_range": (1, 2),
        "goal_axis": "horizontal",
        "allow_goal_overlap": True,
        "spawn": {
            "type": "push_face",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": 0.0,
        "max_steps": 60,
    },
    {
        "name": "S3t_tall_side_push_near",
        "entropy_coef": NEW_COORDINATION_ENTROPY_COEF,
        "target_size": RANDOM_SIZE,
        "target_size_range": ((2, 3), (3, 3)),
        "goal_distance_range": (1, 3),
        "goal_axis": "horizontal",
        "allow_goal_overlap": True,
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": 0.0,
        "max_steps": 100,
    },
    {
        "name": "S3a_size_max3",
        "target_size": RANDOM_SIZE,
        "target_size_range": ((2, 3), (2, 3)),
        "goal_distance_range": (2, 5),
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": 0.0,
    },
    {
        "name": "S3b_size_random",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": (2, 5),
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": 0.0,
    },
    {
        "name": "S4_goal_d2-8",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": (2, 8),
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": HELP_PENALTY,
    },
    {
        "name": "S5_goal_any",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": None,
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": HELP_PENALTY,
    },
    {
        "name": "S6_spawn_r5",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": None,
        "spawn": {
            "type": "near",
            "radius": 5,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": HELP_PENALTY,
    },
    {
        "name": "S7_spawn_random",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": None,
        "spawn": {"type": "random"},
        "obstacle_count_range": (0, 0),
        "help_penalty": HELP_PENALTY,
    },
    {
        "name": "S8_split_2near",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": None,
        "spawn": {
            "type": "split",
            "near_count": 2,
            "near_radius": NEAR_RADIUS,
            "far_min_distance":
                FAR_MIN_DISTANCE,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": HELP_PENALTY,
    },
    {
        "name": "S9_split_1near",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": None,
        "spawn": {
            "type": "split",
            "near_count": 1,
            "near_radius": NEAR_RADIUS,
            "far_min_distance":
                FAR_MIN_DISTANCE,
        },
        "obstacle_count_range": (0, 0),
        "help_penalty": HELP_PENALTY,
    },
    {
        "name": "S10a_one_small_obstacle_near",
        "entropy_coef": NEW_COORDINATION_ENTROPY_COEF,
        "target_size": RANDOM_SIZE,
        "goal_distance_range": (2, 5),
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (1, 1),
        "obstacle_size_range": ((1, 2), (1, 1)),
        "help_penalty": HELP_PENALTY,
    },
    {
        "name": "S10a1_clear_obstacle_small_target_scaffold",
        "entropy_coef": NEW_COORDINATION_ENTROPY_COEF,
        "target_size": RANDOM_SIZE,
        "target_size_range": ((2, 2), (1, 2)),
        "goal_distance_range": (3, 5),
        "goal_axis": "straight",
        "obstacle_in_path": True,
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (1, 1),
        "help_penalty": HELP_PENALTY,
        "max_steps": 100,
    },
    {
        "name": "S10a2_clear_obstacle_in_path_scaffold",
        "entropy_coef": NEW_COORDINATION_ENTROPY_COEF,
        "target_size": RANDOM_SIZE,
        "goal_distance_range": (3, 5),
        "goal_axis": "straight",
        "obstacle_in_path": True,
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (1, 1),
        "help_penalty": HELP_PENALTY,
        "max_steps": 100,
    },
    {
        "name": "S10b_small_obstacle_goal_any",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": None,
        "spawn": {
            "type": "near",
            "radius": NEAR_RADIUS,
        },
        "obstacle_count_range": (1, 1),
        "obstacle_size_range": ((1, 2), (1, 1)),
        "help_penalty": HELP_PENALTY,
    },
    {
        "name": "S10c_small_obstacle_random_spawn",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": None,
        "spawn": {"type": "random"},
        "obstacle_count_range": (1, 1),
        "obstacle_size_range": ((1, 2), (1, 1)),
        "help_penalty": HELP_PENALTY,
    },
    {
        "name": "S10d_one_obstacle_random_spawn",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": None,
        "spawn": {"type": "random"},
        "obstacle_count_range": (1, 1),
        "help_penalty": HELP_PENALTY,
    },
    {
        "name": "S10_obstacles_0-2",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": None,
        "spawn": {
            "type": "split",
            "near_count": 1,
            "near_radius": NEAR_RADIUS,
            "far_min_distance":
                FAR_MIN_DISTANCE,
        },
        "obstacle_count_range": (0, 2),
        "help_penalty": HELP_PENALTY,
    },
    {
        "name": "S11_obstacles_0-4",
        "target_size": RANDOM_SIZE,
        "goal_distance_range": None,
        "spawn": {
            "type": "split",
            "near_count": 1,
            "near_radius": NEAR_RADIUS,
            "far_min_distance":
                FAR_MIN_DISTANCE,
        },
        "obstacle_count_range": (0, 4),
        "help_penalty": HELP_PENALTY,
    },
]


class CommScratchSampler:

    LEVEL_NAMES = [
        level["name"]
        for level in LEVELS
    ]

    def __init__(
        self,
        map_width=12,
        map_height=12,
        max_steps=200,
    ):
        self.map_width = map_width
        self.map_height = map_height
        self.max_steps = max_steps

        self.level = 0

    @property
    def name(self):

        return self.LEVEL_NAMES[
            self.level
        ]

    @property
    def config(self):

        return LEVELS[self.level]

    def _generator(self):

        config = self.config

        return ScratchScenarioGenerator(
            map_width=self.map_width,
            map_height=self.map_height,
            target_size=
                config["target_size"],
            target_size_range=
                config.get(
                    "target_size_range",
                    FULL_SIZE_RANGE,
                ),
            goal_distance_range=
                config[
                    "goal_distance_range"
                ],
            obstacle_count_range=
                config[
                    "obstacle_count_range"
                ],
            obstacle_size_range=config.get(
                "obstacle_size_range",
                DEFAULT_OBSTACLE_SIZE_RANGE,
            ),
            spawn=config["spawn"],
            goal_axis=config.get(
                "goal_axis"
            ),
            allow_goal_overlap=config.get(
                "allow_goal_overlap",
                False,
            ),
            obstacle_in_path=config.get(
                "obstacle_in_path",
                False,
            ),
        )

    def sample(self):

        return self._generator().sample(
            name=self.name
        )

    # --------------------------------------------------------
    # 버킷: (target 크기, target-goal 거리, 스폰, obstacle 수)
    #
    # Stage 8과 달리 거리는 "기본 위치에서 얼마나 옮겨졌나"가
    # 아니라 "target에서 goal까지 얼마나 먼가"로 잰다 (여기엔
    # 기본 위치라는 개념이 없다). goal이 시야 밖인지가 핵심
    # 난이도라서, d6+부터는 goal이 처음에 안 보일 수 있다.
    # obstacle 수는 버킷 폭발을 막으려고 0 / 1-2 / 3-4로 묶는다.
    # --------------------------------------------------------

    def bucket_for(self, scenario):

        target = scenario["target"]

        area = (
            target["width"]
            * target["height"]
        )

        size_bucket = (
            "small"
            if area <= 6
            else "large"
        )

        goal_x = min(
            cell[0]
            for cell in scenario[
                "goal_cells"
            ]
        )

        goal_y = min(
            cell[1]
            for cell in scenario[
                "goal_cells"
            ]
        )

        distance = (
            abs(goal_x - target["x"])
            + abs(goal_y - target["y"])
        )

        if distance <= 3:
            distance_bucket = "g0-3"
        elif distance <= 5:
            distance_bucket = "g4-5"
        elif distance <= 8:
            distance_bucket = "g6-8"
        else:
            distance_bucket = "g9+"

        obstacle_count = len(
            scenario.get(
                "obstacles",
                [],
            )
        )

        if obstacle_count == 0:
            obstacle_bucket = "obs0"
        elif obstacle_count <= 2:
            obstacle_bucket = "obs1-2"
        else:
            obstacle_bucket = "obs3-4"

        spawn_mode = scenario.get(
            "spawn_mode"
        )

        bucket = (
            f"{size_bucket}_"
            f"{distance_bucket}_"
            f"{obstacle_bucket}"
        )

        if spawn_mode:
            bucket += (
                f"_near"
                f"{spawn_mode['near_count']}"
            )

        # 발판 레벨은 밀기 방향(좌/우)별로 따로 본다
        # (stage_8_comm에서 한쪽 방향만 배우고 반대쪽은 0%였던
        # 비대칭이 평균에 가려지지 않게).
        if "variant" in scenario:
            bucket += (
                f"_{scenario['variant']}"
            )

        return bucket

    def train_settings(self):

        if "entropy_coef" in self.config:
            return {
                "entropy_coef":
                    self.config[
                        "entropy_coef"
                    ],
            }

        return {}

    def env_settings(self):

        return {
            "help_penalty":
                self.config[
                    "help_penalty"
                ],
            "max_steps": self.config.get(
                "max_steps",
                self.max_steps,
            ),
        }

    def advance(self):

        if (
            self.level
            < len(self.LEVEL_NAMES) - 1
        ):
            self.level += 1

            return True

        return False
