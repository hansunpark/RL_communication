import random
from collections import deque


# ============================================================
# Stage 7 Scenario Sampler
# ============================================================


class Stage7ScenarioSampler:
    """
    Stage 6 geometry를 기반으로
    target / goal / blocking obstacle의 x 위치만 바꾼다.

    핵심:
        Stage 7A -> 작은 shift
        Stage 7B -> 더 큰 shift
        Stage 7C -> 전체 valid lane

    기존 Stage 6를 일정 확률로 계속 섞어서
    catastrophic forgetting을 줄인다.
    """

    def __init__(
        self,
        map_width=12,
        original_probability=0.4,
    ):
        self.map_width = map_width

        self.original_probability = (
            original_probability
        )

        # 현재 substage
        #
        # 0 = 7A
        # 1 = 7B
        # 2 = 7C
        self.level = 0

        # Stage 6 기준 target 위치
        self.original_x = 4

        # Target 크기
        self.target_width = 3
        self.target_height = 2

    # --------------------------------------------------------
    # 현재 Stage 이름
    # --------------------------------------------------------

    @property
    def name(self):

        names = [
            "stage_7A_shift1",
            "stage_7B_shift2",
            "stage_7C_all_lanes",
        ]

        return names[self.level]

    # --------------------------------------------------------
    # 가능한 x 위치
    # --------------------------------------------------------

    def allowed_x_positions(self):

        # ----------------------------------------------------
        # Stage 7A
        #
        # 기존 x=4에서 한 칸만 이동
        # ----------------------------------------------------

        if self.level == 0:

            return [
                3,
                4,
                5,
            ]

        # ----------------------------------------------------
        # Stage 7B
        #
        # 두 칸까지 이동
        # ----------------------------------------------------

        if self.level == 1:

            return [
                2,
                3,
                4,
                5,
                6,
            ]

        # ----------------------------------------------------
        # Stage 7C
        #
        # 양쪽 벽에 바로 붙이지 않고
        # 한 칸 여유를 둔다.
        #
        # width=12, object width=3이면:
        #
        # x = 1 ... 8
        # ----------------------------------------------------

        min_x = 1

        max_x = (
            self.map_width
            - self.target_width
            - 1
        )

        return list(
            range(
                min_x,
                max_x + 1,
            )
        )

    # --------------------------------------------------------
    # 새로운 episode scenario 생성
    # --------------------------------------------------------

    def sample(self):

        # ----------------------------------------------------
        # 40% 확률:
        #
        # 기존 Stage 6를 그대로 사용한다.
        #
        # None을 반환하면 train.py가
        # env.reset(seed=...)를 사용하면 된다.
        # ----------------------------------------------------

        if (
            random.random()
            < self.original_probability
        ):
            return None

        # ----------------------------------------------------
        # 변형 episode
        # ----------------------------------------------------

        allowed = (
            self.allowed_x_positions()
        )

        # x=4를 제외해서
        # 변형 episode에서는 실제 shift가 일어나게 한다.
        shifted_positions = [
            x
            for x in allowed
            if x != self.original_x
        ]

        # 혹시라도 비어 있으면 안전하게 원래 위치 사용
        if not shifted_positions:

            lane_x = (
                self.original_x
            )

        else:

            lane_x = random.choice(
                shifted_positions
            )

        return self.make_scenario(
            lane_x
        )

    # --------------------------------------------------------
    # 실제 scenario dictionary
    # --------------------------------------------------------

    def make_scenario(
        self,
        lane_x,
    ):

        # ====================================================
        # Target
        # ====================================================

        target = {
            "x": lane_x,
            "y": 2,
            "width": 3,
            "height": 2,
        }

        # ====================================================
        # Goal
        #
        # target과 같은 lane에 둔다.
        # ====================================================

        goal_cells = []

        for y in [
            9,
            10,
        ]:

            for x in range(
                lane_x,
                lane_x + 3,
            ):

                goal_cells.append(
                    [x, y]
                )

        # ====================================================
        # Main blocking obstacle
        #
        # Target과 같은 x에 놓는다.
        #
        # 따라서 문제 구조는 Stage 6와 같고
        # 절대 좌표만 바뀐다.
        # ====================================================

        main_obstacle = {
            "x": lane_x,
            "y": 6,
            "width": 3,
            "height": 1,
        }

        # ====================================================
        # Secondary obstacle
        #
        # lane에 겹치지 않는 쪽에 배치한다.
        #
        # target이 왼쪽이면 오른쪽,
        # target이 오른쪽이면 왼쪽.
        # ====================================================

        lane_center = (
            lane_x + 1
        )

        map_center = (
            self.map_width
            // 2
        )

        if lane_center <= map_center:

            secondary_x = 9

        else:

            secondary_x = 2

        secondary_obstacle = {
            "x": secondary_x,
            "y": 5,
            "width": 1,
            "height": 2,
        }

        # ====================================================
        # Scenario
        # ====================================================

        scenario = {

            "target": target,

            "obstacles": [
                main_obstacle,
                secondary_obstacle,
            ],

            "goal_cells":
                goal_cells,
        }

        return scenario

    # --------------------------------------------------------
    # 다음 substage
    # --------------------------------------------------------

    def advance(self):

        if self.level < 2:

            self.level += 1

            return True

        return False


# ============================================================
# Stage 7 Curriculum Manager
# ============================================================


class Stage7CurriculumManager:

    def __init__(
        self,
        success_threshold=0.85,
        window_size=100,
        min_episodes=200,
    ):

        self.success_threshold = (
            success_threshold
        )

        self.window_size = (
            window_size
        )

        self.min_episodes = (
            min_episodes
        )

        self.success_window = deque(
            maxlen=window_size
        )

        self.episodes_in_level = 0

        self.completed = False

    # --------------------------------------------------------
    # Success 기록
    # --------------------------------------------------------

    def record(
        self,
        success,
    ):

        self.success_window.append(
            float(success)
        )

        self.episodes_in_level += 1

    # --------------------------------------------------------
    # 최근 성공률
    # --------------------------------------------------------

    @property
    def success_rate(self):

        if not self.success_window:

            return 0.0

        return (
            sum(
                self.success_window
            )
            / len(
                self.success_window
            )
        )

    # --------------------------------------------------------
    # mastery 여부
    # --------------------------------------------------------

    def mastered(self):

        # 최소 학습 episode 수
        if (
            self.episodes_in_level
            < self.min_episodes
        ):
            return False

        # window가 꽉 차지 않았다면
        # 안정적이라고 보기 어렵다.
        if (
            len(
                self.success_window
            )
            < self.window_size
        ):
            return False

        return (
            self.success_rate
            >= self.success_threshold
        )

    # --------------------------------------------------------
    # 새로운 substage 시작
    # --------------------------------------------------------

    def reset_level(self):

        self.success_window.clear()

        self.episodes_in_level = 0


# ============================================================
# Left-lane 집중 fine-tuning
#
# stage_7_mastered.pt를 OOD benchmark + lane_diagnostic.py로
# 진단한 결과, x=1/x=2 lane이 완전히 실패(success=0.0)하고
# x=3도 저조(0.72)한 것으로 확인됐다. 나머지 x=4~8은 90%
# 이상으로 이미 잘 풀린다.
#
# Stage7CurriculumManager의 실패를 반복하지 않기 위해:
#   - 전체 평균 success rate가 아니라
#   - 취약한 lane(x=1, x=2, x=3)과 나머지(retention) lane을
#     "따로" 추적해서, 전부 개별적으로 기준을 넘겨야만
#     mastered로 판정한다.
# ============================================================


class LeftLaneFocusSampler:

    def __init__(
        self,
        map_width=12,
    ):

        self._scenario_maker = (
            Stage7ScenarioSampler(
                map_width=
                    map_width
            )
        )

        # 7C 전체 lane 기준으로 scenario를 만든다.
        self._scenario_maker.level = 2

        self.retention_lanes = [
            4,
            5,
            6,
            7,
            8,
        ]

        # (lane_x, 뽑힐 확률) 가중치.
        #
        # x=1, x=2: 완전 실패 -> 가장 높은 비중
        # x=3: 저조 -> 중간 비중
        # 나머지(retention): 이미 잘 풀리지만
        # catastrophic forgetting 방지를 위해 계속 섞는다.
        self.weights = {
            1: 0.25,
            2: 0.25,
            3: 0.20,
            "retention": 0.30,
        }

    @staticmethod
    def bucket_for(lane_x):

        if lane_x in (1, 2, 3):
            return lane_x

        return "retention"

    def sample(self):

        r = random.random()

        cumulative = 0.0

        for key in (
            1,
            2,
            3,
            "retention",
        ):

            cumulative += (
                self.weights[key]
            )

            if r < cumulative:

                if key == "retention":

                    lane_x = (
                        random.choice(
                            self.
                            retention_lanes
                        )
                    )

                else:

                    lane_x = key

                break

        else:

            lane_x = (
                self.retention_lanes[
                    0
                ]
            )

        scenario = (
            self._scenario_maker
            .make_scenario(
                lane_x
            )
        )

        return lane_x, scenario


class LeftLaneFocusCurriculumManager:

    def __init__(
        self,
        window_size=50,
        thresholds=None,
        min_total_episodes=800,
    ):

        self.window_size = (
            window_size
        )

        # bucket별로 다른 기준.
        #
        # x=1, x=2는 지금 0%이므로 완전 회복까지는 요구하지
        # 않고, 실용적으로 쓸만한 수준(0.70)을 목표로 한다.
        # retention은 기존 성능(0.85 이상)을 지키는 게 목표.
        self.thresholds = (
            thresholds
            or {
                1: 0.70,
                2: 0.70,
                3: 0.80,
                "retention": 0.85,
            }
        )

        self.min_total_episodes = (
            min_total_episodes
        )

        self.windows = {
            key: deque(
                maxlen=window_size
            )

            for key in (
                1,
                2,
                3,
                "retention",
            )
        }

        self.total_episodes = 0

        self.completed = False

    def record(
        self,
        lane_x,
        success,
    ):

        bucket = (
            LeftLaneFocusSampler
            .bucket_for(lane_x)
        )

        self.windows[
            bucket
        ].append(
            float(success)
        )

        self.total_episodes += 1

    def success_rate(self, bucket):

        window = self.windows[
            bucket
        ]

        if not window:
            return 0.0

        return (
            sum(window)
            / len(window)
        )

    def bucket_ready(self, bucket):

        window = self.windows[
            bucket
        ]

        if (
            len(window)
            < self.window_size
        ):
            return False

        return (
            self.success_rate(
                bucket
            )
            >= self.thresholds[
                bucket
            ]
        )

    def mastered(self):

        if (
            self.total_episodes
            < self.min_total_episodes
        ):
            return False

        return all(
            self.bucket_ready(bucket)

            for bucket in (
                1,
                2,
                3,
                "retention",
            )
        )

    def status(self):

        return {
            bucket: {
                "n": len(
                    self.windows[
                        bucket
                    ]
                ),

                "success_rate":
                    self.success_rate(
                        bucket
                    ),

                "ready":
                    self.bucket_ready(
                        bucket
                    ),
            }

            for bucket in (
                1,
                2,
                3,
                "retention",
            )
        }