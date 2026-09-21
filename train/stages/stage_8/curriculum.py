from .scenario import RandomScenarioGenerator


# ============================================================
# Stage 8 Scenario Sampler
#
# Stage 8은 통신 없이(comm은 항상 NO_MESSAGE) 공간 일반화만
# 검증한다. 핵심 원칙: 한 번에 새로운 축을 하나씩만 도입한다
# (Stage 0-6도 obstacle을 스폰 완전 랜덤화가 끝난 뒤에야
# 추가했던 것과 동일한 원칙).
#
#   8A : target 크기만 랜덤화 (위치는 기본값 고정)
#   8B1: 크기는 다시 기본값(3x2)으로 고정하고, 위치만
#        기본 위치 기준 반경 2 안에서 랜덤화
#   8B2: 반경 5로 확대 (크기는 계속 고정)
#   8B3: 완전 랜덤 위치 (반경 제한 없음, 크기는 계속 고정)
#   8B4: 8A(크기)와 8B3(위치)를 합친다 — 크기 랜덤 + 위치
#        완전 랜덤을 동시에 (각각 따로는 이미 통과했으니
#        재결합만 검증)
#   8C : obstacle 개수/크기/위치까지 랜덤화
#   8D : obstacle 개수 범위를 더 넓힘
#
# ------------------------------------------------------------
# 실패 기록:
#
# 1차 시도 - 8B를 "완전 랜덤 위치" 단일 단계로 통째로 학습시켰다가
# 실패. warm-start된 stage_7 policy가 이미 저엔트로피 상태라
# 새로운 위치를 전혀 탐색하지 못하고 200 step 내내 target을 한
# 번도 못 건드려서(성공률 0.0 고정) 학습 신호 자체가 없었다.
# -> 반경을 점진적으로 늘리는 8B1/8B2/8B3로 쪼갬.
#
# 2차 시도 - 8B1을 "크기 랜덤 + 위치 랜덤(반경 2)"로 만들었다가
# 다시 정체. large 크기 버킷은 1700+ episode 동안 0%, small
# 버킷도 5~8%에서 못 벗어났다. 원인: 8A가 이미 "크기"라는 새
# 축을 도입했는데, 8B1이 "위치"라는 또 다른 새 축을 동시에
# 더해서 두 축이 한꺼번에 바뀌었다. -> 8B1~8B3에서는 크기를
# 다시 고정값으로 되돌리고 위치만 새 축으로 다루도록 수정하고,
# 8B4에서 둘을 재결합하는 단계를 추가.
# ============================================================


class Stage8ScenarioSampler:

    LEVEL_NAMES = [
        "stage_8A_size",
        "stage_8B1_position_r2",
        "stage_8B2_position_r5",
        "stage_8B3_position_full",
        "stage_8B4_size_and_position",
        "stage_8C_obstacles",
        "stage_8D_full",
    ]

    # 각 레벨의 position_radius. None = 반경 제한 없음(완전 랜덤).
    POSITION_RADIUS = [
        None,  # 8A: randomize_position=False라 안 쓰인다.
        2,     # 8B1
        5,     # 8B2
        None,  # 8B3
        None,  # 8B4
        None,  # 8C
        None,  # 8D
    ]

    # 각 레벨에서 target 크기를 랜덤화할지 여부.
    RANDOMIZE_SIZE = [
        True,   # 8A
        False,  # 8B1 (위치만 새로 도입)
        False,  # 8B2
        False,  # 8B3
        True,   # 8B4 (크기 + 위치 재결합)
        True,   # 8C
        True,   # 8D
    ]

    def __init__(
        self,
        map_width=12,
        map_height=12,
    ):
        self.map_width = map_width
        self.map_height = map_height

        self.level = 0

        # bucket_for()에서 거리 버킷 계산에 쓰는 기준점.
        # RandomScenarioGenerator.default_target과 반드시
        # 일치해야 한다.
        self.anchor_x = 4
        self.anchor_y = 2

    @property
    def name(self):

        return self.LEVEL_NAMES[
            self.level
        ]

    def _generator(self):

        position_radius = (
            self.POSITION_RADIUS[
                self.level
            ]
        )

        randomize_size = (
            self.RANDOMIZE_SIZE[
                self.level
            ]
        )

        if self.level == 0:

            return RandomScenarioGenerator(
                map_width=self.map_width,
                map_height=self.map_height,
                randomize_size=True,
                randomize_position=False,
                obstacle_count_range=(0, 0),
            )

        if self.level in (1, 2, 3, 4):

            return RandomScenarioGenerator(
                map_width=self.map_width,
                map_height=self.map_height,
                randomize_size=randomize_size,
                randomize_position=True,
                position_radius=
                    position_radius,
                obstacle_count_range=(0, 0),
            )

        if self.level == 5:

            return RandomScenarioGenerator(
                map_width=self.map_width,
                map_height=self.map_height,
                randomize_size=True,
                randomize_position=True,
                position_radius=None,
                obstacle_count_range=(0, 3),
            )

        return RandomScenarioGenerator(
            map_width=self.map_width,
            map_height=self.map_height,
            randomize_size=True,
            randomize_position=True,
            position_radius=None,
            obstacle_count_range=(0, 4),
        )

    def sample(self):

        return self._generator().sample(
            name=self.name
        )

    # --------------------------------------------------------
    # 성공률을 독립적으로 추적할 버킷 키.
    #
    # (target 크기, target이 기본 위치에서 얼마나 멀리
    # 떨어졌는지, obstacle 개수)로 이름 붙인다. 거리 버킷을
    # 넣은 이유: 8B* 안에서도 "멀리 떨어진 경우만 계속 실패"가
    # 평균에 가려질 수 있다 (Stage 7 lane_left에서 실제로
    # 있었던 문제와 동일한 함정).
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

        distance = (
            abs(
                target["x"]
                - self.anchor_x
            )
            + abs(
                target["y"]
                - self.anchor_y
            )
        )

        if distance <= 2:
            distance_bucket = "d0-2"
        elif distance <= 5:
            distance_bucket = "d3-5"
        else:
            distance_bucket = "d6+"

        obstacle_count = len(
            scenario.get(
                "obstacles",
                [],
            )
        )

        return (
            f"{size_bucket}_"
            f"{distance_bucket}_"
            f"obs{obstacle_count}"
        )

    def advance(self):

        if (
            self.level
            < len(self.LEVEL_NAMES) - 1
        ):
            self.level += 1

            return True

        return False
