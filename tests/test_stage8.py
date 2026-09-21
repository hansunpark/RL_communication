import random

from env import CooperativeTransportEnv

from train.stages.bucketed_curriculum import (
    BucketedCurriculumManager,
)

from train.stages.stage_8.scenario import (
    RandomScenarioGenerator,
)

from train.stages.stage_8.curriculum import (
    Stage8ScenarioSampler,
)


def _cells_from_dict(rect):

    return {
        (rect["x"] + dx, rect["y"] + dy)
        for dy in range(rect["height"])
        for dx in range(rect["width"])
    }


def _cells_from_goal(goal_cells):

    return {
        tuple(cell)
        for cell in goal_cells
    }


def test_generator_default_matches_baseline():

    random.seed(0)

    generator = RandomScenarioGenerator(
        randomize_size=False,
        randomize_position=False,
        obstacle_count_range=(0, 0),
    )

    scenario = generator.sample()

    assert scenario["target"] == {
        "x": 4,
        "y": 2,
        "width": 3,
        "height": 2,
    }

    assert scenario["obstacles"] == []

    goal_cells = _cells_from_goal(
        scenario["goal_cells"]
    )

    # Same shape as target.
    assert len(goal_cells) == 6


def test_generator_goal_matches_target_shape():

    random.seed(1)

    generator = RandomScenarioGenerator(
        randomize_size=True,
        randomize_position=True,
        obstacle_count_range=(0, 3),
    )

    for _ in range(50):

        scenario = generator.sample()

        target_cells = _cells_from_dict(
            scenario["target"]
        )

        goal_cells = _cells_from_goal(
            scenario["goal_cells"]
        )

        assert len(target_cells) == len(
            goal_cells
        )

        # Target and goal never overlap.
        assert target_cells.isdisjoint(
            goal_cells
        )

        # No obstacle overlaps target or
        # goal (or another obstacle).
        occupied = (
            set(target_cells)
            | set(goal_cells)
        )

        for obstacle in scenario[
            "obstacles"
        ]:

            obstacle_cells = (
                _cells_from_dict(
                    obstacle
                )
            )

            assert obstacle_cells.isdisjoint(
                occupied
            )

            occupied |= obstacle_cells


def test_generator_position_radius_bounds_displacement():

    random.seed(4)

    anchor_x, anchor_y = 4, 2

    radius = 2

    generator = RandomScenarioGenerator(
        randomize_size=False,
        randomize_position=True,
        position_radius=radius,
        obstacle_count_range=(0, 0),
    )

    for _ in range(50):

        scenario = generator.sample()

        target = scenario["target"]

        assert (
            abs(target["x"] - anchor_x)
            <= radius
        )

        assert (
            abs(target["y"] - anchor_y)
            <= radius
        )


def test_generator_scenario_loads_in_env():

    random.seed(2)

    generator = RandomScenarioGenerator(
        randomize_size=True,
        randomize_position=True,
        obstacle_count_range=(0, 2),
    )

    env = CooperativeTransportEnv()

    for episode in range(10):

        scenario = generator.sample()

        observations, infos = (
            env.reset(
                seed=100 + episode,
                options={
                    "scenario": scenario
                },
            )
        )

        assert len(observations) == len(
            env.possible_agents
        )

        assert not env._check_success()


def test_stage8_sampler_bucket_and_advance():

    random.seed(3)

    sampler = Stage8ScenarioSampler()

    assert sampler.name == "stage_8A_size"

    scenario = sampler.sample()

    bucket = sampler.bucket_for(
        scenario
    )

    assert bucket.endswith("obs0")

    for expected in (
        "stage_8B1_position_r2",
        "stage_8B2_position_r5",
        "stage_8B3_position_full",
        "stage_8B4_size_and_position",
        "stage_8C_obstacles",
        "stage_8D_full",
    ):

        advanced = sampler.advance()

        assert advanced is True

        assert sampler.name == expected

    assert sampler.advance() is False


def test_bucketed_curriculum_requires_all_buckets():

    manager = BucketedCurriculumManager(
        threshold=0.8,
        window_size=10,
        min_total_episodes=20,
    )

    for _ in range(10):
        manager.record("a", True)

    # "b" bucket never recorded -> not mastered.
    assert manager.mastered() is False

    for _ in range(10):
        manager.record("b", False)

    # "b" bucket below threshold -> not mastered.
    assert manager.mastered() is False

    manager.reset()

    for _ in range(10):
        manager.record("a", True)

    for _ in range(10):
        manager.record("b", True)

    assert manager.mastered() is True
