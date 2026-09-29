from env import CooperativeTransportEnv

from train.stages.stage_9.scenario import (
    make_stage9_scenario,
)

from train.stages.stage_9.curriculum import (
    Stage9LevelSampler,
)


def test_make_stage9_scenario_shape():

    scenario = make_stage9_scenario(
        near_count=2,
        near_radius=2,
        far_min_distance=6,
        name="test",
    )

    assert scenario["target"] == {
        "x": 4,
        "y": 2,
        "width": 3,
        "height": 2,
    }

    assert len(scenario["goal_cells"]) == 6
    assert scenario["obstacles"] == []

    assert scenario["spawn_mode"] == {
        "type": "split",
        "near_count": 2,
        "near_radius": 2,
        "far_min_distance": 6,
    }


def test_stage9_sampler_progression_easy_to_hard():

    sampler = Stage9LevelSampler()

    assert sampler.name == "stage_9A_3near_1far"

    assert sampler.advance() is True
    assert sampler.name == "stage_9B_2near_2far"

    assert sampler.advance() is True
    assert sampler.name == "stage_9C_1near_3far"

    assert sampler.advance() is False


def test_stage9_scenario_split_spawn_counts_in_env():

    sampler = Stage9LevelSampler(
        near_radius=2,
        far_min_distance=6,
    )

    env = CooperativeTransportEnv()

    for level_index, expected_near in enumerate(
        (3, 2, 1)
    ):

        sampler.level = level_index

        scenario = sampler.sample()

        env.reset(
            seed=level_index,
            options={
                "scenario": scenario
            },
        )

        target = env._target_object()
        reference_cells = target.cells()

        def distance(pos):
            x, y = pos
            return min(
                abs(x - rx) + abs(y - ry)
                for rx, ry in reference_cells
            )

        distances = sorted(
            distance(pos)
            for pos in env.agent_positions.values()
        )

        near_count = sum(
            1 for d in distances if d <= 2
        )

        far_count = sum(
            1 for d in distances if d >= 6
        )

        assert near_count == expected_near
        assert far_count == 4 - expected_near
