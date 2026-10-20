import random

import numpy as np
import pytest
import torch

from gymnasium.spaces import flatdim

from env import CooperativeTransportEnv

from train.network import ActorCritic
from train.ppo import PPO

from train.stages.stage_8.curriculum import (
    Stage8ScenarioSampler,
)

from train.stages.stage_8_comm.curriculum import (
    Stage8CommSampler,
)

from train.stages.comm_scratch.curriculum import (
    LEVELS,
    CommScratchSampler,
)

from train.stages.comm_scratch.scenario import (
    ScratchScenarioGenerator,
)


MAP_SIZE = 12


def _target_pushable_off_walls(scenario):
    """
    target이 벽에 붙어 있지 않은지 확인한다 (벽에 붙으면 벽에서
    멀어지는 방향으로 못 밀어서 풀 수 없는 episode가 된다).
    """

    target = scenario["target"]

    return (
        target["x"] >= 1
        and target["y"] >= 1
        and target["x"] + target["width"]
        <= MAP_SIZE - 1
        and target["y"] + target["height"]
        <= MAP_SIZE - 1
    )


def _goal_matches_target_shape(scenario):

    target = scenario["target"]

    goal_cells = {
        tuple(cell)
        for cell in scenario["goal_cells"]
    }

    goal_x = min(x for x, _ in goal_cells)
    goal_y = min(y for _, y in goal_cells)

    expected = {
        (goal_x + dx, goal_y + dy)
        for dy in range(target["height"])
        for dx in range(target["width"])
    }

    return goal_cells == expected


# ------------------------------------------------------------
# Stage 8 생성기 버그 수정 (벽에 붙은 target)
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "level",
    range(
        1,
        len(Stage8ScenarioSampler.LEVEL_NAMES),
    ),
)
def test_stage8_target_never_touches_wall(level):

    random.seed(level)

    sampler = Stage8ScenarioSampler()

    sampler.level = level

    for _ in range(300):
        assert _target_pushable_off_walls(
            sampler.sample()
        )


# ------------------------------------------------------------
# comm_scratch 생성기
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "level_index",
    range(len(LEVELS)),
)
def test_scratch_levels_generate_valid_scenarios(
    level_index,
):
    random.seed(level_index)

    sampler = CommScratchSampler()

    sampler.level = level_index

    config = LEVELS[level_index]

    env = CooperativeTransportEnv(
        num_agents=4,
    )

    for episode in range(40):

        scenario = sampler.sample()

        assert _target_pushable_off_walls(
            scenario
        )

        assert _goal_matches_target_shape(
            scenario
        )

        if config["target_size"] is not None:
            assert (
                scenario["target"]["width"],
                scenario["target"]["height"],
            ) == config["target_size"]

        low_high = config[
            "goal_distance_range"
        ]

        if low_high is not None:

            goal_x = min(
                c[0]
                for c in scenario["goal_cells"]
            )

            goal_y = min(
                c[1]
                for c in scenario["goal_cells"]
            )

            distance = (
                abs(
                    goal_x
                    - scenario["target"]["x"]
                )
                + abs(
                    goal_y
                    - scenario["target"]["y"]
                )
            )

            assert (
                low_high[0]
                <= distance
                <= low_high[1]
            )

        # split spawn은 target 위치에 따라 불가능할 수 있고,
        # 러너가 재시도한다. 나머지는 항상 reset 가능해야 한다.
        try:
            env.reset(
                seed=episode,
                options={
                    "scenario": scenario
                },
            )
        except RuntimeError:
            assert (
                config["spawn"]["type"]
                == "split"
            )
            continue

        assert not env._check_success()

        assert isinstance(
            sampler.bucket_for(scenario),
            str,
        )


def test_scratch_near_spawn_agents_see_target():

    random.seed(0)

    generator = ScratchScenarioGenerator(
        target_size=(3, 2),
        goal_distance_range=(2, 5),
        spawn={
            "type": "near",
            "radius": 2,
        },
    )

    env = CooperativeTransportEnv(
        num_agents=4,
    )

    for _ in range(50):

        observations, _ = env.reset(
            options={
                "scenario":
                    generator.sample()
            }
        )

        for observation in (
            observations.values()
        ):
            assert (
                observation["local_map"]
                == 4
            ).any()


def test_scratch_help_penalty_zero_until_goal_leaves_vision():

    sampler = CommScratchSampler()

    penalties = []

    while True:

        penalties.append(
            sampler.env_settings()[
                "help_penalty"
            ]
        )

        if not sampler.advance():
            break

    first_positive = next(
        index
        for index, penalty
        in enumerate(penalties)
        if penalty > 0
    )

    assert (
        LEVELS[first_positive]["name"]
        == "S4_goal_d2-8"
    )

    assert all(
        penalty == 0
        for penalty
        in penalties[:first_positive]
    )

    assert all(
        penalty > 0
        for penalty
        in penalties[first_positive:]
    )


# ------------------------------------------------------------
# stage_8_comm sampler
# ------------------------------------------------------------

def test_stage8_comm_sampler_levels():

    sampler = Stage8CommSampler()

    assert (
        sampler.name
        == "stage_8B0sL_lateral_side_spawn_leftbias_comm"
    )

    # 왼쪽 편향 발판만 episode를 짧게 끊는다.
    assert (
        sampler.env_settings()["max_steps"]
        == 60
    )

    random.seed(0)

    env = CooperativeTransportEnv(
        num_agents=4,
    )

    # 8B0s: target과 goal이 같은 행, 옆으로만 1~2칸 (dx=0 없음),
    # 2명은 goal 반대편 옆면에 서 있어서 바로 밀 수 있다.
    # 일부는 "아래로 밀기"를 잊지 않도록 8A 과제(goal이 바로 아래)다.
    shifts = set()

    below_count = 0

    for episode in range(200):

        scenario = sampler.sample()

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        shifts.add(
            sampler.bucket_for(scenario)
        )

        if scenario.get("variant") == "below":

            below_count += 1

            assert scenario["target"]["x"] == 4
            assert scenario["target"]["y"] == 2

            assert min(
                cell[1]
                for cell in scenario["goal_cells"]
            ) > 2

            continue

        assert scenario["target"] == {
            "x": 4,
            "y": 2,
            "width": 3,
            "height": 2,
        }

        assert _goal_matches_target_shape(
            scenario
        )

        goal_x = min(
            cell[0]
            for cell in scenario["goal_cells"]
        )

        assert min(
            cell[1]
            for cell in scenario["goal_cells"]
        ) == 2

        push_x = 3 if goal_x > 4 else 7

        positions = {
            tuple(position)
            for position in scenario[
                "agent_positions"
            ].values()
        }

        assert {
            (push_x, 2),
            (push_x, 3),
        } <= positions

    # 왼쪽 편향 발판의 below_mix = 0.3
    assert 30 <= below_count <= 90

    assert shifts == {
        "goal_below",
        "goal_dx-2",
        "goal_dx-1",
        "goal_dx+1",
        "goal_dx+2",
    }

    # 8B0s: 좌우 균등, 기본 max_steps
    sampler.advance()

    assert (
        sampler.env_settings()["max_steps"]
        == 200
    )

    # 8B0: 같은 배치, 옆면 스폰 없음
    sampler.advance()

    for _ in range(20):
        assert (
            "agent_positions"
            not in sampler.sample()
        )

    # 8B0a: target 기본 위치, goal은 아래 + 옆으로 +-1
    sampler.advance()

    shifts = set()

    for _ in range(200):

        scenario = sampler.sample()

        assert scenario["target"] == {
            "x": 4,
            "y": 2,
            "width": 3,
            "height": 2,
        }

        shifts.add(
            sampler.bucket_for(scenario)
        )

    assert shifts == {
        "goal_dx-1",
        "goal_dx+0",
        "goal_dx+1",
    }

    sampler.level = 0

    names = [sampler.name]

    while sampler.advance():
        names.append(sampler.name)

    assert names[:13] == [
        "stage_8B0sL_lateral_side_spawn_leftbias_comm",
        "stage_8B0s_lateral_side_spawn_comm",
        "stage_8B0_lateral_only_comm",
        "stage_8B0a_goal_lateral1_comm",
        "stage_8B0b_goal_lateral2_comm",
        "stage_8B0uS2_up_narrow_side_spawn_comm",
        "stage_8B0uS_up_side_spawn_comm",
        "stage_8B0u_up_comm",
        "stage_8B0uLs_up_long_side_spawn_comm",
        "stage_8B0uL_up_long_comm",
        "stage_8B0tS_tall_side_spawn_comm",
        "stage_8B0t_tall_lateral_comm",
        "stage_8B1_position_r2_comm",
    ]

    assert names[-3:] == [
        "stage_9_split_3near_comm",
        "stage_9_split_2near_comm",
        "stage_9_split_1near_comm",
    ]

    random.seed(0)

    scenario = sampler.sample()

    assert (
        scenario["spawn_mode"][
            "near_count"
        ]
        == 1
    )

    assert sampler.bucket_for(
        scenario
    ).endswith("_near1")


# ------------------------------------------------------------
# PPO: 이동/통신 entropy 분리
# ------------------------------------------------------------

def test_evaluate_actions_matches_get_action_and_value():

    torch.manual_seed(0)

    network = ActorCritic(
        obs_dim=10,
    )

    obs = torch.randn(8, 10)

    (
        action,
        comm_action,
        log_prob,
        entropy,
        value,
    ) = network.get_action_and_value(obs)

    (
        log_prob_2,
        move_entropy,
        comm_entropy,
        value_2,
    ) = network.evaluate_actions(
        obs,
        action,
        comm_action,
    )

    assert torch.allclose(
        log_prob,
        log_prob_2,
    )

    assert torch.allclose(
        entropy,
        move_entropy + comm_entropy,
    )

    assert torch.allclose(
        value,
        value_2,
    )


def test_comm_entropy_coef_defaults_to_entropy_coef():

    ppo = PPO(
        obs_dim=10,
        entropy_coef=0.03,
    )

    assert ppo.comm_entropy_coef == 0.03

    ppo = PPO(
        obs_dim=10,
        entropy_coef=0.005,
        comm_entropy_coef=0.01,
    )

    assert ppo.entropy_coef == 0.005
    assert ppo.comm_entropy_coef == 0.01


def test_act_batch_shapes():

    env = CooperativeTransportEnv(
        num_agents=4,
    )

    obs_dim = flatdim(
        env.observation_space(
            env.possible_agents[0]
        )
    )

    ppo = PPO(obs_dim=obs_dim)

    (
        actions,
        comm_actions,
        log_probs,
        values,
    ) = ppo.act_batch(
        np.zeros(
            (6, obs_dim),
            dtype=np.float32,
        )
    )

    assert actions.shape == (6,)
    assert comm_actions.shape == (6,)
    assert log_probs.shape == (6,)
    assert values.shape == (6,)

    assert set(
        comm_actions.tolist()
    ) <= {0, 1}


def test_stage8_comm_lateral_levels_raise_entropy_only():

    sampler = Stage8CommSampler()

    coefs = []

    while True:

        coefs.append(
            sampler.train_settings().get(
                "entropy_coef"
            )
        )

        if not sampler.advance():
            break

    assert coefs[:12] == [0.03] * 12

    raised_after_lateral = [
        name
        for name, coef in zip(
            Stage8CommSampler.LEVEL_NAMES[12:],
            coefs[12:],
        )
        if coef is not None
    ]

    # 가로/위로 밀기 레벨 뒤에서는 짧은 발판(8C0s)만 올린다 (8C0a를
    # 0.03으로 돌렸다가 붕괴했다).
    assert raised_after_lateral == [
        "stage_8C0s1_clear_obstacle_small_target_comm",
        "stage_8C0s_clear_obstacle_scaffold_comm",
    ]


def test_stage8_comm_narrow_up_push_scaffold():

    random.seed(0)

    sampler = Stage8CommSampler()

    sampler.level = sampler.LEVEL_NAMES.index(
        "stage_8B0uS2_up_narrow_side_spawn_comm"
    )

    assert (
        sampler.env_settings()["max_steps"]
        == 60
    )

    env = CooperativeTransportEnv(
        num_agents=4,
    )

    buckets = set()

    for episode in range(100):

        scenario = sampler.sample()

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        buckets.add(
            sampler.bucket_for(scenario)
        )

        if scenario.get("variant") == "below":
            continue

        assert scenario["target"]["width"] == 2

        assert _goal_matches_target_shape(
            scenario
        )

        positions = {
            tuple(p)
            for p in scenario[
                "agent_positions"
            ].values()
        }

        # 2x2 target (4,2)의 아래 면
        assert {(4, 4), (5, 4)} <= positions

    assert buckets == {
        "goal_below",
        "goal_up1",
    }


def test_stage8_comm_up_push_levels():

    random.seed(0)

    sampler = Stage8CommSampler()

    sampler.level = sampler.LEVEL_NAMES.index(
        "stage_8B0uS_up_side_spawn_comm"
    )

    assert (
        sampler.env_settings()["max_steps"]
        == 60
    )

    env = CooperativeTransportEnv(
        num_agents=4,
    )

    buckets = set()

    for episode in range(100):

        scenario = sampler.sample()

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        buckets.add(
            sampler.bucket_for(scenario)
        )

        if scenario.get("variant") == "below":
            continue

        positions = {
            tuple(p)
            for p in scenario[
                "agent_positions"
            ].values()
        }

        # 3x2 target (4,2)의 아래 면
        assert {
            (4, 4),
            (5, 4),
            (6, 4),
        } <= positions

    assert buckets == {
        "goal_below",
        "goal_up1",
        "goal_up2",
    }

    sampler.advance()

    assert (
        sampler.name
        == "stage_8B0u_up_comm"
    )

    for _ in range(20):
        assert (
            "agent_positions"
            not in sampler.sample()
        )

    sampler.advance()

    assert (
        sampler.name
        == "stage_8B0uLs_up_long_side_spawn_comm"
    )


def test_stage8_comm_long_up_push_random_positions():

    random.seed(0)

    sampler = Stage8CommSampler()

    sampler.level = sampler.LEVEL_NAMES.index(
        "stage_8B0uLs_up_long_side_spawn_comm"
    )

    env = CooperativeTransportEnv(
        num_agents=4,
    )

    xs = set()
    ys = set()
    buckets = set()

    for episode in range(300):

        scenario = sampler.sample()

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        buckets.add(
            sampler.bucket_for(scenario)
        )

        if scenario.get("variant") == "below":
            continue

        target = scenario["target"]

        xs.add(target["x"])
        ys.add(target["y"])

        assert _target_pushable_off_walls(
            scenario
        )

        assert _goal_matches_target_shape(
            scenario
        )

        positions = {
            tuple(p)
            for p in scenario[
                "agent_positions"
            ].values()
        }

        bottom = target["y"] + target["height"]

        assert {
            (target["x"] + cx, bottom)
            for cx in range(target["width"])
        } <= positions

    # target 위치가 실제로 흩어져 있다 (좌표 암기 방지)
    assert len(xs) >= 6
    assert len(ys) >= 5

    assert buckets == {
        "goal_below",
        "goal_up1",
        "goal_up2",
        "goal_up3",
    }


def test_scratch_entropy_raised_only_for_new_coordination_levels():

    sampler = CommScratchSampler()

    raised = []

    while True:

        if "entropy_coef" in sampler.train_settings():
            raised.append(sampler.name)

        if not sampler.advance():
            break

    assert raised == [
        "S3s_tall_side_push_scaffold",
        "S3t_tall_side_push_near",
        "S10a_one_small_obstacle_near",
        "S10a1_clear_obstacle_small_target_scaffold",
        "S10a2_clear_obstacle_in_path_scaffold",
        "S10a3a_clear_thin_obstacle_scaffold",
        "S10a3b_clear_big_obstacle_scaffold",
    ]


def test_scratch_tall_side_push_scaffold():

    random.seed(0)

    sampler = CommScratchSampler()

    while sampler.name != (
        "S3s_tall_side_push_scaffold"
    ):
        sampler.advance()

    assert (
        sampler.env_settings()["max_steps"]
        == 60
    )

    env = CooperativeTransportEnv(
        num_agents=4,
    )

    buckets = set()

    for episode in range(100):

        scenario = sampler.sample()

        target = scenario["target"]

        assert target["height"] == 3

        goal_x = min(
            c[0] for c in scenario["goal_cells"]
        )

        goal_y = min(
            c[1] for c in scenario["goal_cells"]
        )

        assert goal_y == target["y"]

        push_x = (
            target["x"] - 1
            if goal_x > target["x"]
            else target["x"] + target["width"]
        )

        positions = {
            tuple(p)
            for p in scenario[
                "agent_positions"
            ].values()
        }

        assert {
            (push_x, target["y"] + dy)
            for dy in range(3)
        } <= positions

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        buckets.add(
            sampler.bucket_for(scenario)
        )

    assert any(
        b.endswith("_push_left")
        for b in buckets
    )

    assert any(
        b.endswith("_push_right")
        for b in buckets
    )


# ------------------------------------------------------------
# PPO KL 조기 종료
# ------------------------------------------------------------

def _filled_buffer(ppo, steps=64):

    from train.rollout_buffer import (
        MultiAgentRolloutBuffer,
    )

    buffer = MultiAgentRolloutBuffer(
        device="cpu"
    )

    rng = np.random.default_rng(0)

    for step in range(steps):

        observation = rng.normal(
            size=10
        ).astype(np.float32)

        action, comm_action, log_prob, value = (
            ppo.act(observation)
        )

        buffer.add(
            agent="agent_0",
            observation=observation,
            action=action,
            comm_action=comm_action,
            log_prob=log_prob,
            reward=float(action == 1),
            done=False,
            value=value,
            next_value=0.0,
        )

    return buffer


def test_ppo_target_kl_stops_update_early():

    torch.manual_seed(0)

    ppo = PPO(
        obs_dim=10,
        learning_rate=1e-1,
        minibatch_size=8,
        update_epochs=10,
        target_kl=1e-6,
    )

    metrics = ppo.update(
        _filled_buffer(ppo),
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert metrics["kl_early_stop"] == 1.0

    assert "approx_kl" in metrics


def test_ppo_without_target_kl_runs_all_epochs():

    torch.manual_seed(0)

    ppo = PPO(
        obs_dim=10,
        learning_rate=1e-1,
        minibatch_size=8,
        update_epochs=3,
    )

    metrics = ppo.update(
        _filled_buffer(ppo),
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert metrics["kl_early_stop"] == 0.0


def test_stage8_comm_tall_lateral_scaffold():

    random.seed(0)

    sampler = Stage8CommSampler()

    sampler.level = sampler.LEVEL_NAMES.index(
        "stage_8B0tS_tall_side_spawn_comm"
    )

    env = CooperativeTransportEnv(
        num_agents=4,
    )

    buckets = set()

    for episode in range(400):

        scenario = sampler.sample()

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        buckets.add(
            sampler.bucket_for(scenario)
        )

        if scenario.get("variant") == "below":
            continue

        target = scenario["target"]

        assert target["height"] == 3

        assert _target_pushable_off_walls(
            scenario
        )

        assert _goal_matches_target_shape(
            scenario
        )

        goal_x = min(
            c[0] for c in scenario["goal_cells"]
        )

        goal_y = min(
            c[1] for c in scenario["goal_cells"]
        )

        assert goal_y == target["y"]

        assert 1 <= abs(goal_x - target["x"]) <= 2

        push_x = (
            target["x"] - 1
            if goal_x > target["x"]
            else target["x"] + target["width"]
        )

        positions = {
            tuple(p)
            for p in scenario[
                "agent_positions"
            ].values()
        }

        assert {
            (push_x, target["y"] + dy)
            for dy in range(3)
        } <= positions

    assert buckets == {
        "goal_below",
        "2x3_left",
        "2x3_right",
        "3x3_left",
        "3x3_right",
        "4x3_left",
        "4x3_right",
    }


def test_scratch_obstacle_levels_introduce_one_obstacle_first():

    random.seed(0)

    sampler = CommScratchSampler()

    names = list(sampler.LEVEL_NAMES)

    assert names.index("S10a_one_small_obstacle_near") < names.index(
        "S10b_small_obstacle_goal_any"
    ) < names.index(
        "S10c_small_obstacle_random_spawn"
    ) < names.index(
        "S10d_one_obstacle_random_spawn"
    ) < names.index("S10_obstacles_0-2")

    sampler.level = names.index(
        "S10a_one_small_obstacle_near"
    )

    for _ in range(100):

        scenario = sampler.sample()

        assert len(scenario["obstacles"]) == 1

        obstacle = scenario["obstacles"][0]

        assert obstacle["width"] <= 2
        assert obstacle["height"] == 1

        assert "agent_positions" in scenario


def test_stage8_comm_obstacle_intro_levels():

    random.seed(0)

    sampler = Stage8CommSampler()

    names = list(sampler.LEVEL_NAMES)

    assert names.index(
        "stage_8B4_size_and_position_comm"
    ) + 1 == names.index(
        "stage_8C0s1_clear_obstacle_small_target_comm"
    )

    assert names.index(
        "stage_8C0s1_clear_obstacle_small_target_comm"
    ) + 1 == names.index(
        "stage_8C0s_clear_obstacle_scaffold_comm"
    )

    assert names.index(
        "stage_8C0s_clear_obstacle_scaffold_comm"
    ) + 1 == names.index(
        "stage_8C0a_one_small_obstacle_comm"
    )

    assert names.index(
        "stage_8C0a_one_small_obstacle_comm"
    ) + 1 == names.index(
        "stage_8C0a2_one_small_obstacle_goal_any_comm"
    )

    assert names.index(
        "stage_8C0a2_one_small_obstacle_goal_any_comm"
    ) + 1 == names.index(
        "stage_8C0b_one_obstacle_comm"
    )

    assert names.index(
        "stage_8C0b_one_obstacle_comm"
    ) + 1 == names.index(
        "stage_8C_obstacles_comm"
    )

    env = CooperativeTransportEnv(
        num_agents=4,
    )

    sampler.level = names.index(
        "stage_8C0a_one_small_obstacle_comm"
    )

    for episode in range(100):

        scenario = sampler.sample()

        assert len(scenario["obstacles"]) == 1

        assert scenario["obstacles"][0]["width"] <= 2
        assert scenario["obstacles"][0]["height"] == 1

        # 쉬운 기반: goal은 target에서 2-5칸
        goal_x = min(
            c[0] for c in scenario["goal_cells"]
        )

        goal_y = min(
            c[1] for c in scenario["goal_cells"]
        )

        assert 2 <= (
            abs(goal_x - scenario["target"]["x"])
            + abs(goal_y - scenario["target"]["y"])
        ) <= 5

        assert _target_pushable_off_walls(
            scenario
        )

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        assert sampler.bucket_for(
            scenario
        ).endswith("_obs1")

    # 기존 Stage 8 레벨은 같은 Stage 8 생성기 레벨을 그대로 쓴다.
    sampler.level = names.index(
        "stage_8C_obstacles_comm"
    )

    counts = {
        len(sampler.sample()["obstacles"])
        for _ in range(200)
    }

    assert counts <= {0, 1, 2, 3}
    assert 3 in counts


def test_scratch_clear_obstacle_scaffold_blocks_the_path():

    random.seed(0)

    sampler = CommScratchSampler()

    sampler.level = sampler.LEVEL_NAMES.index(
        "S10a2_clear_obstacle_in_path_scaffold"
    )

    assert (
        sampler.env_settings()["max_steps"]
        == 100
    )

    env = CooperativeTransportEnv(
        num_agents=4,
        reward_mode="obstacle_shaping",
    )

    for episode in range(100):

        scenario = sampler.sample()

        target = scenario["target"]

        goal_x = min(
            c[0] for c in scenario["goal_cells"]
        )

        goal_y = min(
            c[1] for c in scenario["goal_cells"]
        )

        # 같은 행 또는 같은 열
        assert (
            goal_x == target["x"]
            or goal_y == target["y"]
        )

        assert len(scenario["obstacles"]) == 1

        obstacle = scenario["obstacles"][0]

        assert (
            obstacle["width"],
            obstacle["height"],
        ) == (1, 1)

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        # env 기준으로도 obstacle이 경로를 막고 있다
        assert env._obstacle_blocking_score() > 0


def test_stage8_comm_clear_obstacle_scaffold():

    random.seed(0)

    sampler = Stage8CommSampler()

    sampler.level = sampler.LEVEL_NAMES.index(
        "stage_8C0s_clear_obstacle_scaffold_comm"
    )

    assert (
        sampler.env_settings()["max_steps"]
        == 100
    )

    env = CooperativeTransportEnv(
        num_agents=4,
        reward_mode="obstacle_shaping",
    )

    for episode in range(100):

        scenario = sampler.sample()

        assert "agent_positions" in scenario

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        assert env._obstacle_blocking_score() > 0


@pytest.mark.parametrize(
    "sampler_cls,level_name",
    [
        (
            CommScratchSampler,
            "S10a1_clear_obstacle_small_target_scaffold",
        ),
        (
            Stage8CommSampler,
            "stage_8C0s1_clear_obstacle_small_target_comm",
        ),
    ],
)
def test_small_target_clear_obstacle_scaffold(
    sampler_cls,
    level_name,
):
    random.seed(0)

    sampler = sampler_cls()

    sampler.level = sampler.LEVEL_NAMES.index(
        level_name
    )

    env = CooperativeTransportEnv(
        num_agents=4,
        reward_mode="obstacle_shaping",
    )

    for episode in range(60):

        scenario = sampler.sample()

        target = scenario["target"]

        assert target["width"] == 2
        assert target["height"] <= 2

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        assert env._obstacle_blocking_score() > 0


def test_scratch_big_obstacle_in_path_scaffold():

    random.seed(0)

    sampler = CommScratchSampler()

    sampler.level = sampler.LEVEL_NAMES.index(
        "S10a3b_clear_big_obstacle_scaffold"
    )

    env = CooperativeTransportEnv(
        num_agents=4,
        reward_mode="obstacle_shaping",
    )

    sizes = set()

    for episode in range(150):

        scenario = sampler.sample()

        assert len(scenario["obstacles"]) == 1

        obstacle = scenario["obstacles"][0]

        sizes.add(
            (obstacle["width"], obstacle["height"])
        )

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        assert env._obstacle_blocking_score() > 0

    # 1x1만이 아니라 큰 obstacle도 나온다
    assert any(
        w * h > 1 for w, h in sizes
    )


def test_scratch_thin_obstacle_scaffold_shapes():

    random.seed(1)

    sampler = CommScratchSampler()

    sampler.level = sampler.LEVEL_NAMES.index(
        "S10a3a_clear_thin_obstacle_scaffold"
    )

    for _ in range(200):

        obstacle = sampler.sample()[
            "obstacles"
        ][0]

        assert (
            obstacle["width"]
            * obstacle["height"]
            <= 3
        )

        # 벽에 붙지 않는다
        assert obstacle["x"] > 0
        assert obstacle["y"] > 0
        assert (
            obstacle["x"] + obstacle["width"]
            < 12
        )
        assert (
            obstacle["y"] + obstacle["height"]
            < 12
        )


def test_scratch_two_obstacle_scaffold():

    random.seed(2)

    sampler = CommScratchSampler()

    sampler.level = sampler.LEVEL_NAMES.index(
        "S10f_two_obstacles_scaffold"
    )

    env = CooperativeTransportEnv(
        num_agents=4,
        reward_mode="obstacle_shaping",
    )

    for episode in range(100):

        scenario = sampler.sample()

        assert len(scenario["obstacles"]) == 2

        env.reset(
            seed=episode,
            options={"scenario": scenario},
        )

        assert env._obstacle_blocking_score() > 0


def test_scratch_two_small_obstacles_scaffold():

    random.seed(3)

    sampler = CommScratchSampler()

    sampler.level = sampler.LEVEL_NAMES.index(
        "S10f1_two_small_obstacles_scaffold"
    )

    for _ in range(100):

        scenario = sampler.sample()

        assert len(scenario["obstacles"]) == 2

        for obstacle in scenario["obstacles"]:
            assert (
                obstacle["width"]
                * obstacle["height"]
                <= 3
            )


def test_overall_threshold_with_bucket_floor():

    from train.stages.bucketed_curriculum import (
        BucketedCurriculumManager,
    )

    manager = BucketedCurriculumManager(
        threshold=0.6,
        window_size=10,
        min_total_episodes=0,
        overall_threshold=0.85,
        overall_window=40,
    )

    # 쉬운 버킷 30개 전부 성공, 어려운 버킷 10개 중 7개 성공
    for _ in range(30):
        manager.record("easy", True)

    for index in range(10):
        manager.record("hard", index < 7)

    # 전체 37/40 = 0.925 >= 0.85, 어려운 버킷 0.7 >= 0.6
    assert manager.mastered()

    # 어려운 버킷이 하한 아래로 떨어지면 통과하지 못한다
    for _ in range(10):
        manager.record("hard", False)

    assert not manager.mastered()


def test_scratch_final_levels_use_relaxed_mastery():

    sampler = CommScratchSampler()

    relaxed = []

    while True:

        settings = sampler.train_settings()

        if "overall_threshold" in settings:
            relaxed.append(sampler.name)

            assert settings[
                "bucket_threshold"
            ] == pytest.approx(0.6)

        if not sampler.advance():
            break

    assert relaxed == [
        "S10e_obstacles_0-2_random_spawn",
        "S10_obstacles_0-2",
        "S11a_small_obstacles_0-4",
        "S11b_obstacles_0-3",
        "S11_obstacles_0-4",
    ]
