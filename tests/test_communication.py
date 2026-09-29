import numpy as np
import torch

from env import CooperativeTransportEnv
from env.constants import STAY, HELP, NO_MESSAGE

from train.network import ActorCritic
from train.ppo import PPO
from train.rollout_buffer import MultiAgentRolloutBuffer

from evaluation.evaluator import PolicyEvaluator


def _action(physical, message=NO_MESSAGE):

    return np.array(
        [physical, message],
        dtype=np.int64,
    )


def test_actor_critic_dual_head_shapes():

    network = ActorCritic(
        obs_dim=10,
        num_actions=5,
        num_comm_actions=2,
        hidden_dim=16,
    )

    obs = torch.zeros((3, 10))

    logits, comm_logits, value = network(obs)

    assert logits.shape == (3, 5)
    assert comm_logits.shape == (3, 2)
    assert value.shape == (3,)


def test_actor_critic_joint_log_prob():

    network = ActorCritic(
        obs_dim=10,
        num_actions=5,
        num_comm_actions=2,
        hidden_dim=16,
    )

    obs = torch.zeros((4, 10))

    (
        action,
        comm_action,
        log_prob,
        entropy,
        value,
    ) = network.get_action_and_value(obs)

    assert action.shape == (4,)
    assert comm_action.shape == (4,)

    # Joint log_prob/entropy must equal the sum of the two
    # independent categorical heads.
    logits, comm_logits, _ = network(obs)

    from torch.distributions import Categorical

    expected_log_prob = (
        Categorical(logits=logits).log_prob(action)
        + Categorical(logits=comm_logits).log_prob(
            comm_action
        )
    )

    assert torch.allclose(
        log_prob,
        expected_log_prob,
        atol=1e-6,
    )


def test_ppo_act_returns_movement_and_comm():

    ppo = PPO(obs_dim=10, num_actions=5, device="cpu")

    observation = np.zeros(10, dtype=np.float32)

    action, comm_action, log_prob, value = ppo.act(
        observation
    )

    assert 0 <= action < 5
    assert 0 <= comm_action < 2
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_ppo_update_with_comm_actions():

    ppo = PPO(
        obs_dim=10,
        num_actions=5,
        device="cpu",
        minibatch_size=8,
    )

    buffer = MultiAgentRolloutBuffer(device="cpu")

    observation = np.zeros(10, dtype=np.float32)

    for _ in range(16):

        action, comm_action, log_prob, value = ppo.act(
            observation
        )

        buffer.add(
            agent="agent_0",
            observation=observation,
            action=action,
            comm_action=comm_action,
            log_prob=log_prob,
            reward=0.0,
            done=False,
            value=value,
            next_value=0.0,
        )

    metrics = ppo.update(
        buffer,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


def test_ppo_load_old_checkpoint_without_comm_head(
    tmp_path,
):

    # Simulate a checkpoint saved before the comm head
    # existed: save a network's state dict, then strip the
    # comm_actor keys before loading into a fresh PPO
    # instance (mirrors loading stage_8A_size_mastered.pt,
    # which predates communication).
    old_ppo = PPO(obs_dim=10, num_actions=5, device="cpu")

    state_dict = old_ppo.network.state_dict()

    stripped = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("comm_actor")
    }

    checkpoint_path = tmp_path / "old_checkpoint.pt"

    torch.save(
        {"network": stripped},
        checkpoint_path,
    )

    new_ppo = PPO(obs_dim=10, num_actions=5, device="cpu")

    # Must not raise, and must leave comm_actor at its
    # freshly initialized values (not crash / not silently
    # skip the rest of the network).
    new_ppo.load(
        str(checkpoint_path),
        load_optimizer=False,
    )

    assert torch.equal(
        new_ppo.network.encoder[0].weight,
        old_ppo.network.encoder[0].weight,
    )


def test_env_help_penalty_charged_only_to_sender():

    env = CooperativeTransportEnv(
        curriculum_stage=0,
        help_penalty=0.05,
    )

    env.reset(seed=1)

    actions = {
        agent: _action(STAY)
        for agent in env.agents
    }

    actions["agent_0"] = _action(STAY, HELP)

    _, rewards, _, _, _ = env.step(actions)

    assert (
        rewards["agent_0"]
        ==
        rewards["agent_1"] - 0.05
    )


def test_evaluator_communication_ablation_runs():

    from evaluation.benchmark_scenarios import (
        IID_SCENARIOS,
    )

    evaluator_with_comm = PolicyEvaluator(
        checkpoint="checkpoints/stage_8A_size_mastered.pt",
        device="cpu",
        action_mode="deterministic",
        disable_communication=False,
    )

    evaluator_no_comm = PolicyEvaluator(
        checkpoint="checkpoints/stage_8A_size_mastered.pt",
        device="cpu",
        action_mode="deterministic",
        disable_communication=True,
    )

    for evaluator in (
        evaluator_with_comm,
        evaluator_no_comm,
    ):

        result = evaluator.run_episode(
            scenario=IID_SCENARIOS[0],
            seed=123,
        )

        assert "success" in result
