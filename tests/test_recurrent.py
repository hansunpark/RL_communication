import json
import random

import numpy as np
import pytest
import torch

from train.network import (
    ActorCritic,
    RecurrentActorCritic,
)

from train.ppo import PPO

from train.recurrent_buffer import (
    RecurrentRolloutBuffer,
)

from train.recurrent_ppo import (
    RecurrentPPO,
)


OBS_DIM = 10


def _mlp_and_recurrent():

    torch.manual_seed(0)

    mlp = ActorCritic(
        obs_dim=OBS_DIM,
    )

    recurrent = RecurrentActorCritic(
        obs_dim=OBS_DIM,
    )

    result = recurrent.load_state_dict(
        mlp.state_dict(),
        strict=False,
    )

    # 새로 생기는 것은 GRU와 memory_proj뿐이어야 한다.
    assert set(result.missing_keys) == {
        "rnn.weight_ih",
        "rnn.weight_hh",
        "rnn.bias_ih",
        "rnn.bias_hh",
        "memory_proj.weight",
        "memory_proj.bias",
    }

    assert not result.unexpected_keys

    return mlp, recurrent


# ------------------------------------------------------------
# 네트워크
# ------------------------------------------------------------

def test_zero_init_memory_matches_mlp_exactly():

    mlp, recurrent = _mlp_and_recurrent()

    obs = torch.randn(7, 5, OBS_DIM)

    starts = torch.zeros(7, 5)

    hidden = torch.randn(
        5,
        recurrent.rnn_hidden_dim,
    )

    logits, comm_logits, values = (
        recurrent.unroll(
            obs,
            hidden,
            starts,
        )
    )

    mlp_logits, mlp_comm, mlp_values = mlp(
        obs.reshape(-1, OBS_DIM)
    )

    assert torch.allclose(
        logits.reshape(-1, 5),
        mlp_logits,
        atol=1e-6,
    )

    assert torch.allclose(
        comm_logits.reshape(-1, 2),
        mlp_comm,
        atol=1e-6,
    )

    assert torch.allclose(
        values.reshape(-1),
        mlp_values,
        atol=1e-6,
    )


def test_unroll_matches_step_by_step_with_resets():

    torch.manual_seed(1)

    network = RecurrentActorCritic(
        obs_dim=OBS_DIM,
    )

    # 메모리가 실제로 출력에 영향을 주게 한다.
    torch.nn.init.normal_(
        network.memory_proj.weight,
        std=0.5,
    )

    steps, batch = 6, 3

    obs = torch.randn(
        steps,
        batch,
        OBS_DIM,
    )

    starts = torch.zeros(steps, batch)
    starts[0] = 1.0
    starts[3, 1] = 1.0

    hidden0 = torch.randn(
        batch,
        network.rnn_hidden_dim,
    )

    logits, _, values = network.unroll(
        obs,
        hidden0,
        starts,
    )

    hidden = hidden0

    for t in range(steps):

        step_logits, _, step_value, hidden = (
            network.step(
                obs[t],
                hidden,
                starts[t],
            )
        )

        assert torch.allclose(
            logits[t],
            step_logits,
            atol=1e-5,
        )

        assert torch.allclose(
            values[t],
            step_value,
            atol=1e-5,
        )


def test_memory_changes_output_after_reset():

    torch.manual_seed(2)

    network = RecurrentActorCritic(
        obs_dim=OBS_DIM,
    )

    torch.nn.init.normal_(
        network.memory_proj.weight,
        std=0.5,
    )

    obs = torch.randn(1, OBS_DIM)

    hidden = torch.randn(
        1,
        network.rnn_hidden_dim,
    )

    kept, _, _, _ = network.step(
        obs,
        hidden,
        torch.zeros(1),
    )

    reset, _, _, _ = network.step(
        obs,
        hidden,
        torch.ones(1),
    )

    fresh, _, _, _ = network.step(
        obs,
        network.initial_hidden(1),
        torch.zeros(1),
    )

    # episode 시작이면 이전 hidden을 버린다.
    assert torch.allclose(reset, fresh)
    assert not torch.allclose(kept, reset)


# ------------------------------------------------------------
# RecurrentPPO
# ------------------------------------------------------------

def _collect(
    ppo,
    steps=8,
    rows=4,
    seed=0,
):
    rng = np.random.default_rng(seed)

    buffer = RecurrentRolloutBuffer(
        device="cpu"
    )

    hidden = ppo.initial_hidden(rows)

    starts = np.ones(
        rows,
        dtype=np.float32,
    )

    for _ in range(steps):

        obs = rng.normal(
            size=(rows, OBS_DIM)
        ).astype(np.float32)

        (
            actions,
            comm_actions,
            log_probs,
            values,
            new_hidden,
        ) = ppo.act_batch_recurrent(
            obs,
            hidden,
            starts,
        )

        dones = (
            rng.random(rows) < 0.2
        ).astype(np.float32)

        buffer.add_step(
            observations=obs,
            hidden_in=hidden,
            episode_starts=starts,
            actions=actions,
            comm_actions=comm_actions,
            log_probs=log_probs,
            values=values,
            rewards=(actions == 1).astype(
                np.float32
            ),
            dones=dones,
        )

        hidden = new_hidden

        starts = dones

    last_values = ppo.value_batch_recurrent(
        rng.normal(
            size=(rows, OBS_DIM)
        ).astype(np.float32),
        hidden,
        starts,
    )

    return buffer, last_values


def test_replayed_chunks_reproduce_rollout_log_probs():
    """
    저장한 hidden_in에서 시퀀스 조각을 다시 펼치면 rollout 때와
    같은 log_prob이 나와야 한다 (첫 epoch의 ratio가 1).
    """

    torch.manual_seed(3)

    ppo = RecurrentPPO(
        obs_dim=OBS_DIM,
        bptt_length=4,
    )

    torch.nn.init.normal_(
        ppo.network.memory_proj.weight,
        std=0.5,
    )

    buffer, last_values = _collect(ppo)

    data = buffer.compute(last_values)

    sequences, length = ppo._chunk(data)

    assert length == 4

    with torch.no_grad():

        logits, comm_logits, _ = (
            ppo.network.unroll(
                sequences["observations"]
                .transpose(0, 1),
                sequences["hidden_start"],
                sequences["episode_starts"]
                .transpose(0, 1),
            )
        )

    log_prob = (
        torch.distributions.Categorical(
            logits=logits
        ).log_prob(
            sequences["actions"]
            .transpose(0, 1)
        )
        + torch.distributions.Categorical(
            logits=comm_logits
        ).log_prob(
            sequences["comm_actions"]
            .transpose(0, 1)
        )
    )

    assert torch.allclose(
        log_prob,
        sequences["log_probs"]
        .transpose(0, 1),
        atol=1e-5,
    )


def test_gae_cuts_at_episode_end():

    buffer = RecurrentRolloutBuffer(
        device="cpu"
    )

    hidden = torch.zeros(1, 4)

    for reward, done in (
        (1.0, 1.0),
        (0.0, 0.0),
    ):
        buffer.add_step(
            observations=np.zeros(
                (1, OBS_DIM)
            ),
            hidden_in=hidden,
            episode_starts=[0.0],
            actions=[0],
            comm_actions=[0],
            log_probs=[0.0],
            values=[0.0],
            rewards=[reward],
            dones=[done],
        )

    data = buffer.compute(
        last_values=[5.0],
        gamma=0.9,
        gae_lambda=1.0,
    )

    # 첫 step은 거기서 끝났으므로 뒤의 value(5.0)가 섞이면 안 된다.
    assert data["returns"][0, 0].item() == (
        pytest.approx(1.0)
    )

    assert data["returns"][1, 0].item() == (
        pytest.approx(0.9 * 5.0)
    )


def test_update_recurrent_runs_and_kl_stop():

    torch.manual_seed(4)

    ppo = RecurrentPPO(
        obs_dim=OBS_DIM,
        bptt_length=4,
        minibatch_size=8,
        update_epochs=3,
    )

    buffer, last_values = _collect(ppo)

    before = [
        p.detach().clone()
        for p in ppo.network.parameters()
    ]

    metrics = ppo.update_recurrent(
        buffer,
        last_values,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert metrics["kl_early_stop"] == 0.0

    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            before,
            ppo.network.parameters(),
        )
    )

    tight = RecurrentPPO(
        obs_dim=OBS_DIM,
        bptt_length=4,
        minibatch_size=8,
        update_epochs=10,
        learning_rate=1e-1,
        target_kl=1e-6,
    )

    buffer, last_values = _collect(tight)

    metrics = tight.update_recurrent(
        buffer,
        last_values,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert metrics["kl_early_stop"] == 1.0


def test_critic_warmup_only_moves_critic():

    torch.manual_seed(5)

    ppo = RecurrentPPO(
        obs_dim=OBS_DIM,
        bptt_length=4,
        minibatch_size=8,
    )

    buffer, last_values = _collect(ppo)

    before = {
        name: p.detach().clone()
        for name, p
        in ppo.network.named_parameters()
    }

    ppo.update_recurrent(
        buffer,
        last_values,
        gamma=0.99,
        gae_lambda=0.95,
        update_policy=False,
    )

    for name, p in (
        ppo.network.named_parameters()
    ):
        changed = not torch.equal(
            before[name],
            p,
        )

        assert changed == name.startswith(
            "critic."
        ), name


def test_mlp_checkpoint_loads_into_recurrent_ppo(
    tmp_path,
):
    torch.manual_seed(6)

    mlp = PPO(obs_dim=OBS_DIM)

    path = tmp_path / "mlp.pt"

    mlp.save(str(path))

    recurrent = RecurrentPPO(
        obs_dim=OBS_DIM,
    )

    recurrent.load(
        str(path),
        load_optimizer=True,
    )

    obs = torch.randn(4, OBS_DIM)

    logits, comm, value, _ = (
        recurrent.network.step(
            obs,
            recurrent.initial_hidden(4),
            torch.zeros(4),
        )
    )

    mlp_logits, mlp_comm, mlp_value = (
        mlp.network(obs)
    )

    assert torch.allclose(
        logits,
        mlp_logits,
        atol=1e-6,
    )

    assert torch.allclose(
        value,
        mlp_value,
        atol=1e-6,
    )


# ------------------------------------------------------------
# 러너 (MLP / GRU 둘 다 끝까지 도는지)
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "recurrent",
    [False, True],
)
def test_runner_stop_and_resume(
    tmp_path,
    monkeypatch,
    recurrent,
):
    from train.stages.comm_runner import (
        RunConfig,
        train,
    )

    from train.stages.comm_scratch.curriculum import (
        CommScratchSampler,
    )

    from train.stages.comm_scratch.train import (
        ENV_KWARGS,
    )

    monkeypatch.chdir(tmp_path)

    random.seed(0)

    def config():

        return RunConfig(
            run_name="smoke",
            num_envs=2,
            rollout_steps=8,
            bptt_length=4,
            minibatch_size=16,
            success_threshold=0.0,
            window_size=2,
            min_episodes_per_level=2,
            ablation_episodes=2,
            print_interval_episodes=100000,
            env_kwargs=dict(
                ENV_KWARGS,
                max_steps=20,
            ),
            device="cpu",
            torch_threads=1,
            recurrent=recurrent,
            stop_after_current_level=True,
        )

    result = train(
        config(),
        CommScratchSampler(),
    )

    assert result == (
        "stopped after S0_push_2x1_d2-3"
    )

    with open(
        "checkpoints/smoke_latest.json",
        encoding="utf-8",
    ) as file:
        assert json.load(file)["level"] == 1

    assert (
        tmp_path
        / "evaluation_results"
        / "smoke"
        / "ablation_S0_push_2x1_d2-3_mastered.json"
    ).exists()

    result = train(
        config(),
        CommScratchSampler(),
        resume=True,
    )

    assert result == (
        "stopped after S1_goal_d2-5"
    )
