import numpy as np

import torch
import torch.nn as nn

from torch.distributions import (
    Categorical,
)

from .network import (
    RecurrentActorCritic,
)

from .ppo import PPO


# ============================================================
# RecurrentPPO
#
# PPO와 같은 손실(clip, 이동/통신 entropy 분리, target_kl 조기 종료,
# critic warmup)을 쓰되, 정책이 GRU hidden state를 들고 다닌다.
#
# 학습: rollout을 길이 bptt_length의 시퀀스 조각으로 나누고, 각 조각을
# rollout 때 저장한 hidden(hidden_in[t0])에서 다시 펼친다 (truncated
# BPTT). update당 gradient step 수와 minibatch 크기(transition 수)는
# MLP PPO와 같게 맞춘다: minibatch 하나 = minibatch_size // bptt_length
# 개의 조각.
#
# save/load는 PPO 것을 그대로 쓴다. MLP 체크포인트를 불러오면 rnn과
# memory_proj만 새로 초기화되고, memory_proj가 0이라 처음에는 MLP와
# 똑같이 행동한다.
# ============================================================


class RecurrentPPO(PPO):

    recurrent = True

    def __init__(
        self,
        obs_dim,
        rnn_hidden_dim=128,
        bptt_length=32,
        **kwargs,
    ):
        # _build_network가 super().__init__ 안에서 불리므로 먼저 둔다.
        self.rnn_hidden_dim = (
            rnn_hidden_dim
        )

        self.bptt_length = bptt_length

        super().__init__(
            obs_dim,
            **kwargs,
        )

    def _build_network(
        self,
        obs_dim,
        num_actions,
        hidden_dim,
    ):
        return RecurrentActorCritic(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            rnn_hidden_dim=
                self.rnn_hidden_dim,
        )

    def initial_hidden(
        self,
        batch_size,
    ):
        return self.network.initial_hidden(
            batch_size
        )

    def _as_tensor(
        self,
        array,
    ):
        return torch.as_tensor(
            array,
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def act_batch_recurrent(
        self,
        observations,
        hidden,
        episode_starts,
    ):
        """
        (actions, comm_actions, log_probs, values) numpy 배열과
        new_hidden 텐서를 반환한다.
        """

        (
            logits,
            comm_logits,
            value,
            new_hidden,
        ) = self.network.step(
            self._as_tensor(observations),
            hidden,
            self._as_tensor(episode_starts),
        )

        distribution = Categorical(
            logits=logits
        )

        comm_distribution = Categorical(
            logits=comm_logits
        )

        action = distribution.sample()

        comm_action = (
            comm_distribution.sample()
        )

        log_prob = (
            distribution.log_prob(action)
            + comm_distribution.log_prob(
                comm_action
            )
        )

        return (
            action.cpu().numpy(),
            comm_action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
            new_hidden,
        )

    @torch.no_grad()
    def value_batch_recurrent(
        self,
        observations,
        hidden,
        episode_starts,
    ):
        _, _, value, _ = (
            self.network.step(
                self._as_tensor(
                    observations
                ),
                hidden,
                self._as_tensor(
                    episode_starts
                ),
            )
        )

        return value.cpu().numpy()

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def _chunk(
        self,
        data,
    ):
        """
        [T, N, ...] 텐서를 길이 L 조각 [C*N, L, ...]로 자른다.
        T가 L로 나누어떨어지지 않으면 L=T(조각 하나)로 둔다.
        """

        steps, rows = (
            data["actions"].shape
        )

        length = (
            self.bptt_length
            if steps
            % self.bptt_length
            == 0
            else steps
        )

        chunks = steps // length

        def split(tensor):

            shape = tensor.shape[2:]

            return (
                tensor.reshape(
                    chunks,
                    length,
                    rows,
                    *shape,
                )
                .transpose(1, 2)
                .reshape(
                    chunks * rows,
                    length,
                    *shape,
                )
            )

        sequences = {
            name: split(tensor)
            for name, tensor
            in data.items()
            if name != "hidden_in"
        }

        # 각 조각 첫 step 직전의 hidden
        sequences["hidden_start"] = (
            split(data["hidden_in"])[
                :, 0
            ]
        )

        return sequences, length

    def update_recurrent(
        self,
        buffer,
        last_values,
        gamma,
        gae_lambda,
        update_policy=True,
    ):
        data = buffer.compute(
            last_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        if update_policy:

            advantages = data["advantages"]

            if advantages.numel() > 1:
                data["advantages"] = (
                    advantages
                    - advantages.mean()
                ) / (
                    advantages.std()
                    + 1e-8
                )

        sequences, length = self._chunk(
            data
        )

        num_sequences = (
            sequences["actions"].shape[0]
        )

        sequences_per_batch = max(
            1,
            self.minibatch_size
            // length,
        )

        indices = np.arange(
            num_sequences
        )

        policy_losses = []
        value_losses = []
        entropies = []
        move_entropies = []
        comm_entropies = []
        approx_kls = []

        stop_early = False

        for _ in range(
            self.update_epochs
        ):

            if stop_early:
                break

            np.random.shuffle(indices)

            for start in range(
                0,
                num_sequences,
                sequences_per_batch,
            ):
                batch_idx = indices[
                    start:
                    start
                    + sequences_per_batch
                ]

                def seq(name):
                    # [B, L, ...] -> [L, B, ...]
                    return (
                        sequences[name][
                            batch_idx
                        ]
                        .transpose(0, 1)
                    )

                returns = seq("returns")

                if not update_policy:

                    # critic warmup: 메모리를 더한 feature는
                    # 고정하고 critic head만 맞춘다.
                    with torch.no_grad():
                        features = (
                            self.network
                            .unroll_features(
                                seq(
                                    "observations"
                                ),
                                sequences[
                                    "hidden_start"
                                ][batch_idx],
                                seq(
                                    "episode_starts"
                                ),
                            )
                        )

                    value = (
                        self.network.critic(
                            features
                        )
                        .squeeze(-1)
                    )

                    value_loss = (
                        0.5
                        * (value - returns)
                        .pow(2)
                        .mean()
                    )

                    self.optimizer.zero_grad()

                    value_loss.backward()

                    self.optimizer.step()

                    value_losses.append(
                        value_loss.item()
                    )

                    continue

                (
                    logits,
                    comm_logits,
                    new_value,
                ) = self.network.unroll(
                    seq("observations"),
                    sequences[
                        "hidden_start"
                    ][batch_idx],
                    seq("episode_starts"),
                )

                distribution = Categorical(
                    logits=logits
                )

                comm_distribution = (
                    Categorical(
                        logits=comm_logits
                    )
                )

                new_log_prob = (
                    distribution.log_prob(
                        seq("actions")
                    )
                    + comm_distribution
                    .log_prob(
                        seq("comm_actions")
                    )
                )

                log_ratio = (
                    new_log_prob
                    - seq("log_probs")
                )

                ratio = torch.exp(
                    log_ratio
                )

                with torch.no_grad():
                    approx_kl = (
                        (ratio - 1.0)
                        - log_ratio
                    ).mean().item()

                approx_kls.append(
                    approx_kl
                )

                if (
                    self.target_kl
                    is not None
                    and approx_kl
                    > 1.5 * self.target_kl
                ):
                    stop_early = True

                    break

                batch_adv = seq(
                    "advantages"
                )

                policy_loss = torch.max(
                    -batch_adv * ratio,
                    -batch_adv
                    * torch.clamp(
                        ratio,
                        1.0
                        - self.clip_coef,
                        1.0
                        + self.clip_coef,
                    ),
                ).mean()

                value_loss = (
                    0.5
                    * (
                        new_value
                        - returns
                    )
                    .pow(2)
                    .mean()
                )

                move_entropy_mean = (
                    distribution
                    .entropy()
                    .mean()
                )

                comm_entropy_mean = (
                    comm_distribution
                    .entropy()
                    .mean()
                )

                total_loss = (
                    policy_loss
                    + self.value_coef
                    * value_loss
                    - self.entropy_coef
                    * move_entropy_mean
                    - self.comm_entropy_coef
                    * comm_entropy_mean
                )

                self.optimizer.zero_grad()

                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                policy_losses.append(
                    policy_loss.item()
                )

                value_losses.append(
                    value_loss.item()
                )

                move_entropies.append(
                    move_entropy_mean.item()
                )

                comm_entropies.append(
                    comm_entropy_mean.item()
                )

                entropies.append(
                    move_entropies[-1]
                    + comm_entropies[-1]
                )

        def mean(values):

            return (
                float(np.mean(values))
                if values
                else 0.0
            )

        return {
            "policy_loss": mean(
                policy_losses
            ),
            "value_loss": mean(
                value_losses
            ),
            "entropy": mean(entropies),
            "move_entropy": mean(
                move_entropies
            ),
            "comm_entropy": mean(
                comm_entropies
            ),
            "approx_kl": mean(
                approx_kls
            ),
            "kl_early_stop": float(
                stop_early
            ),
        }
