import torch
import torch.nn as nn

from torch.distributions import (
    Categorical,
)


class ActorCritic(
    nn.Module
):

    def __init__(
        self,
        obs_dim,
        num_actions=5,
        num_comm_actions=2,
        hidden_dim=256,
    ):

        super().__init__()

        self.encoder = (
            nn.Sequential(
                nn.Linear(
                    obs_dim,
                    hidden_dim,
                ),

                nn.Tanh(),

                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),

                nn.Tanh(),
            )
        )

        self.actor = nn.Linear(
            hidden_dim,
            num_actions,
        )

        # 통신 head (NO_MESSAGE / HELP, env.constants 참고).
        # 이동 head와 별도의 categorical distribution으로 다룬다.
        self.comm_actor = nn.Linear(
            hidden_dim,
            num_comm_actions,
        )

        self.critic = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        obs,
    ):

        features = (
            self.encoder(
                obs
            )
        )

        logits = (
            self.actor(
                features
            )
        )

        comm_logits = (
            self.comm_actor(
                features
            )
        )

        value = (
            self.critic(
                features
            )
            .squeeze(-1)
        )

        return logits, comm_logits, value

    def get_action_and_value(
        self,
        obs,
        action=None,
        comm_action=None,
    ):

        logits, comm_logits, value = (
            self(obs)
        )

        distribution = (
            Categorical(
                logits=logits
            )
        )

        comm_distribution = (
            Categorical(
                logits=comm_logits
            )
        )

        if action is None:

            action = (
                distribution.sample()
            )

        if comm_action is None:

            comm_action = (
                comm_distribution.sample()
            )

        log_prob = (
            distribution.log_prob(
                action
            )
            +
            comm_distribution.log_prob(
                comm_action
            )
        )

        entropy = (
            distribution.entropy()
            +
            comm_distribution.entropy()
        )

        return (
            action,
            comm_action,
            log_prob,
            entropy,
            value,
        )

    def evaluate_actions(
        self,
        obs,
        action,
        comm_action,
    ):
        """
        get_action_and_value(obs, action, comm_action)와
        같은 log_prob/value를 주지만, entropy를 이동/통신
        head별로 따로 반환한다. PPO가 두 head에 서로 다른
        entropy 계수를 걸 수 있게 하기 위함.
        """

        logits, comm_logits, value = (
            self(obs)
        )

        distribution = (
            Categorical(
                logits=logits
            )
        )

        comm_distribution = (
            Categorical(
                logits=comm_logits
            )
        )

        log_prob = (
            distribution.log_prob(
                action
            )
            +
            comm_distribution.log_prob(
                comm_action
            )
        )

        return (
            log_prob,
            distribution.entropy(),
            comm_distribution.entropy(),
            value,
        )


class RecurrentActorCritic(
    nn.Module
):
    """
    ActorCritic + 잔차 GRU 메모리.

    encoder feature f_t에 GRU 출력을 더한 f_t + W h_t를 세 head
    (actor, comm_actor, critic)에 넣는다.

    W(memory_proj)를 0으로 초기화하므로, ActorCritic 체크포인트를
    불러온 직후에는 기존 MLP 정책과 출력이 완전히 같다. 그 상태에서
    학습하면서 메모리를 쓰는 법을 배운다. encoder / actor /
    comm_actor / critic 이름을 ActorCritic과 똑같이 맞춰서, MLP
    체크포인트가 load_state_dict(strict=False)로 그대로 들어온다
    (새로 생기는 것은 rnn, memory_proj뿐).

    도입 이유: 메모리 없는 MLP는 시야(5x5) 밖으로 나간 target/goal과
    직전 step에만 유효한 HELP를 기억하지 못한다. obstacle 치우기처럼
    여러 단계가 필요한 레벨(comm_scratch S10a2, stage8_comm 8C0s1)에서
    두 트랙 모두 정체했다. docs/curriculum_plan.md 참고.
    """

    def __init__(
        self,
        obs_dim,
        num_actions=5,
        num_comm_actions=2,
        hidden_dim=256,
        rnn_hidden_dim=128,
    ):

        super().__init__()

        self.rnn_hidden_dim = (
            rnn_hidden_dim
        )

        self.encoder = (
            nn.Sequential(
                nn.Linear(
                    obs_dim,
                    hidden_dim,
                ),

                nn.Tanh(),

                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),

                nn.Tanh(),
            )
        )

        self.rnn = nn.GRUCell(
            hidden_dim,
            rnn_hidden_dim,
        )

        self.memory_proj = nn.Linear(
            rnn_hidden_dim,
            hidden_dim,
        )

        nn.init.zeros_(
            self.memory_proj.weight
        )

        nn.init.zeros_(
            self.memory_proj.bias
        )

        self.actor = nn.Linear(
            hidden_dim,
            num_actions,
        )

        self.comm_actor = nn.Linear(
            hidden_dim,
            num_comm_actions,
        )

        self.critic = nn.Linear(
            hidden_dim,
            1,
        )

    def initial_hidden(
        self,
        batch_size,
    ):

        return torch.zeros(
            batch_size,
            self.rnn_hidden_dim,
            device=(
                self.memory_proj.weight
                .device
            ),
        )

    def _heads(
        self,
        features,
    ):

        return (
            self.actor(features),
            self.comm_actor(features),
            self.critic(features)
            .squeeze(-1),
        )

    def step(
        self,
        obs,
        hidden,
        episode_starts,
    ):
        """
        한 step. obs [B, D], hidden [B, H], episode_starts [B]
        (1이면 그 행의 hidden을 0으로 초기화한 뒤 진행).

        (logits, comm_logits, value, new_hidden)을 반환한다.
        """

        features = self.encoder(obs)

        hidden = hidden * (
            1.0 - episode_starts
        ).unsqueeze(-1)

        new_hidden = self.rnn(
            features,
            hidden,
        )

        features = (
            features
            + self.memory_proj(
                new_hidden
            )
        )

        logits, comm_logits, value = (
            self._heads(features)
        )

        return (
            logits,
            comm_logits,
            value,
            new_hidden,
        )

    def unroll_features(
        self,
        obs,
        hidden,
        episode_starts,
    ):
        """
        시퀀스를 펼친다. obs [T, B, D], hidden [B, H] (첫 step
        직전의 hidden), episode_starts [T, B].

        메모리를 더한 feature [T, B, hidden_dim]을 반환한다.
        """

        steps, batch = obs.shape[:2]

        features = self.encoder(
            obs.reshape(
                steps * batch,
                -1,
            )
        ).reshape(
            steps,
            batch,
            -1,
        )

        memories = []

        for t in range(steps):

            hidden = hidden * (
                1.0 - episode_starts[t]
            ).unsqueeze(-1)

            hidden = self.rnn(
                features[t],
                hidden,
            )

            memories.append(hidden)

        return (
            features
            + self.memory_proj(
                torch.stack(memories)
            )
        )

    def unroll(
        self,
        obs,
        hidden,
        episode_starts,
    ):
        """
        unroll_features + head. (logits [T, B, A],
        comm_logits [T, B, C], values [T, B])를 반환한다.
        """

        return self._heads(
            self.unroll_features(
                obs,
                hidden,
                episode_starts,
            )
        )
