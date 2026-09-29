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
