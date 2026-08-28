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

        value = (
            self.critic(
                features
            )
            .squeeze(-1)
        )

        return logits, value

    def get_action_and_value(
        self,
        obs,
        action=None,
    ):

        logits, value = (
            self(obs)
        )

        distribution = (
            Categorical(
                logits=logits
            )
        )

        if action is None:

            action = (
                distribution.sample()
            )

        return (
            action,

            distribution.log_prob(
                action
            ),

            distribution.entropy(),

            value,
        )