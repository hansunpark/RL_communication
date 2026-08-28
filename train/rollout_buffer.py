from collections import (
    defaultdict,
)

import numpy as np
import torch


class MultiAgentRolloutBuffer:

    def __init__(
        self,
        device,
    ):

        self.device = device

        self.reset()

    def reset(self):

        self.trajectories = (
            defaultdict(list)
        )

        self.total_size = 0

    def __len__(self):

        return self.total_size

    def add(
        self,
        agent,
        observation,
        action,
        log_prob,
        reward,
        done,
        value,
        next_value,
    ):

        self.trajectories[
            agent
        ].append(
            {
                "observation":
                    np.asarray(
                        observation,
                        dtype=np.float32,
                    ),

                "action":
                    int(action),

                "log_prob":
                    float(log_prob),

                "reward":
                    float(reward),

                "done":
                    float(done),

                "value":
                    float(value),

                "next_value":
                    float(
                        next_value
                    ),
            }
        )

        self.total_size += 1

    def compute(
        self,
        gamma=0.99,
        gae_lambda=0.95,
    ):

        all_observations = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []

        for (
            agent,
            trajectory,
        ) in (
            self.trajectories.items()
        ):

            n = len(
                trajectory
            )

            advantages = (
                np.zeros(
                    n,
                    dtype=np.float32,
                )
            )

            last_gae = 0.0

            for t in reversed(
                range(n)
            ):

                transition = (
                    trajectory[t]
                )

                done = (
                    transition["done"]
                )

                reward = (
                    transition["reward"]
                )

                value = (
                    transition["value"]
                )

                next_value = (
                    transition[
                        "next_value"
                    ]
                )

                delta = (
                    reward
                    +
                    gamma
                    * next_value
                    * (1.0 - done)
                    -
                    value
                )

                last_gae = (
                    delta
                    +
                    gamma
                    * gae_lambda
                    * (1.0 - done)
                    * last_gae
                )

                advantages[t] = (
                    last_gae
                )

            for (
                transition,
                advantage,
            ) in zip(
                trajectory,
                advantages,
            ):

                all_observations.append(
                    transition[
                        "observation"
                    ]
                )

                all_actions.append(
                    transition[
                        "action"
                    ]
                )

                all_log_probs.append(
                    transition[
                        "log_prob"
                    ]
                )

                all_advantages.append(
                    advantage
                )

                all_returns.append(
                    advantage
                    +
                    transition[
                        "value"
                    ]
                )

        observations = (
            torch.tensor(
                np.asarray(
                    all_observations
                ),

                dtype=torch.float32,

                device=self.device,
            )
        )

        actions = torch.tensor(
            all_actions,

            dtype=torch.long,

            device=self.device,
        )

        old_log_probs = (
            torch.tensor(
                all_log_probs,

                dtype=torch.float32,

                device=self.device,
            )
        )

        advantages = (
            torch.tensor(
                all_advantages,

                dtype=torch.float32,

                device=self.device,
            )
        )

        returns = torch.tensor(
            all_returns,

            dtype=torch.float32,

            device=self.device,
        )

        return (
            observations,
            actions,
            old_log_probs,
            advantages,
            returns,
        )