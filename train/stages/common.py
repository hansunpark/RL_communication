import random

import numpy as np
import torch

from gymnasium.spaces import flatten


# ==========================================================
# 모든 stage 학습 스크립트가 공유하는 helper.
#
# stage_0_6/train.py, stage_7/train.py 그리고 앞으로 추가될
# stage_8, stage_9, stage_10 학습 스크립트에서 동일하게 쓴다.
# ==========================================================


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def flatten_obs(
    space,
    observation,
):

    return np.asarray(
        flatten(
            space,
            observation,
        ),

        dtype=np.float32,
    )
