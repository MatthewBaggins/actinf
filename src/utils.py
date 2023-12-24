import random

import numpy as np
import torch as t


def seed(val: float):
    random.seed(val)
    np.random.seed(val)
    t.manual_seed(val)
