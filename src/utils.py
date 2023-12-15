import random
import torch as t


def seed(val: float):
    random.seed(val)
    t.manual_seed(val)
