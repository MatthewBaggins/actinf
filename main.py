import torch as t

from src.env import make_envs, observe_envs
from src.utils import seed
from src.thermostat import Thermostat

SEED = 42
N_ENVS = 10


def main() -> None:
    seed(SEED)
    envs = make_envs(N_ENVS)
    model = Thermostat()
    obs = observe_envs(envs, 0)
    logits = model(obs)


if __name__ == "__main__":
    main()
