from src.env import make_envs


def test_make_envs():
    envs = make_envs(10)
    assert envs.shape == (10, 3)
