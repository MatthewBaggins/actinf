from src.envs import make_envs, observe_envs


envs10 = make_envs(10)


def test_make_envs():
    assert envs10.shape == (10, 3)


def test_observe_envs1():
    assert observe_envs(envs10, 1).shape == (10, 2)


def test_observe_envs2():
    assert observe_envs(envs10, 2).shape == (10, 2)
