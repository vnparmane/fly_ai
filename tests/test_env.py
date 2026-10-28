"""Tests for Space Invaders environment and baseline controllers."""

import numpy as np
import pytest

from flybrain.control.heuristic import HeuristicController
from flybrain.control.random_ctrl import RandomController
from flybrain.env.invaders import InvadersConfig, SpaceInvadersEnv


def test_env_determinism():
    env1 = SpaceInvadersEnv()
    env2 = SpaceInvadersEnv()

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    for k in obs1:
        assert np.allclose(obs1[k], obs2[k])

    action = np.array([0, 1, 1], dtype=np.int32)
    for _ in range(50):
        o1, r1, t1, tr1, _ = env1.step(action)
        o2, r2, t2, tr2, _ = env2.step(action)
        assert r1 == r2
        assert t1 == t2
        for k in o1:
            assert np.allclose(o1[k], o2[k])


def test_10000_random_steps():
    env = SpaceInvadersEnv()
    obs, _ = env.reset(seed=123)
    ctrl = RandomController(p=0.3)
    ctrl.reset(123)

    for i in range(10000):
        act = ctrl.act(obs)
        obs, reward, term, trunc, info = env.step(act)
        # Check obs stay in declared bounds
        assert -1.0 <= obs["player_x"][0] <= 1.0
        assert -1.0 <= obs["target_dx"][0] <= 1.0
        assert -1.0 <= obs["target_dy"][0] <= 1.0
        assert 0.0 <= obs["enemy_count"][0] <= 1.0
        assert obs["can_fire"][0] in {0.0, 1.0}
        assert -1.0 <= obs["threat_dx"][0] <= 1.0
        assert -1.0 <= obs["threat_dy"][0] <= 1.0

        if term or trunc:
            obs, _ = env.reset(seed=i)


def test_termination_paths():
    # 1. Truncation at max steps
    env = SpaceInvadersEnv(InvadersConfig(max_steps=5))
    env.reset(seed=1)
    act = np.array([0, 0, 0])
    truncated = False
    for _ in range(5):
        _, _, _, truncated, _ = env.step(act)
    assert truncated

    # 2. Player hit by enemy bullet
    env = SpaceInvadersEnv(InvadersConfig())
    env.reset(seed=1)
    # Spawn enemy bullet directly above player
    env.enemy_bullets = [[env.player_x, env.player_y - 2.0]]
    _, _, terminated, _, _ = env.step(act)
    assert terminated

    # 3. Enemies reaching player row
    env = SpaceInvadersEnv(InvadersConfig())
    env.reset(seed=1)
    env.enemies[0] = [env.player_x, env.player_y - 2.0, 1.0]
    _, _, terminated, _, _ = env.step(act)
    assert terminated


def test_m2_gate_heuristic_vs_random():
    """M2 Gate: Heuristic mean score >= 3x random over 30 seeds."""
    env = SpaceInvadersEnv()
    seeds = range(30)

    random_ctrl = RandomController(p=0.1)
    random_kills = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        random_ctrl.reset(s)
        ep_kills = 0
        done = False
        while not done:
            act = random_ctrl.act(obs)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            ep_kills = info["kills"]
        random_kills.append(ep_kills)

    heuristic_ctrl = HeuristicController()
    heuristic_kills = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        heuristic_ctrl.reset(s)
        ep_kills = 0
        done = False
        while not done:
            act = heuristic_ctrl.act(obs)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            ep_kills = info["kills"]
        heuristic_kills.append(ep_kills)

    mean_random = float(np.mean(random_kills))
    mean_heuristic = float(np.mean(heuristic_kills))

    print(f"\nM2 Gate Results: Random mean kills = {mean_random:.2f}, Heuristic mean kills = {mean_heuristic:.2f}")
    assert mean_heuristic >= 3.0 * mean_random, (
        f"M2 Gate failed! Heuristic ({mean_heuristic:.2f}) < 3x Random ({mean_random:.2f})"
    )
