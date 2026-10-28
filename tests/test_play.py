"""Unit tests for ArcadeViewer and live player (headless check)."""

import numpy as np
import pytest
from flybrain.env.invaders import InvadersConfig, SpaceInvadersEnv
from flybrain.env.render import ArcadeViewer
from flybrain.brain.lif import SpikeBatch


def test_arcade_viewer_headless():
    """Verify that ArcadeViewer initializes and renders dual-pane frames headlessly."""
    viewer = ArcadeViewer(width=980, height=640, headless=True)
    env = SpaceInvadersEnv(InvadersConfig())
    obs, _ = env.reset(seed=123)

    action = np.array([1, 0, 1], dtype=np.int8)
    rates = {"move_left": 15.0, "move_right": 2.0, "fire": 10.0}

    # Simulate spike recording
    batch = SpikeBatch(
        t_ms=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        neuron_idx=np.array([5, 12, 100], dtype=np.int32),
        duration_ms=50.0,
    )
    viewer.record_spikes(batch)

    frame = viewer.render_frame(
        env=env,
        action=action,
        controller_name="brain",
        rates=rates,
        fps_actual=20.0,
        is_paused=False,
    )

    assert frame.shape == (640, 980, 3)
    assert frame.dtype == np.uint8
    # Check that game and neural areas have non-zero pixels
    assert np.any(frame[:, :480] > 0)
    assert np.any(frame[:, 480:] > 0)

    events = viewer.process_events()
    assert isinstance(events, dict)
    assert "quit" in events

    viewer.close()


def test_play_headless_integration():
    """Verify that play.py can run headlessly for 5 steps without error."""
    from flybrain.play import run_interactive
    run_interactive(
        controller_name="heuristic",
        seed=42,
        headless_check=True,
        max_check_steps=5,
    )
