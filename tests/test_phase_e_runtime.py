from src.config import load_config
from src.phase_e_runtime import build_phase_e_runtime


def test_phase_e_runtime_binds_dimensions_and_network_snapshots() -> None:
    raw = load_config("configs/debug.yaml")

    runtime = build_phase_e_runtime(raw, total_slow_frames=32)
    network = runtime.network_for_slot(7, frame_index=2)

    assert runtime.observation_adapter.total_episode_slots == (
        32 * runtime.config.slow_frame_slots
    )
    assert (
        runtime.observation_adapter.action_spec.sha256
        == runtime.controller.action_spec.sha256
    )
    assert network.version == 8
    assert network.current_slot == 7
    assert network.serving_mec == 2
    assert network.links == runtime.network_links
    assert runtime.training_seed == int(raw["dppo"]["training"]["seed"])


def test_phase_e_runtime_rejects_nonpositive_frame_count() -> None:
    raw = load_config("configs/debug.yaml")

    try:
        build_phase_e_runtime(raw, total_slow_frames=0)
    except ValueError as error:
        assert "total_slow_frames" in str(error)
    else:
        raise AssertionError("nonpositive total_slow_frames was accepted")
