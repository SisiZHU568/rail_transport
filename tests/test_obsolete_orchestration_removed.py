from pathlib import Path


def test_obsolete_projection_and_path_scheduler_are_removed() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    obsolete = (
        "src/dppo_action_space.py",
        "src/dppo_projection.py",
        "src/dppo_scenario.py",
        "src/dppo_slow_timescale_env.py",
        "src/fast_convex_scheduler.py",
        "src/fast_slot_executor.py",
    )
    assert [path for path in obsolete if (repository_root / path).exists()] == []
