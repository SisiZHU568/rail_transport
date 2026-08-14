import numpy as np

from src.config import load_config
from src.failure_process import FailureSnapshot
from src.instance_lifecycle import InstanceLifecycleManager
from src.orchestration_config import load_phase_a_config
from src.phase_d_observation import ObservationBundleVersions
from src.phase_e_main_controller import PhaseEMainController


def test_main_controller_decodes_and_atomically_commits_new_deployment() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    memory = {0: 256.0, 1: 512.0, 2: 768.0}
    lifecycle = InstanceLifecycleManager(config=config, function_memory_mb=memory)
    failure = FailureSnapshot(
        0, {0: True, 1: True, 2: True, 3: True},
        {node_id: True for node_id in config.node_resources},
        {node_id: True for node_id in config.node_resources}, 1,
    )
    controller = PhaseEMainController(config, memory)
    result = controller.decode_and_commit(
        np.zeros(controller.action_spec.action_dim),
        lifecycle_manager=lifecycle,
        failure_snapshot=failure,
        observation_versions=ObservationBundleVersions(0, 0, 1, 0, 1),
        required_replica_nodes={0: 1, 1: 1, 2: 1},
    )

    assert result.code == "OK"
    assert result.lifecycle_commit is not None and result.lifecycle_commit.accepted
    assert sum(batch.count for batch in lifecycle.snapshot().batches) >= 3
    assert result.decoder_result.used_fallback is False
