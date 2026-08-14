"""测试在线环境快照按配置生成固定维度 DPPO 观察。"""

from src.config import load_config
from src.failure_process import FailureSnapshot
from src.fast_resource_model import NetworkSnapshot
from src.instance_lifecycle import InstanceBatch, LifecycleSnapshot, LifecycleStatus
from src.orchestration_config import load_phase_a_config
from src.phase_e_observation_adapter import PhaseEObservationAdapter
from src.phase_e_environment import SlowDecisionContext
from src.phase_d_observation import ObservationBundleVersions
from src.queue_state import BatchRecord, QueueFragment, QueueSnapshot
from src.safe_deployment_decoder import ActionSpec


def test_online_observation_uses_configured_entities_and_runtime_snapshots() -> None:
    config = load_phase_a_config(load_config("configs/debug.yaml"))
    action_spec = ActionSpec.from_phase_a_config(config)
    lifecycle = LifecycleSnapshot(
        2,
        5,
        (InstanceBatch("warm", 0, 0, LifecycleStatus.WARM, 2, 0, 10),),
        {node_id: (512.0 if node_id == 0 else 0.0) for node_id in config.node_resources},
    )
    batch = BatchRecord("b", 0, 5.0, 12.0, total_input_equivalent_bits=1e6)
    queue = QueueSnapshot(
        3,
        5,
        (batch,),
        (QueueFragment("u", "b", 0, -1, None, None, 1e6, 5),),
        (), (), (),
    )
    failure = FailureSnapshot(
        5, {0: True},
        {node_id: True for node_id in config.node_resources},
        {node_id: True for node_id in config.node_resources}, 4,
    )
    network = NetworkSnapshot(5, 5, 0, 1e-6, 1e6, 1.0, ())
    context = SlowDecisionContext(
        queue, lifecycle, failure, network,
        ObservationBundleVersions(3, 2, 4, 5, 1),
    )
    adapter = PhaseEObservationAdapter(
        config,
        action_spec,
        total_episode_slots=100,
        maximum_drain_slots=20,
    )

    snapshot = adapter.encode(context, next_mec=1)

    assert len(snapshot.values) == snapshot.spec.dimension
    assert snapshot.spec.ordered_node_ids == tuple(sorted(config.node_resources))
    assert snapshot.spec.ordered_function_ids == (0, 1, 2)
    assert snapshot.versions == context.versions
    assert all(value == value for value in snapshot.values)
