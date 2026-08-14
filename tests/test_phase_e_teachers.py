import numpy as np

from src.config import load_config
from src.fast_resource_model import NetworkLinkSnapshot, NetworkSnapshot, load_fast_resource_config
from src.fast_resource_optimizer import FastResourceOptimizer
from src.failure_process import ScriptedFailureProcess
from src.instance_lifecycle import (
    DeploymentTarget,
    InstanceLifecycleManager,
    LifecycleDeploymentPlan,
)
from src.orchestration_config import load_phase_a_config
from src.phase_e_teacher_generation import (
    CausalTeacherProxy,
    TeacherMetrics,
    build_teacher_candidate,
    generate_exact_teacher_labels,
)
from src.phase_e_environment import ArrivalBatch
from src.phase_e_main_controller import PhaseEMainController
from src.queue_manager import QueueStateManager
from src.queue_state import StageFlowConfig
from src.phase_e_teachers import TeacherCandidate, select_teacher_labels
from src.safe_deployment_decoder import ActionSpec, DecoderInput, SafeDeploymentDecoder
from src.topology import build_linear_topology


def test_three_service_first_teachers_deduplicate_identical_plans() -> None:
    candidates = (
        TeacherCandidate("a", np.array([0.1, 0.5]), 1.0, 0.04, 0.5, 0.3, "a"),
        TeacherCandidate("b", np.array([0.8, 0.5]), 0.5, 0.01, 0.4, 0.1, "b"),
        TeacherCandidate("c", np.array([0.9, 0.5]), 0.5, 0.02, 0.8, 0.2, "c"),
    )

    labels = select_teacher_labels(candidates)

    assert {label.plan_hash for label in labels} == {"b", "c"}
    assert next(label for label in labels if label.plan_hash == "b").teacher_types == (
        "COST", "BALANCE"
    )
    assert next(label for label in labels if label.plan_hash == "c").teacher_types == (
        "RELIABILITY",
    )
    assert all(np.array_equal(label.unbounded_action, np.log(label.scores / (1-label.scores))) for label in labels)


def _teacher_context():
    raw = load_config("configs/debug.yaml")
    config = load_phase_a_config(raw)
    memory = {
        item["function_id"]: float(item["memory_mb"])
        for item in raw["rl_scenario"]["functions"]
    }
    manager = InstanceLifecycleManager(config=config, function_memory_mb=memory)
    failure = ScriptedFailureProcess(build_linear_topology(raw)).state_for_slot(0)
    spec = ActionSpec.from_phase_a_config(config)
    decoder = SafeDeploymentDecoder(config, spec, function_memory_mb=memory)
    decoder_input = DecoderInput(
        failure.effective_node_up,
        {},
        {function_id: 1 for function_id, _ in spec.pairs},
    )
    return config, manager, failure, spec, decoder, decoder_input


def test_teacher_candidate_uses_read_only_lifecycle_preview_and_roundtrip() -> None:
    config, manager, failure, spec, decoder, decoder_input = _teacher_context()
    plan = decoder.decode(np.full(spec.action_dim, 0.35), decoder_input).plan
    before = manager.snapshot()

    candidate = build_teacher_candidate(
        plan=plan,
        eligible_teacher_types=("COST",),
        decoder=decoder,
        decoder_input=decoder_input,
        lifecycle_manager=manager,
        failure_snapshot=failure,
        metrics=TeacherMetrics(0.2, 1.5, 0.03, 0.4, 0.02),
    )

    assert candidate.code == "OK"
    assert candidate.candidate is not None
    assert candidate.candidate.plan == plan
    assert decoder.decode(candidate.candidate.scores, decoder_input).plan == plan
    assert manager.snapshot() == before


def test_exact_teacher_search_limit_never_returns_partial_labels() -> None:
    config, manager, failure, spec, decoder, decoder_input = _teacher_context()

    result = generate_exact_teacher_labels(
        decoder=decoder,
        decoder_input=decoder_input,
        lifecycle_manager=manager,
        failure_snapshot=failure,
        retention_profiles={
            "COST": {pair: spec.retention_slot_options[0] for pair in spec.pairs},
            "RELIABILITY": {pair: spec.retention_slot_options[-1] for pair in spec.pairs},
            "BALANCE": {pair: spec.retention_slot_options[0] for pair in spec.pairs},
        },
        evaluator=lambda plan, preview: TeacherMetrics(0.0, 0.0, 0.1, 0.0, 0.0),
        max_generated_patterns=1,
    )

    assert result.code == "TEACHER_SEARCH_LIMIT"
    assert result.labels == ()
    assert result.candidate_count == 0


def test_causal_teacher_proxy_runs_real_fast_path_without_mutating_state() -> None:
    raw = load_config("configs/debug.yaml")
    config, manager, failure, spec, decoder, decoder_input = _teacher_context()
    flow = StageFlowConfig(tuple(
        float(item["output_ratio"]) for item in raw["rl_scenario"]["functions"]
    ))
    queues = QueueStateManager(flow, slot_seconds=config.fast_slot_seconds)
    queues.admit_batch("teacher-batch", 0, 0.0, 10.0, 2e5)
    plan = decoder.decode(np.full(spec.action_dim, 0.35), decoder_input).plan
    lifecycle_plan = LifecycleDeploymentPlan(
        manager.snapshot().version,
        failure.version,
        manager.snapshot().current_slot,
        tuple(
            DeploymentTarget(function_id, node_id, plan.instance_counts[pair], plan.retention_slots[pair])
            for pair in spec.pairs
            for function_id, node_id in (pair,)
        ),
    )
    preview = manager.preview_deployment(lifecycle_plan, failure)
    links = tuple(
        NetworkLinkSnapshot(
            2 * node_id + direction,
            node_id if direction == 0 else node_id + 1,
            node_id + 1 if direction == 0 else node_id,
            10e6,
            0.1,
            1e-9,
        )
        for node_id in range(len(config.node_resources) - 1)
        for direction in (0, 1)
    )
    proxy = CausalTeacherProxy(
        controller=PhaseEMainController(config, manager.function_memory_mb),
        optimizer=FastResourceOptimizer(load_fast_resource_config(raw), flow),
        lifecycle_manager=manager,
        queue_manager=queues,
        failure_snapshot=failure,
        structural_reliability_margin=lambda _: 0.02,
        function_cycles_per_input_bit={
            item["function_id"]: float(item["cpu_cycles_per_input_bit"])
            for item in raw["rl_scenario"]["functions"]
        },
        network_for_slot=lambda slot: NetworkSnapshot(
            slot + 1, slot, 0, 1e-6, 1e6, 1.0, links
        ),
        predicted_arrivals_by_slot={},
    )
    lifecycle_before = manager.snapshot()
    queue_before = queues.snapshot()

    metrics = proxy.evaluate(plan, preview)

    assert metrics.predicted_deficit >= 0.0
    assert metrics.predicted_cost >= 0.0
    assert metrics.structural_reliability_margin == 0.02
    assert manager.snapshot() == lifecycle_before
    assert queues.snapshot() == queue_before
