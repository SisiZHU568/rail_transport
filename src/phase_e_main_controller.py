"""新 A→E 主链的慢层入口：策略评分、安全解码、生命周期原子提交。"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.failure_process import FailureSnapshot
from src.instance_lifecycle import (
    DeploymentTarget,
    InstanceLifecycleManager,
    LifecycleCommitResult,
    LifecycleDeploymentPlan,
)
from src.orchestration_config import PhaseAConfig
from src.phase_d_observation import ObservationBundleVersions
from src.safe_deployment_decoder import (
    ActionSpec,
    DecoderInput,
    DecoderResult,
    SafeDeploymentDecoder,
)


@dataclass(frozen=True)
class SlowDecisionResult:
    code: str
    decoder_result: DecoderResult
    lifecycle_commit: LifecycleCommitResult | None


class PhaseEMainController:
    """不做投影、不做启发式回退，提交失败时保持原状态。"""

    def __init__(self, config: PhaseAConfig, function_memory_mb: Mapping[int, float]) -> None:
        self.config = config
        self.action_spec = ActionSpec.from_phase_a_config(config)
        self.decoder = SafeDeploymentDecoder(
            config, self.action_spec, function_memory_mb=function_memory_mb
        )

    def decode_and_commit(
        self,
        policy_scores: np.ndarray,
        *,
        lifecycle_manager: InstanceLifecycleManager,
        failure_snapshot: FailureSnapshot,
        observation_versions: ObservationBundleVersions,
        required_replica_nodes: Mapping[int, int],
        fault_domain_by_node: Mapping[int, int] | None = None,
        minimum_fault_domains: Mapping[int, int] | None = None,
        domain_availability: Mapping[int, float] | None = None,
        node_conditional_availability: Mapping[int, float] | None = None,
        maximum_vnf_unavailability: Mapping[int, float] | None = None,
    ) -> SlowDecisionResult:
        lifecycle = lifecycle_manager.snapshot()
        expected = (
            observation_versions.lifecycle_version,
            observation_versions.failure_version,
        )
        if expected != (lifecycle.version, failure_snapshot.version):
            empty = self.decoder.decode(
                np.full(self.action_spec.action_dim, np.nan),
                DecoderInput(failure_snapshot.effective_node_up, {}, required_replica_nodes),
            )
            return SlowDecisionResult("STALE_SNAPSHOT", empty, None)
        locked = {
            pair: lifecycle.locked_count(*pair, current_slot=lifecycle.current_slot)
            for pair in self.action_spec.pairs
        }
        decoded = self.decoder.decode(
            policy_scores,
            DecoderInput(
                failure_snapshot.effective_node_up,
                locked,
                required_replica_nodes,
                fault_domain_by_node or {},
                minimum_fault_domains or {},
                domain_availability or {},
                node_conditional_availability or {},
                maximum_vnf_unavailability or {},
            ),
        )
        if decoded.code != "OK" or decoded.plan is None:
            return SlowDecisionResult(decoded.code, decoded, None)
        plan = LifecycleDeploymentPlan(
            lifecycle.version,
            failure_snapshot.version,
            lifecycle.current_slot,
            tuple(
                DeploymentTarget(
                    function_id, node_id,
                    decoded.plan.instance_counts[(function_id, node_id)],
                    decoded.plan.retention_slots[(function_id, node_id)],
                )
                for function_id, node_id in self.action_spec.pairs
            ),
        )
        commit = lifecycle_manager.commit_deployment(plan, failure_snapshot)
        return SlowDecisionResult("OK" if commit.accepted else commit.code, decoded, commit)
