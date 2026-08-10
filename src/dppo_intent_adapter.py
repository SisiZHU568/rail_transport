"""把成功的 DPPO 投影结果转换为带有效期的通用执行意图。"""

from src.dppo_projection import ProjectionResult
from src.sfc_deployment_intent import SFCDeploymentIntent


class DPPOIntentAdapter:
    """集中添加 DPPO 来源和慢尺度窗口元数据。"""

    def to_intent(
        self,
        projection: ProjectionResult,
        *,
        decision_slot: int,
        valid_until_slot: int,
    ) -> SFCDeploymentIntent:
        """成功投影才能进入快层；时隙合法性由通用意图继续校验。"""

        if not isinstance(projection, ProjectionResult):
            raise TypeError("projection must be a ProjectionResult.")
        if not projection.success or projection.function_intents is None:
            raise ValueError("A failed projection cannot become an execution intent.")
        return SFCDeploymentIntent(
            decision_slot=decision_slot,
            valid_until_slot=valid_until_slot,
            function_intents=projection.function_intents,
            source_algorithm="dppo",
        )
