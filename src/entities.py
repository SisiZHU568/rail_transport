"""
entities.py

本文件定义轨道边缘 Serverless SFC 系统中的基础数据对象。

当前这些类只负责保存数据和检查参数，
暂时不执行列车移动、冷启动或优化算法。
"""

from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    """
    计算节点类型。
    """

    # 沿铁路部署的轨旁 MEC
    TRACKSIDE = "trackside"

    # 远端中心云
    CLOUD = "cloud"


class ServicePriority(str, Enum):
    """
    铁路业务优先级。
    """

    # 安全关键业务，例如列车控制和故障告警
    CRITICAL = "critical"

    # 重要业务，例如设备状态监测
    IMPORTANT = "important"

    # 普通业务，例如乘客信息服务
    NORMAL = "normal"

class InstanceStatus(str, Enum):
    """
    Serverless 函数实例的生命周期状态。

    ABSENT:
        当前节点上不存在这个函数容器。

    COLD_STARTING:
        容器正在创建，运行环境正在初始化。

    WARM:
        容器已经准备完成，可以立即处理请求。

    BUSY:
        容器正在处理请求。

    FAILED:
        容器或所在节点发生故障，无法处理请求。
    """

    ABSENT = "absent"
    COLD_STARTING = "cold_starting"
    WARM = "warm"
    BUSY = "busy"
    FAILED = "failed"

@dataclass
class EdgeNode:
    """
    边缘计算节点。

    参数说明
    ----------
    node_id:
        节点编号。

    name:
        节点名称，例如 MEC-1。

    node_type:
        节点类型，例如轨旁 MEC、车载节点或中心云。

    cpu_capacity:
        节点的计算资源容量。
        当前使用抽象计算单位。

    memory_capacity_mb:
        节点内存容量，单位为 MB。

    reliability:
        节点局部可用率，取值范围为 (0, 1]。

        在可靠性模块中，它表示：

            当节点所属故障域正常时，
            该节点自身保持可用的条件概率。

        节点的最终可用率还需要同时考虑
        所属故障域是否正常。

    fault_domain:
        节点所属故障域。
        同一故障域中的节点可能一起发生故障。
    """

    node_id: int
    name: str
    node_type: NodeType
    cpu_capacity: float
    memory_capacity_mb: float
    reliability: float
    fault_domain: int

    def __post_init__(self) -> None:
        """
        对象创建完成后自动检查参数。
        """

        if self.node_id < 0:
            raise ValueError("node_id 不能小于 0。")

        if not self.name:
            raise ValueError("节点名称不能为空。")

        if self.cpu_capacity <= 0:
            raise ValueError("CPU 容量必须大于 0。")

        if self.memory_capacity_mb <= 0:
            raise ValueError("内存容量必须大于 0。")

        if not 0 < self.reliability <= 1:
            raise ValueError("节点可靠性必须位于 (0, 1]。")

        if self.fault_domain < 0:
            raise ValueError("故障域编号不能小于 0。")

    def has_sufficient_capacity(
        self,
        cpu_demand: float,
        memory_demand_mb: float,
    ) -> bool:
        """
        判断给定负载能否同时满足节点的 CPU 和内存容量。

        容量与需求恰好相等时仍然可行；只有需求严格大于容量时，
        才属于资源超限。该方法只做判断，不会修改节点状态。

        Parameters
        ----------
        cpu_demand:
            当前时隙需要的抽象 CPU 计算量。

        memory_demand_mb:
            当前活动函数实例需要的内存，单位为 MB。
        """

        if cpu_demand < 0:
            raise ValueError("CPU 需求不能小于 0。")

        if memory_demand_mb < 0:
            raise ValueError("内存需求不能小于 0。")

        return (
            cpu_demand <= self.cpu_capacity
            and memory_demand_mb <= self.memory_capacity_mb
        )


@dataclass
class ServerlessFunction:
    """
    一个 Serverless 函数。

    例如：
    数据清洗、特征提取、异常检测。
    """

    # 函数编号
    function_id: int

    # 函数名称
    name: str

    # 一个函数实例需要的内存，单位为 MB
    memory_mb: float

    # 每个请求需要的计算量，当前使用抽象单位
    cpu_cycles_per_request: float

    # 容器镜像大小，单位为 MB
    image_size_mb: float

    # 容器已处于 warm 状态时的执行时间
    warm_exec_time_ms: float

    # 创建新容器时的冷启动时间
    cold_start_time_ms: float

    # 函数输出数据量与输入数据量的比例
    output_ratio: float

    def __post_init__(self) -> None:
        """
        检查函数参数是否合法。
        """

        if self.function_id < 0:
            raise ValueError("function_id 不能小于 0。")

        if not self.name:
            raise ValueError("函数名称不能为空。")

        if self.memory_mb <= 0:
            raise ValueError("函数内存必须大于 0。")

        if self.cpu_cycles_per_request <= 0:
            raise ValueError("函数计算量必须大于 0。")

        if self.image_size_mb < 0:
            raise ValueError("镜像大小不能小于 0。")

        if self.warm_exec_time_ms < 0:
            raise ValueError("温执行时间不能小于 0。")

        if self.cold_start_time_ms < 0:
            raise ValueError("冷启动时间不能小于 0。")

        if self.output_ratio < 0:
            raise ValueError("输出比例不能小于 0。")

    def cpu_demand(self, request_count: int) -> float:
        """
        计算一批请求在当前时隙产生的抽象 CPU 需求。

        `cpu_cycles_per_request` 在当前项目中表示每个请求需要的
        抽象计算单位，尚未换算成真实处理器周期或 CPU 利用率。

        Parameters
        ----------
        request_count:
            当前快时隙内需要处理的请求数量。
        """

        if request_count < 0:
            raise ValueError("请求数量不能小于 0。")

        return (
            self.cpu_cycles_per_request
            * request_count
        )


@dataclass
class SFCType:
    """
    一类服务功能链。

    function_ids 中函数编号的排列顺序，
    就是这些函数的实际执行顺序。
    """

    # SFC 编号
    sfc_id: int

    # SFC 名称
    name: str

    # 按执行顺序排列的函数编号
    function_ids: list[int]

    # 最大端到端时延，单位为毫秒
    deadline_ms: float

    # 最低可靠性要求
    reliability_target: float

    # 业务优先级
    priority: ServicePriority

    def __post_init__(self) -> None:
        """
        检查 SFC 参数是否合法。
        """

        if self.sfc_id < 0:
            raise ValueError("sfc_id 不能小于 0。")

        if not self.name:
            raise ValueError("SFC 名称不能为空。")

        if len(self.function_ids) == 0:
            raise ValueError("一条 SFC 至少包含一个函数。")

        if any(function_id < 0 for function_id in self.function_ids):
            raise ValueError("函数编号不能小于 0。")

        if self.deadline_ms <= 0:
            raise ValueError("时延限制必须大于 0。")

        if not 0 < self.reliability_target <= 1:
            raise ValueError("可靠性阈值必须位于 (0, 1]。")


@dataclass(frozen=True)
class SlotConstraintAudit:
    """
    保存一个快时隙的资源、部署计划和可靠性审计结果。

    该对象只记录检查结果，不会主动拒绝请求或修改部署方案。
    后续快层优化器可以读取这些字段，定位并修复不可行方案。
    """

    # 实际发生资源需求的节点及其抽象 CPU 需求。
    node_cpu_demand: dict[int, float]

    # 实际存在活动实例的节点及其内存需求，单位为 MB。
    node_memory_demand_mb: dict[int, float]

    # 发生资源超限的节点编号，使用有序元组保证结果可复现。
    cpu_violation_node_ids: tuple[int, ...]
    memory_violation_node_ids: tuple[int, ...]

    # 副本计划中引用的未知节点，以及缺少部署的函数。
    invalid_replica_node_ids: tuple[int, ...]
    missing_function_ids: tuple[int, ...]

    # 计划无效、无法计算可靠性时，该值为 None。
    exact_sfc_reliability: float | None
    reliability_target: float
    reliability_target_met: bool

    resource_constraints_met: bool
    replica_plan_valid: bool
    all_constraints_met: bool

    # 使用中文保存详细原因，便于直接查看日志和实验结果。
    violation_reasons: tuple[str, ...]

    # 实际副本数与慢层许可数量不一致的函数编号。
    # 默认空元组用于兼容尚未迁移到新审计器的旧调用位置。
    replica_count_violation_function_ids: tuple[int, ...] = ()


@dataclass
class RequestBatch:
    """
    一个快时隙内到达的一批同类请求。

    第一阶段使用批量请求，
    而不是给每个请求分别创建对象。
    """

    # 请求到达的时隙
    time_slot: int

    # 请求对应的 SFC 编号
    sfc_id: int

    # 当前时隙到达的请求数量
    request_count: int

    # 单个请求输入数据量，单位为 MB
    input_size_mb: float

    def __post_init__(self) -> None:
        """
        检查请求参数是否合法。
        """

        if self.time_slot < 0:
            raise ValueError("time_slot 不能小于 0。")

        if self.sfc_id < 0:
            raise ValueError("sfc_id 不能小于 0。")

        if self.request_count < 0:
            raise ValueError("请求数量不能小于 0。")

        if self.input_size_mb < 0:
            raise ValueError("请求输入数据量不能小于 0。")

@dataclass
class FunctionInstance:
    """
    部署在某个计算节点上的 Serverless 函数实例。

    ServerlessFunction 描述“函数本身”；
    FunctionInstance 描述“函数当前在某个节点上的运行状态”。

    例如：

        ServerlessFunction:
            异常检测函数需要 768 MB 内存，
            冷启动时间为 800 ms。

        FunctionInstance:
            异常检测函数目前在 MEC-2 上，
            状态为 WARM，已经空闲了 2 个时隙。
    """

    # 函数实例所在的计算节点编号
    node_id: int

    # 对应的 Serverless 函数编号
    function_id: int

    # 当前实例状态，默认不存在
    status: InstanceStatus = InstanceStatus.ABSENT

    # 容器已经连续空闲了多少个时隙
    idle_slots: int = 0

    # 剩余冷启动时间，单位为毫秒
    remaining_startup_time_ms: float = 0.0

    def __post_init__(self) -> None:
        """
        创建函数实例后自动检查参数。
        """

        if self.node_id < 0:
            raise ValueError("函数实例所在节点编号不能小于 0。")

        if self.function_id < 0:
            raise ValueError("函数编号不能小于 0。")

        if self.idle_slots < 0:
            raise ValueError("空闲时隙数量不能小于 0。")

        if self.remaining_startup_time_ms < 0:
            raise ValueError("剩余启动时间不能小于 0。")

@dataclass
class TrainState:
    """
    某个快时隙中的列车状态。
    """

    # 当前时隙编号
    time_slot: int

    # 列车在线路上的位置，单位为米
    position_m: float

    # 列车速度，单位为米/秒
    speed_mps: float

    # 当前为列车提供接入服务的 MEC 编号
    serving_mec: int

    # 列车接下来将进入的 MEC 编号
    next_mec: int

    # 当前覆盖区域内的预计剩余驻留时间
    remaining_dwell_time_s: float

    def __post_init__(self) -> None:
        """
        检查列车状态是否合法。
        """

        if self.time_slot < 0:
            raise ValueError("time_slot 不能小于 0。")

        if self.position_m < 0:
            raise ValueError("列车位置不能小于 0。")

        if self.speed_mps < 0:
            raise ValueError("列车速度不能小于 0。")

        if self.serving_mec < 0:
            raise ValueError("当前 MEC 编号不能小于 0。")

        if self.next_mec < 0:
            raise ValueError("下一 MEC 编号不能小于 0。")

        if self.remaining_dwell_time_s < 0:
            raise ValueError("剩余驻留时间不能小于 0。")
