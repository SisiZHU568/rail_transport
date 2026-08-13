"""测试阶段 C 无线、有线和计算资源只读快照。"""

from dataclasses import FrozenInstanceError

import pytest

from src.fast_resource_model import (
    FastResourceConfig,
    NetworkLinkSnapshot,
    NetworkSnapshot,
    NodeFastResource,
    VNFComputeResource,
)


def build_config() -> FastResourceConfig:
    return FastResourceConfig(
        slot_seconds=1.0,
        noise_psd_watt_per_hz=1e-12,
        energy_price_per_joule=0.01,
        absolute_lex_tolerance=1e-7,
        relative_lex_tolerance=1e-7,
        residual_tolerance=1e-6,
        active_time_tolerance_seconds=1e-9,
        solver_name="CLARABEL",
        max_iterations=200,
        allow_optimal_inaccurate=False,
        nodes={0: NodeFastResource(0, 4e9, 4, 1e-27, 0.1, 0.0)},
        vnfs={(0, 0): VNFComputeResource(0, 0, 1000.0, 2e9)},
    )


def test_config_and_nested_maps_are_immutable() -> None:
    config = build_config()

    with pytest.raises(FrozenInstanceError):
        config.slot_seconds = 2.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.nodes[1] = config.nodes[0]  # type: ignore[index]


def test_network_snapshot_validates_positive_link_delay_and_capacity() -> None:
    with pytest.raises(ValueError, match="propagation"):
        NetworkLinkSnapshot(0, 0, 1, 1e6, 0.0, 1e-9)
    with pytest.raises(ValueError, match="capacity"):
        NetworkLinkSnapshot(0, 0, 1, 0.0, 0.1, 1e-9)


def test_network_snapshot_is_versioned_and_read_only() -> None:
    snapshot = NetworkSnapshot(
        version=3,
        current_slot=5,
        serving_mec=0,
        channel_gain=1e-8,
        uplink_bandwidth_hz=1e6,
        maximum_uplink_power_watt=1.0,
        links=(NetworkLinkSnapshot(0, 0, 1, 1e6, 0.01, 1e-9),),
    )

    assert snapshot.version == 3
    with pytest.raises(FrozenInstanceError):
        snapshot.channel_gain = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda: NodeFastResource(0, 1.0, 1, 0.0, 1.0, 0.0),
        lambda: NodeFastResource(0, 1.0, 1, 1.0, 0.0, 0.0),
        lambda: VNFComputeResource(0, 0, 0.0, 1.0),
    ],
)
def test_resource_parameters_reject_nonphysical_values(mutation: object) -> None:
    with pytest.raises(ValueError):
        mutation()  # type: ignore[operator]
