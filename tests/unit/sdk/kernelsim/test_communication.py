import pytest

from aiconfigurator_core.sdk.kernelsim.communication import (
    COMMUNICATION_MODEL_VERSION,
    CommunicationParameters,
    estimate_communication,
)


def test_hopper_calibrated_communication_parameters():
    estimate = estimate_communication(0.300)

    assert COMMUNICATION_MODEL_VERSION
    assert estimate.startup_ms == pytest.approx(0.007)
    assert estimate.payload_ms == pytest.approx(0.300 / 0.75)
    assert estimate.latency_ms == pytest.approx(0.007 + 0.300 / 0.75)
    assert estimate.sol_latency_ms == pytest.approx(0.300)
    assert estimate.eta == pytest.approx(0.75)


def test_noop_communication_does_not_pay_startup():
    estimate = estimate_communication(0.0)

    assert estimate.latency_ms == 0.0
    assert estimate.startup_ms == 0.0
    assert estimate.payload_ms == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"startup_us": -1}, "startup"),
        ({"eta": 0}, "eta"),
        ({"eta": 1.1}, "eta"),
    ],
)
def test_communication_parameter_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CommunicationParameters(**kwargs)


def test_negative_sol_latency_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        estimate_communication(-0.1)
