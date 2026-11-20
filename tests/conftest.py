import pytest

from tracesim.device import DeviceDescriptor, OpClassModel
from tracesim.trace import ELEMENT_CLASSES, FLOP_CLASSES


@pytest.fixture
def device() -> DeviceDescriptor:
    """A made-up device with round numbers, so expected costs can be worked out by hand."""
    classes = {}
    for c in FLOP_CLASSES:
        classes[c] = OpClassModel(
            unit="flops", rate={"float32": 1e10, "float64": 5e9}, call_overhead_s=1e-5,
            serial_fraction=0.1, fork_overhead_s=1e-5, backward_overhead_ratio=1.0,
            backward_ratio=2.0, backward_ratio_no_input_grad=1.0,
        )
    for c in ELEMENT_CLASSES:
        classes[c] = OpClassModel(
            unit="elements", rate={"float32": 1e9, "float64": 5e8}, call_overhead_s=1e-5,
            serial_fraction=0.5, fork_overhead_s=1e-5, backward_overhead_ratio=1.0,
            backward_ratio=1.0,
        )
    classes["free"] = OpClassModel(unit="none", call_overhead_s=1e-6,
                                   backward_overhead_ratio=1.0, backward_ratio=0.0)
    return DeviceDescriptor(
        name="test", cores=4, cache_bytes=8 << 20, mem_bandwidth_bytes_per_s=1e10,
        mem_serial_fraction=0.5, op_classes=classes,
    )
