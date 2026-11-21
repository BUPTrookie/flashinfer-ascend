"""Arch22 public histgram accuracy against an independent CPU reference."""

import pytest

import flashinfer
from flashinfer.operators.fused_moe.histgram import cases

pytestmark = [pytest.mark.real_compile, pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
              pytest.mark.usefixtures("aot_runtime")]


@pytest.mark.parametrize("configuration", cases.CASE_PARAMETERS)
def test_accuracy(npu_runtime, configuration):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, **configuration)
    actual = flashinfer.histgram(*args, **options)
    assert torch_npu.get_npu_format(actual) == 2
    atol, rtol = cases.TOLERANCES[0]
    torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("ids,experts,dtype,match", [
    ([], 4, "int32", "nonempty"), ([[0, 1]], 2, "int32", "one-dimensional"),
    ([0, 2], 0, "int32", "positive"), ([0, 1], 2, "int64", "int32"),
    ([-1, 0], 3, "int32", "expert IDs"), ([0, 3], 3, "int32", "expert IDs"),
])
def test_invalid_input(npu_runtime, ids, experts, dtype, match):
    torch, torch_npu = npu_runtime
    x = torch_npu.npu_format_cast(torch.tensor(ids, dtype=getattr(torch, dtype), device="npu"), 2)
    with pytest.raises(ValueError, match=match):
        flashinfer.histgram(x, experts)
