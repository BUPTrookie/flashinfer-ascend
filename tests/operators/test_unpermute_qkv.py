"""Arch22 public unpermute_qkv accuracy against an independent CPU reference."""

import pytest

import flashinfer
from flashinfer.operators.attention.unpermute_qkv import cases
from flashinfer.runtime import DispatchError

pytestmark = [pytest.mark.real_compile, pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
              pytest.mark.usefixtures("aot_runtime")]


@pytest.mark.parametrize("configuration", cases.CASE_PARAMETERS)
def test_accuracy(npu_runtime, configuration):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, **configuration)
    actual = flashinfer.unpermute_qkv(*args, **options)
    assert len(actual) == len(expected) == len(cases.TOLERANCES)
    for output, reference, (atol, rtol) in zip(actual, expected, cases.TOLERANCES):
        assert torch_npu.get_npu_format(output) == 2
        torch.testing.assert_close(output.cpu(), reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape,n_k,dtype,error,match", [
    ((2, 3, 8, 16), 4, "float16", ValueError, "dimensions"),
    ((2, 3, 1, 8, 16), 0, "float16", ValueError, "n_k"),
    ((2, 3, 1, 8, 16), 3, "float16", ValueError, "divisible"),
    ((2, 3, 1, 7, 16), 4, "float16", ValueError, "query heads"),
    ((2, 3, 1, 8, 16), 4, "float32", ValueError, "dtype"),
    ((2, 3, 1, 8, 17), 4, "float16", DispatchError, "D"),
    ((2, 1, 1, 8, 6144), 4, "float16", DispatchError, "UB"),
])
def test_invalid_input(npu_runtime, shape, n_k, dtype, error, match):
    torch, torch_npu = npu_runtime
    x = torch_npu.npu_format_cast(torch.empty(shape, dtype=getattr(torch, dtype), device="npu"), 2)
    with pytest.raises(error, match=match):
        flashinfer.unpermute_qkv(x, n_k)
