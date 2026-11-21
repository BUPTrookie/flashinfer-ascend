"""Public permute_qkv accuracy and contract tests on arch22."""

import pytest

import flashinfer
from flashinfer.operators.attention.permute_qkv import cases

pytestmark = [pytest.mark.real_compile, pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
              pytest.mark.usefixtures("aot_runtime")]


@pytest.mark.parametrize("configuration", cases.CASE_PARAMETERS)
def test_accuracy(npu_runtime, configuration):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, **configuration)
    actual = flashinfer.permute_qkv(*args, **options)
    assert torch_npu.get_npu_format(actual) == 2
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("problem,match", [
    ("rank", "shape"), ("empty", "positive"), ("kv_shape", "shapes"), ("q_heads", "divisible"),
    ("zero_split", "split_size"), ("uneven_split", "split_size"), ("mixed_dtype", "dtype"),
])
def test_invalid_input(npu_runtime, problem, match):
    torch, torch_npu = npu_runtime
    def tensor(heads, *, rows=3, dtype=torch.float16):
        return torch_npu.npu_format_cast(torch.empty((rows, 2, heads, 16), dtype=dtype, device="npu"), 2)
    q = tensor(7 if problem == "q_heads" else 8, rows=0 if problem == "empty" else 3)
    k = tensor(4)
    v = tensor(2 if problem == "kv_shape" else 4, dtype=torch.bfloat16 if problem == "mixed_dtype" else q.dtype)
    if problem == "rank":
        q = q[0]
    split = {"zero_split": 0, "uneven_split": 3}.get(problem, 2)
    with pytest.raises(ValueError, match=match):
        flashinfer.permute_qkv(q, k, v, split)
