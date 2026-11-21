"""Shape-only tests for arch22 dispatch, admission and the QKV permutation."""

from dataclasses import replace

import pytest
import torch

import flashinfer
from flashinfer.operators.attention.permute_qkv.operator import OPERATOR
from flashinfer.operators.runtime import signature
from flashinfer.runtime import DispatchError, HardwareSignature, dispatch


def request(s=3, b=2, nq=8, nkv=4, d=64, split=2):
    q, k, v = (torch.empty(s, b, n, d, device="meta", dtype=torch.float16) for n in (nq, nkv, nkv))
    args = OPERATOR.bind(q, k, v, split)
    call, _, _ = signature(OPERATOR.op, args)
    return replace(call, physical_layouts=("ND",) * 4)


@pytest.mark.parametrize("soc", ["Ascend910B", "Ascend910C"])
def test_arch22_routing(soc):
    prefix = "ascendc/attention/permute_qkv/native/"
    hardware = HardwareSignature(soc, 24)
    assert dispatch(request(), hardware, backend="ascendc").kernel_id == prefix + soc
    other = "Ascend910B" if soc == "Ascend910C" else "Ascend910C"
    with pytest.raises(DispatchError):
        dispatch(request(), hardware, kernel_id=prefix + other)


@pytest.mark.parametrize("shape,diagnostic", [
    ({"d": 17}, "D"),
    ({"s": 2, "b": 1, "nq": 64, "d": 1024}, "UB"),
    ({"s": 2**24, "b": 1, "nq": 4, "d": 16}, "32-bit"),
])
def test_rejects_unrepresentable_transfers(shape, diagnostic):
    with pytest.raises(DispatchError, match=diagnostic):
        dispatch(request(**shape), HardwareSignature("Ascend910B", 24), backend="ascendc")


def test_public_fake_shape_and_dtype():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        q, k, v = (torch.empty(3, 2, n, 16, dtype=torch.bfloat16) for n in (4, 2, 2))
        out = flashinfer.permute_qkv(q, k, v, 2)
        assert out.shape == (8, 3, 2, 16) and out.dtype == q.dtype
