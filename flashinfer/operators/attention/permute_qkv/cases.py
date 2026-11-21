"""Independent CPU reference and deterministic inputs for exact head permutation."""


def golden(torch, q, k, v, split_size):
    groups = zip(*(tensor.chunk(split_size, dim=2) for tensor in (q, k, v)))
    return torch.cat([head_slice for group in groups for head_slice in group], dim=2).movedim(2, 0).contiguous()


def sample(torch, torch_npu, *, s=3, b=2, nq=8, nkv=4, d=64, split_size=2, dtype="float16"):
    rng = torch.Generator().manual_seed(910 + s + nq + split_size)
    q, k, v = (torch.randn((s, b, heads, d), generator=rng).to(getattr(torch, dtype))
               for heads in (nq, nkv, nkv))
    tensors = tuple(torch_npu.npu_format_cast(t.npu(), 2) for t in (q, k, v))
    return tensors, {"split_size": split_size}, golden(torch, q, k, v, split_size)


CASE_PARAMETERS = (
    {},
    dict(s=1, b=1, nq=1, nkv=1, d=16, split_size=1),
    dict(s=5, b=3, nq=12, nkv=3, split_size=3, dtype="bfloat16"),
    dict(s=64, b=2, nq=8, nkv=8, d=128, split_size=4, dtype="bfloat16"),
    dict(s=65, b=2, nq=16, nkv=2, split_size=1),
    # More rows per active core than fit in a Q tile on arch22.
    dict(s=257, b=1, nq=32, nkv=8, d=512, split_size=4, dtype="bfloat16"),
    # A single 64 KiB DMA block exceeds the short (uint16) byte-length field.
    dict(s=1, b=1, nq=1, nkv=1, d=32768, split_size=1),
)
TOLERANCES = ((0, 0),)
