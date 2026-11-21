"""Independent CPU reference: split each group and concatenate matching heads."""


def golden(torch, x, n_k):
    split, _, _, group, _ = x.shape
    kv = n_k // split
    pieces = [row.split((group - 2 * kv, kv, kv), dim=2) for row in x.unbind(0)]
    return tuple(torch.cat(parts, dim=2) for parts in zip(*pieces))


def sample(torch, torch_npu, *, s=3, b=2, nq=8, n_k=4, split=2, d=64, dtype="float16"):
    rng = torch.Generator().manual_seed(301 + s + split)
    x = torch.randn((split, s, b, (nq + 2*n_k) // split, d), generator=rng).to(getattr(torch, dtype))
    return (torch_npu.npu_format_cast(x.npu(), 2),), {"n_k": n_k}, golden(torch, x, n_k)


CASE_PARAMETERS = (
    {},
    dict(s=1, b=1, nq=1, n_k=1, split=1, d=16),
    dict(s=5, b=3, nq=12, n_k=3, split=3, dtype="bfloat16"),
    dict(s=53, b=2, nq=8, n_k=8, split=4, dtype="bfloat16"),
    # Several UB tiles per core, including the final partial tile.
    dict(s=257, b=1, nq=32, n_k=8, split=4, d=1024),
    # One Q transfer is 65536 bytes. The byte count must not truncate to uint16.
    dict(s=1, b=1, nq=32, n_k=1, split=1, d=1024, dtype="bfloat16"),
)
TOLERANCES = ((0, 0),) * 3
