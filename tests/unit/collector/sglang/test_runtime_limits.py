from types import SimpleNamespace

import torch

from collector.sglang.runtime_limits import alloc_prefix_indices


class _NonPagedAllocator:
    def __init__(self):
        self.alloc_calls = []

    def alloc_extend(self, *_args, **_kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc(self, token_count):
        self.alloc_calls.append(token_count)
        return torch.arange(1, token_count + 1, dtype=torch.int64)


def test_alloc_prefix_indices_falls_back_for_non_paged_allocator():
    allocator = _NonPagedAllocator()
    runner = SimpleNamespace(
        device="cpu",
        server_args=SimpleNamespace(chunked_prefill_size=2),
        token_to_kv_pool_allocator=allocator,
    )

    result = alloc_prefix_indices(runner, batch_size=2, prefix_len=3)

    assert allocator.alloc_calls == [6]
    assert [indices.tolist() for indices in result] == [[1, 2, 3], [4, 5, 6]]
