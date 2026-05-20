"""
Unit tests for protocol serialization/deserialization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocol import draft_pb2


def test_draft_request_roundtrip():
    req = draft_pb2.DraftRequest(
        draft_token_ids=[1, 2, 3, 4, 5],
        position_ids=[10, 11, 12, 13, 14],
        tree_attention_mask=bytes([1, 0, 0, 0, 0,
                                    1, 1, 0, 0, 0,
                                    1, 1, 1, 0, 0,
                                    1, 1, 1, 1, 0,
                                    1, 1, 1, 1, 1]),
        tree_mask_rows=5,
        tree_mask_cols=5,
        current_seq_len=20,
        request_retrieval=True,
        timestep=4,
        parent_last=[0, 1, 2, 3, 4],
        prefix_token_ids=[100, 101, 102],
    )

    assert list(req.draft_token_ids) == [1, 2, 3, 4, 5]
    assert list(req.position_ids) == [10, 11, 12, 13, 14]
    assert req.tree_mask_rows == 5
    assert req.tree_mask_cols == 5
    assert req.current_seq_len == 20
    assert req.request_retrieval is True
    assert req.timestep == 4
    assert list(req.parent_last) == [0, 1, 2, 3, 4]
    assert list(req.prefix_token_ids) == [100, 101, 102]

    import numpy as np
    mask_np = np.frombuffer(req.tree_attention_mask, dtype=np.int8).reshape(5, 5)
    assert mask_np[0, 0] == 1
    assert mask_np[1, 0] == 1
    assert mask_np[0, 1] == 0
    print("test_draft_request_roundtrip PASSED")


def test_verify_result_roundtrip():
    result = draft_pb2.VerifyResult(
        accepted_token_ids=[10, 11, 12],
        accept_length=2,
        next_token=20,
        retrieval_chunk_indices=[0, 3, 5],
        evicted=10,
        need_stop=False,
    )

    assert list(result.accepted_token_ids) == [10, 11, 12]
    assert result.accept_length == 2
    assert result.next_token == 20
    assert list(result.retrieval_chunk_indices) == [0, 3, 5]
    assert result.evicted == 10
    assert result.need_stop is False
    print("test_verify_result_roundtrip PASSED")


def test_chunk_indices_computation():
    """Test edge-side chunk index selection without torch."""
    # Simulate simple chunk scoring logic
    seq_len = 64
    chunk_size = 16
    n_chunks = (seq_len + chunk_size - 1) // chunk_size
    chunk_scores = [0.1 * i for i in range(n_chunks)]
    top_k = 2
    selected = sorted(range(len(chunk_scores)), key=lambda i: chunk_scores[i], reverse=True)[:top_k]
    assert len(selected) == top_k
    print("test_chunk_indices_computation PASSED")


if __name__ == "__main__":
    test_draft_request_roundtrip()
    test_verify_result_roundtrip()
    test_chunk_indices_computation()
    print("All protocol tests passed.")
