"""
gRPC client for the Edge side.
Encapsulates connection to the Cloud SpecExtendService and
serialization/deserialization of DraftRequest / VerifyResult.
"""

import grpc
import torch
from typing import Iterator, Tuple, List

from protocol import draft_pb2
from protocol import draft_pb2_grpc


class EdgeClient:
    def __init__(self, cloud_address: str = "localhost:50051"):
        self.cloud_address = cloud_address
        self.channel = None
        self.stub = None
        self._connected = False

    def connect(self):
        """Establish gRPC channel to cloud server."""
        self.channel = grpc.insecure_channel(self.cloud_address)
        self.stub = draft_pb2_grpc.SpecExtendServiceStub(self.channel)
        self._connected = True

    def close(self):
        """Close the gRPC channel."""
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.stub = None
            self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def build_draft_request(
        self,
        draft_token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        parent_last: torch.Tensor,
        current_seq_len: int,
        request_retrieval: bool = False,
        timestep: int = 0,
        prefix_token_ids: torch.Tensor = None,
    ) -> draft_pb2.DraftRequest:
        """
        Pack a draft tree into a protobuf DraftRequest.

        Args:
            draft_token_ids: [1, tree_len] or [tree_len]
            position_ids: [tree_len]
            tree_attention_mask: [tree_len, tree_len] (int8 or bool)
            parent_last: [tree_len] parent indices
            current_seq_len: current length of target KV cache
            request_retrieval: whether cloud should compute attention scores
            timestep: current generation timestep
            prefix_token_ids: [prefix_len] prompt token IDs (only on first request)
        """
        if draft_token_ids.ndim == 2:
            draft_token_ids = draft_token_ids.squeeze(0)
        if parent_last.ndim == 0:
            parent_last = parent_last.unsqueeze(0)

        # Serialize tensors to lists / bytes
        token_ids_list = draft_token_ids.cpu().tolist()
        position_ids_list = position_ids.cpu().tolist()
        parent_list = parent_last.cpu().tolist()

        # tree_attention_mask: convert to int8 numpy then bytes
        mask_np = tree_attention_mask.cpu().to(torch.int8).numpy()
        mask_bytes = mask_np.tobytes()

        req = draft_pb2.DraftRequest(
            draft_token_ids=token_ids_list,
            position_ids=position_ids_list,
            tree_attention_mask=mask_bytes,
            tree_mask_rows=tree_attention_mask.size(0),
            tree_mask_cols=tree_attention_mask.size(1),
            current_seq_len=current_seq_len,
            request_retrieval=request_retrieval,
            timestep=timestep,
            parent_last=parent_list,
        )

        if prefix_token_ids is not None:
            if prefix_token_ids.ndim == 2:
                prefix_token_ids = prefix_token_ids.squeeze(0)
            req.prefix_token_ids.extend(prefix_token_ids.cpu().tolist())

        return req

    def parse_verify_result(
        self, result: draft_pb2.VerifyResult
    ) -> Tuple[List[int], int, int, List[int], bool]:
        """
        Unpack a VerifyResult protobuf.

        Returns:
            accepted_token_ids: list of accepted token IDs
            accept_length: number of accepted tokens
            next_token: resampled next token
            retrieval_chunk_indices: selected chunk indices (empty if none)
            need_stop: whether generation should stop
        """
        accepted_token_ids = list(result.accepted_token_ids)
        accept_length = result.accept_length
        next_token = result.next_token
        retrieval_chunk_indices = list(result.retrieval_chunk_indices)
        need_stop = result.need_stop
        return (
            accepted_token_ids,
            accept_length,
            next_token,
            retrieval_chunk_indices,
            need_stop,
        )

    def speculate_stream(
        self,
        request_iterator: Iterator[draft_pb2.DraftRequest],
    ) -> Iterator[draft_pb2.VerifyResult]:
        """
        Open a bidirectional streaming RPC to the cloud.
        Yields VerifyResult messages as they arrive.
        """
        if not self._connected or self.stub is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        response_stream = self.stub.SpeculateDraft(request_iterator)
        for response in response_stream:
            yield response
