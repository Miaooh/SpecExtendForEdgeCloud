"""
gRPC Service implementation for the Cloud side.
Receives DraftRequest stream from Edge, runs verification via CloudVerifyEngine,
and returns VerifyResult stream.
"""

import grpc
import numpy as np
import torch
from typing import Iterator

from protocol import draft_pb2
from protocol import draft_pb2_grpc
from cloud_server.engine import CloudVerifyEngine


class SpecExtendServicer(draft_pb2_grpc.SpecExtendServiceServicer):
    def __init__(self, engine: CloudVerifyEngine):
        self.engine = engine

    def SpeculateDraft(
        self, request_iterator: Iterator[draft_pb2.DraftRequest], context
    ) -> Iterator[draft_pb2.VerifyResult]:
        """
        Bidirectional streaming RPC.
        For each DraftRequest from edge:
          1. Deserialize tensors
          2. Run tree_decoding + verify
          3. Compute retrieval chunk indices if requested
          4. Yield VerifyResult

        NOTE: Per-stream local state is maintained inside this generator
        so concurrent connections do not interfere.
        """
        # Per-stream state
        cloud_input_ids = None
        input_len = 0
        first_request = True

        for req in request_iterator:
            # Deserialize draft tree tensors
            draft_token_ids = torch.tensor(
                list(req.draft_token_ids), dtype=torch.long, device=self.engine.device
            ).unsqueeze(0)
            position_ids = torch.tensor(
                list(req.position_ids), dtype=torch.long, device=self.engine.device
            )
            parent_last = torch.tensor(
                list(req.parent_last), dtype=torch.long, device=self.engine.device
            )
            tree_mask_rows = req.tree_mask_rows
            tree_mask_cols = req.tree_mask_cols

            # Deserialize attention mask from bytes
            mask_np = np.frombuffer(req.tree_attention_mask, dtype=np.int8)
            if mask_np.size != tree_mask_rows * tree_mask_cols:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    f"Mask size mismatch: {mask_np.size} vs {tree_mask_rows * tree_mask_cols}"
                )
                return

            tree_attention_mask = torch.from_numpy(
                mask_np.reshape(tree_mask_rows, tree_mask_cols)
            ).to(self.engine.device)

            request_retrieval = req.request_retrieval
            current_seq_len = req.current_seq_len

            # Handle prefix / prefill on first request
            if first_request and len(req.prefix_token_ids) > 0:
                prefix_ids = torch.tensor(
                    list(req.prefix_token_ids), dtype=torch.long, device=self.engine.device
                ).unsqueeze(0)
                input_len = prefix_ids.shape[1]

                # Initialize target KV cache
                max_new_tokens_guess = current_seq_len - input_len + 256
                self.engine.init_generation(prefix_ids, max_new_tokens=max_new_tokens_guess)

                # Prefill target model
                first_token, _ = self.engine.prefill(prefix_ids)

                # cloud_input_ids starts with prefix + first_token
                cloud_input_ids = torch.cat([prefix_ids, first_token.to(self.engine.device)], dim=-1)

                # The draft tree from edge should be verified starting from first_token as root.
                # Prepend first_token to draft tree for verification.
                draft_token_ids = torch.cat([first_token.to(draft_token_ids.device), draft_token_ids], dim=-1)
                position_ids = torch.cat([
                    torch.tensor([position_ids[0].item() - 1], dtype=torch.long, device=position_ids.device),
                    position_ids
                ], dim=-1)
                # Expand tree mask for root node
                tree_attention_mask = torch.cat([
                    torch.zeros(1, tree_attention_mask.size(1), dtype=tree_attention_mask.dtype, device=tree_attention_mask.device),
                    tree_attention_mask
                ], dim=0)
                tree_attention_mask = torch.cat([
                    torch.ones(tree_attention_mask.size(0), 1, dtype=tree_attention_mask.dtype, device=tree_attention_mask.device),
                    tree_attention_mask
                ], dim=-1)
                # Parent indices: root has no parent (-1 shifted to 0), shift existing parents by 1
                parent_last = torch.cat([
                    torch.tensor([0], dtype=torch.long, device=parent_last.device),
                    parent_last + 1
                ], dim=-1)

                first_request = False
            elif first_request:
                # No prefix provided; cannot proceed
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("First request must include prefix_token_ids for cloud prefill.")
                return
            else:
                # Subsequent request: edge already prepended root token.
                # Sync length if needed.
                if cloud_input_ids is None:
                    cloud_input_ids = torch.zeros(
                        (1, current_seq_len), dtype=torch.long, device=self.engine.device
                    )
                if cloud_input_ids.shape[1] < current_seq_len:
                    pad_len = current_seq_len - cloud_input_ids.shape[1]
                    pad = torch.zeros(
                        (1, pad_len), dtype=torch.long, device=self.engine.device
                    )
                    cloud_input_ids = torch.cat([cloud_input_ids, pad], dim=-1)

            # === DEBUG: print tensor shapes before tree_decoding ===
            print(f"[Cloud Debug] draft_token_ids shape: {draft_token_ids.shape}")
            print(f"[Cloud Debug] position_ids shape: {position_ids.shape}")
            print(f"[Cloud Debug] tree_attention_mask shape: {tree_attention_mask.shape}")
            print(f"[Cloud Debug] parent_last shape: {parent_last.shape}")
            # =======================================================

            # Tree decoding
            tree_logits, hidden_states, outputs = self.engine.tree_decoding(
                draft_input_ids=draft_token_ids,
                draft_position_ids=position_ids,
                tree_attention_mask=tree_attention_mask,
                retrieve_attn_scores=request_retrieval,
            )

            # Verify
            try:
                _, accepted_token_ids, accept_length, next_token, evicted = (
                    self.engine.verify(
                        input_ids=cloud_input_ids,
                        logits=tree_logits,
                        draft_input_ids=draft_token_ids,
                        draft_position_ids=position_ids,
                        tree_attention_mask=tree_attention_mask,
                        parent=parent_last,
                    )
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return

            # Compute retrieval chunk indices if requested
            retrieval_chunk_indices = []
            if request_retrieval:
                total_seq_len = current_seq_len + len(accepted_token_ids)
                retrieval_chunk_indices = self.engine.compute_retrieval_chunk_indices(
                    total_seq_len
                )

            # Determine if we should stop (next_token is EOS)
            need_stop = next_token.item() == self.engine.tokenizer.eos_token_id

            # Update cloud's local input_ids tracking
            if len(accepted_token_ids) > 0:
                accepted = torch.tensor(
                    accepted_token_ids, dtype=torch.long, device=self.engine.device
                ).unsqueeze(0)
                cloud_input_ids = torch.cat([cloud_input_ids, accepted], dim=-1)
            cloud_input_ids = torch.cat(
                [cloud_input_ids, next_token.to(self.engine.device)], dim=-1
            )

            yield draft_pb2.VerifyResult(
                accepted_token_ids=accepted_token_ids,
                accept_length=accept_length,
                next_token=next_token.item(),
                retrieval_chunk_indices=retrieval_chunk_indices,
                evicted=evicted,
                need_stop=need_stop,
                accepted_parent_ids=[],
            )
