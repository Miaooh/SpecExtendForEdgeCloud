"""
EdgeDraftEngine: Runs on the edge device.
Responsible for:
  - Loading and running the small draft model (e.g., vicuna-68m)
  - Managing the draft KV cache (full + working cache)
  - Generating the draft token tree via Tree search
  - Updating retrieval cache based on cloud-returned chunk indices
  - Maintaining generation state (input_ids, timestep, etc.)
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from shared.opt_tree import Tree
from shared.kv_cache import initialize_past_key_values
from transformers import AutoTokenizer


class EdgeDraftEngine:
    def __init__(
        self,
        draft_model: nn.Module,
        draft_model_path: str,
        device: torch.device,
        use_retrieval_cache: bool = True,
        retrieval_chunk_size: int = 32,
        retrieve_top_k: int = 32,
        retrieve_every_n_steps: int = 4,
    ):
        self.draft_model = draft_model
        self.draft_model_path = draft_model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(draft_model_path)

        self.use_retrieval_cache = use_retrieval_cache
        self.retrieval_chunk_size = retrieval_chunk_size
        self.retrieve_top_k = retrieve_top_k
        self.retrieve_every_n_steps = retrieve_every_n_steps

        # Cache management attributes (populated in init_generation)
        self.full_draft_kv: List[Tuple[torch.Tensor, torch.Tensor]] = None
        self.draft_stable_kv = None
        self.total_seq_len = 0
        self.seq_len_total_old = 0
        self.evicted = 0
        self.num_chunks_old = 0
        self.retrieval_condition = False
        self.chunks = None
        self.selected_chunks = None
        self.timestep = 0
        self.full_cache_budget = 0

        # Draft model device tracking
        self.draft_device = next(draft_model.parameters()).device

    def init_generation(self, input_ids: torch.LongTensor, max_new_tokens: int):
        """Initialize caches and state for a new generation session."""
        self.timestep = 0
        self.total_seq_len = 0
        self.seq_len_total_old = 0
        self.evicted = 0
        self.num_chunks_old = 0
        self.retrieval_condition = False
        self.chunks = None
        self.selected_chunks = None
        input_len = input_ids.shape[1]
        self.full_cache_budget = input_len + max_new_tokens + 100

        if self.use_retrieval_cache:
            self._init_caches()
        else:
            self.draft_stable_kv = None
            self.full_draft_kv = None
            self.evicted = 0

    def _init_caches(self):
        """Preallocate full and working draft KV caches."""
        num_hidden_layers = self.draft_model.config.num_hidden_layers
        num_heads = self.draft_model.config.num_attention_heads
        head_dim = self.draft_model.config.hidden_size // num_heads

        self.full_draft_kv = []
        for _ in range(num_hidden_layers):
            full_K = torch.zeros(
                [1, num_heads, self.full_cache_budget, head_dim],
                dtype=torch.float16,
                device=self.draft_device,
            )
            full_V = torch.zeros(
                [1, num_heads, self.full_cache_budget, head_dim],
                dtype=torch.float16,
                device=self.draft_device,
            )
            self.full_draft_kv.append((full_K, full_V))

        self.total_seq_len = 0
        self.seq_len_total_old = 0
        self.evicted = 0
        self.draft_stable_kv = None
        self.chunks = None
        self.draft_model.model.past_key_position_ids = None
        self.recent_start = 0
        self.recent_end = 0

    def _update_full_draft_cache(self, new_kv: List[Tuple[torch.Tensor, torch.Tensor]], tokens_appended: int):
        """Update the full draft KV cache with newly generated tokens."""
        if self.total_seq_len + tokens_appended > self.full_cache_budget:
            raise RuntimeError(
                f"Full cache budget exceeded: {self.total_seq_len} + {tokens_appended} > {self.full_cache_budget}"
            )

        dest_start = self.total_seq_len
        dest_end = dest_start + tokens_appended
        device = self.draft_device

        for i, (new_K, new_V) in enumerate(new_kv):
            full_K, full_V = self.full_draft_kv[i]
            new_K = new_K.to(device, non_blocking=True)
            new_V = new_V.to(device, non_blocking=True)
            full_K[:, :, dest_start:dest_end, :].copy_(new_K[:, :, -tokens_appended:, :])
            full_V[:, :, dest_start:dest_end, :].copy_(new_V[:, :, -tokens_appended:, :])

        self.total_seq_len = dest_end

    def _prepare_chunks(self):
        """Split full cache into chunks of size retrieval_chunk_size."""
        self.chunks = []
        current_start = 0
        chunk_idx = 0
        while current_start < self.total_seq_len:
            end_pos = min(current_start + self.retrieval_chunk_size, self.total_seq_len)
            self.chunks.append((chunk_idx, current_start, end_pos))
            chunk_idx += 1
            current_start = end_pos
        self.seq_len_total_old = self.total_seq_len
        self.num_chunks = len(self.chunks)
        self.num_chunks_old = self.num_chunks

    def _update_chunks(self) -> bool:
        """Update chunk metadata after new tokens appended. Returns True if new chunk added."""
        new_tokens = self.total_seq_len - self.seq_len_total_old
        if new_tokens <= 0:
            return False

        if not self.chunks or len(self.chunks) == 0:
            self._prepare_chunks()
            return True

        last_chunk_idx, last_start, last_end = self.chunks[-1]
        last_chunk_size = last_end - last_start
        remaining_new_tokens = new_tokens

        capacity = self.retrieval_chunk_size - last_chunk_size
        if capacity > 0:
            tokens_to_add = min(capacity, remaining_new_tokens)
            self.chunks[-1] = (last_chunk_idx, last_start, last_end + tokens_to_add)
            remaining_new_tokens -= tokens_to_add

        current_start = self.total_seq_len - remaining_new_tokens
        while remaining_new_tokens > 0:
            tokens_in_chunk = min(self.retrieval_chunk_size, remaining_new_tokens)
            new_chunk = (self.chunks[-1][0] + 1, current_start, current_start + tokens_in_chunk)
            self.chunks.append(new_chunk)
            current_start += tokens_in_chunk
            remaining_new_tokens -= tokens_in_chunk

        self.seq_len_total_old = self.total_seq_len
        self.num_chunks = len(self.chunks)

        if self.num_chunks > self.num_chunks_old:
            self.num_chunks_old = self.num_chunks
            return True
        return False

    def update_working_cache_from_retrieval(
        self,
        retrieval_chunk_indices: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Build working cache from full cache using cloud-provided chunk indices.
        This replaces the local attention-score-based selection.
        """
        if self.chunks is None:
            raise ValueError("Chunks not initialized. Call _prepare_chunks first.")

        # If cloud provides explicit chunk indices, select those chunks
        if retrieval_chunk_indices and len(retrieval_chunk_indices) > 0:
            selected_chunks = []
            for idx in retrieval_chunk_indices:
                if 0 <= idx < len(self.chunks):
                    selected_chunks.append(self.chunks[idx])
            # Sort by chunk id to maintain order
            selected_chunks.sort(key=lambda x: x[0])
            self.selected_chunks = selected_chunks
        else:
            # Fallback: use recent chunks if no retrieval signal
            if not hasattr(self, "selected_chunks") or self.selected_chunks is None:
                num_init = min(self.retrieve_top_k, len(self.chunks))
                self.selected_chunks = self.chunks[-num_init:]

        # Update last selected chunk to match current full cache end
        if self.selected_chunks and self.chunks:
            if self.selected_chunks[-1][0] == self.chunks[-1][0]:
                chunk_id, start, _ = self.selected_chunks[-1]
                new_end = self.chunks[-1][2]
                self.selected_chunks[-1] = (chunk_id, start, new_end)

        all_indices = []
        for (_, start, end) in self.selected_chunks:
            all_indices.extend(range(start, end))

        if len(all_indices) == 0:
            raise ValueError("No tokens retrieved from the full cache.")

        retrieved_indices = torch.tensor(all_indices, dtype=torch.long)
        retrieved_indices = torch.unique(retrieved_indices, sorted=True).to(self.draft_device)

        if retrieved_indices.numel() == 0:
            raise ValueError("No tokens retrieved from the full cache.")

        working_kv = []
        for (full_K, full_V) in self.full_draft_kv:
            working_K_layer = full_K.index_select(dim=2, index=retrieved_indices)
            working_V_layer = full_V.index_select(dim=2, index=retrieved_indices)
            working_kv.append((working_K_layer, working_V_layer))
        self.draft_stable_kv = working_kv

        self.evicted = self.total_seq_len - retrieved_indices.numel()

        # Update past_key_position_ids
        past_ids = self.draft_model.model.past_key_position_ids
        if past_ids is not None:
            current_length = past_ids.shape[1]
            target_length = retrieved_indices.numel()
            if current_length < target_length:
                extra_ids = torch.arange(current_length, target_length, device=past_ids.device).unsqueeze(0)
                new_past_ids = torch.cat([past_ids, extra_ids], dim=1)
            else:
                new_past_ids = past_ids[:, :target_length]
            self.draft_model.model.past_key_position_ids = new_past_ids
        else:
            self.draft_model.model.past_key_position_ids = torch.arange(
                0, retrieved_indices.numel(), device=self.draft_device
            ).unsqueeze(0)

        return working_kv

    def update_working_cache_main(self, retrieval_chunk_indices: List[int] = None):
        """Convenience: update chunks then rebuild working cache."""
        is_updated_chunks = self._update_chunks()

        if retrieval_chunk_indices is not None and len(retrieval_chunk_indices) > 0:
            self.update_working_cache_from_retrieval(retrieval_chunk_indices)
        else:
            # No explicit signal: auto-append new chunk if added
            if is_updated_chunks and self.chunks:
                new_chunk = self.chunks[-1]
                new_chunk_id = new_chunk[0]
                if self.selected_chunks is not None:
                    existing_ids = {cid for cid, _, _ in self.selected_chunks}
                    if new_chunk_id not in existing_ids:
                        self.selected_chunks.append(new_chunk)
                else:
                    self.selected_chunks = [new_chunk]
            self.update_working_cache_from_retrieval([])

    @torch.no_grad()
    def draft(
        self,
        input_ids: torch.LongTensor,
        nodes: int = 50,
        threshold: float = 0.7,
        max_depth: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a draft token tree.

        Returns:
            draft_input_ids: [1, tree_len] token IDs
            draft_position_ids: [tree_len] position IDs
            tree_attention_mask: [tree_len, tree_len] attention mask
            parent_last: [tree_len] parent indices
        """
        len_posi = input_ids.shape[1] - 1

        # Initial forward or continuation with stable KV
        if hasattr(self, "draft_stable_kv") and self.draft_stable_kv is not None:
            if self.use_retrieval_cache:
                full_kv_len = self.total_seq_len
                draft_outputs = self.draft_model.model(
                    input_ids=input_ids[:, full_kv_len:].to(self.draft_device),
                    past_key_values=self.draft_stable_kv,
                    return_kv=True,
                    draft_use_flash_prefill=self.use_retrieval_cache,
                )
            else:
                kv_len = self.draft_stable_kv[0][0].shape[2]
                draft_outputs = self.draft_model.model(
                    input_ids=input_ids[:, kv_len:].to(self.draft_device),
                    past_key_values=self.draft_stable_kv,
                    return_kv=True,
                    draft_use_flash_prefill=self.use_retrieval_cache,
                )
        else:
            # Prefill
            draft_outputs = self.draft_model.model(
                input_ids=input_ids.to(self.draft_device),
                return_kv=True,
                init=True,
                draft_use_flash_prefill=self.use_retrieval_cache,
            )

        if self.use_retrieval_cache:
            newly_appended_len = input_ids.shape[-1] - self.total_seq_len
            self._update_full_draft_cache(draft_outputs[1], tokens_appended=newly_appended_len)
            self.update_working_cache_main()
        else:
            self.draft_stable_kv = draft_outputs[1]

        past_key_values = self.draft_stable_kv
        init_len = past_key_values[0][0].size(2)
        target_model_pos_diff = len_posi - (init_len - 1)

        last_hidden = draft_outputs[0][:, -1]
        last_headout = self.draft_model.lm_head(last_hidden)

        tree = Tree(nodes, last_hidden.device, threshold, max_depth)
        logits = last_headout.unsqueeze(0)

        step = 0
        while True:
            tree_output = tree.update(
                torch.softmax(logits.to(last_hidden.device), dim=-1, dtype=torch.float32)
            )

            draft_input_ids = tree_output["input_ids"].unsqueeze(0)

            if self.use_retrieval_cache:
                position_ids = tree_output["position_ids"] + init_len - 1
            else:
                position_ids = tree_output["position_ids"] + len_posi

            if tree_output["is_final"]:
                break

            tree_attention_mask_with_kv = self._process_tree_mask(
                tree_output["attention_mask"], init_len
            )

            draft_outputs = self.draft_model.model(
                input_ids=draft_input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                tree_attention_mask=tree_attention_mask_with_kv,
                return_kv=True,
                draft_use_flash_prefill=self.use_retrieval_cache,
            )

            past_key_values = draft_outputs[1]
            last_hidden = draft_outputs[0]
            last_headout = self.draft_model.lm_head(last_hidden)
            logits = last_headout
            step += 1

        if self.use_retrieval_cache:
            position_ids += target_model_pos_diff

        return draft_input_ids, position_ids, tree_output["attention_mask"], tree_output["parent_last"]

    def _process_tree_mask(self, tree_attention_mask, init_len):
        attention_mask = torch.full(
            (tree_attention_mask.size(0), init_len), 0, device=tree_attention_mask.device
        )
        tree_mask = torch.where(
            tree_attention_mask == 0,
            torch.finfo(torch.float32).min,
            0,
        )
        attention_mask = torch.cat([attention_mask, tree_mask], dim=-1)
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    def advance_state(
        self,
        input_ids: torch.LongTensor,
        accepted_token_ids: List[int],
        next_token: int,
        retrieval_chunk_indices: List[int] = None,
    ) -> torch.LongTensor:
        """
        Update local state after receiving verification result from cloud.
        Returns updated input_ids.
        """
        # Append accepted tokens
        if len(accepted_token_ids) > 0:
            accepted = torch.tensor(
                accepted_token_ids, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
            input_ids = torch.cat([input_ids, accepted], dim=-1)

        # Append next token (resampled)
        next_t = torch.tensor([[next_token]], dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([input_ids, next_t], dim=-1)

        # Update retrieval cache if enabled
        if self.use_retrieval_cache:
            self.update_working_cache_main(retrieval_chunk_indices)

        self.timestep += 1
        return input_ids

    def should_request_retrieval(self) -> bool:
        """Check if this timestep should trigger retrieval attention score computation."""
        if not self.use_retrieval_cache:
            return False
        return self.timestep % self.retrieve_every_n_steps == 0
