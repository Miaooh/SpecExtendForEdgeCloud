"""
CloudVerifyEngine: Runs on the cloud server.
Responsible for:
  - Loading and running the large target model (e.g., vicuna-7b)
  - Managing the target KV cache
  - Tree-decoding the draft token tree in one forward pass
  - Verifying draft tokens against target logits
  - Computing attention scores and selecting top-k chunks for retrieval
  - Returning accepted tokens + retrieval signals to the edge
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from shared.kv_cache import initialize_past_key_values
from transformers import AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)


def prepare_logits_processor(
    temperature: float = 0.0,
    repetition_penalty: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


class CloudVerifyEngine:
    def __init__(
        self,
        target_model: nn.Module,
        target_model_path: str,
        device: torch.device,
        use_retrieval_cache: bool = True,
        retrieval_chunk_size: int = 32,
        retrieve_top_k: int = 32,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
    ):
        self.target_model = target_model
        self.target_model_path = target_model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_path)

        self.use_retrieval_cache = use_retrieval_cache
        self.retrieval_chunk_size = retrieval_chunk_size
        self.retrieve_top_k = retrieve_top_k
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        # KV cache for target model
        self.past_key_values = None
        self.past_key_values_data = None
        self.current_length_data = None
        self.full_cache_budget = 0

        # Attention scores for retrieval
        self.attn_scores = None
        self.attn_scores_final = None

    def init_generation(self, input_ids: torch.LongTensor, max_new_tokens: int):
        """Initialize target model KV cache for a new generation session."""
        input_len = input_ids.shape[1]
        self.full_cache_budget = input_len + max_new_tokens + 100

        (
            self.past_key_values,
            self.past_key_values_data,
            self.current_length_data,
        ) = initialize_past_key_values(self.target_model, self.full_cache_budget)

        self.attn_scores = None
        self.attn_scores_final = None

        if self.temperature > 1e-5:
            self.logits_processor = prepare_logits_processor(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        else:
            self.logits_processor = None

    def prefill(self, input_ids: torch.LongTensor):
        """Run target model prefill and return the first token + hidden states."""
        outputs = self.target_model.model(
            input_ids=input_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_kv=True,
            init=True,
            target_use_flash_prefill=self.use_retrieval_cache,
            target_use_hybrid_tree_attn=self.use_retrieval_cache,
        )
        orig_logits = self.target_model.lm_head(outputs[0])

        if self.logits_processor is not None:
            logits_last = self.logits_processor(None, orig_logits[:, -1])
            probabilities = torch.nn.functional.softmax(logits_last, dim=1)
            token = torch.multinomial(probabilities, 1)
        else:
            token = torch.argmax(orig_logits[:, -1])
            token = token[None, None]

        return token, outputs

    @torch.no_grad()
    def tree_decoding(
        self,
        draft_input_ids: torch.LongTensor,
        draft_position_ids: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        retrieve_attn_scores: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run target model forward on the draft token tree.

        Returns:
            tree_logits: logits for each draft position
            hidden_states: hidden states
            outputs: raw model outputs (contains attentions if retrieve_attn_scores)
        """
        outputs = self.target_model.model(
            input_ids=draft_input_ids,
            tree_attention_mask=tree_attention_mask,
            past_key_values=self.past_key_values,
            position_ids=draft_position_ids,
            output_attentions=True,
            init=False,
            target_use_flash_prefill=self.use_retrieval_cache,
            target_use_hybrid_tree_attn=self.use_retrieval_cache,
            retrieve_attn_scores=retrieve_attn_scores,
        )

        tree_logits = self.target_model.lm_head(outputs[0])
        hidden_states = outputs[0].clone()

        if self.use_retrieval_cache and retrieve_attn_scores:
            # Last layer attention scores: [bsz, num_heads, q_len, kv_seq_len]
            self.attn_scores = outputs.attentions[-1]

        return tree_logits, hidden_states, outputs

    @torch.no_grad()
    def verify(
        self,
        input_ids: torch.LongTensor,
        logits: torch.Tensor,
        draft_input_ids: torch.LongTensor,
        draft_position_ids: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        parent: torch.Tensor,
    ) -> Tuple[torch.LongTensor, List[int], int, torch.Tensor, int]:
        """
        Verify draft tokens against target logits.

        Returns:
            input_ids: updated input_ids with accepted + resampled token
            accepted_token_ids: list of accepted token IDs (excluding resampled)
            accept_length: number of accepted tokens
            next_token: resampled token ID tensor [1,1]
            evicted: evicted count (informational)
        """
        if self.logits_processor is None:
            next_token_all = torch.argmax(logits, dim=-1)
        else:
            logits_proc = self.logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits_proc, dim=-1)[0]
            next_token_all = torch.multinomial(probabilities, 1).view(1, -1)

        next_token_all = next_token_all.to(draft_input_ids.device)

        # Reconstruct parent indices for tree matching
        parent = torch.where(
            parent == torch.arange(parent.size(0), device=parent.device),
            -1,
            parent,
        )
        parent = torch.cat(
            [torch.tensor([0], device=parent.device), parent + 1], dim=-1
        ).to(draft_input_ids.device)

        correct = torch.where(
            draft_input_ids[0] != next_token_all[0][parent],
            0,
            torch.ones(draft_input_ids.size(1), device=draft_input_ids.device),
        )
        correct[0] = 1

        last_sum = torch.sum(correct)
        while True:
            correct = torch.where(correct[parent] == 0, 0, correct)
            if torch.sum(correct) == last_sum:
                break
            last_sum = torch.sum(correct)

        id = torch.argmax(correct * draft_position_ids)
        best_candidate = []
        best_candidate_id = []
        max_id = id
        parent[0] = -1
        while id != -1:
            best_candidate.append(draft_input_ids[0][id].item())
            best_candidate_id.append(id)
            id = parent[id].item()

        best_candidate.reverse()
        best_candidate_id.reverse()
        next_token = next_token_all[0][max_id].unsqueeze(0).unsqueeze(0)
        accept_length = len(best_candidate) - 1

        # Compute final attention scores for retrieval
        if self.use_retrieval_cache and self.attn_scores is not None:
            prev_input_len = input_ids.shape[1]
            last_query_index = best_candidate_id[-1]
            # Mean over attention heads
            last_attn_scores = (
                self.attn_scores[:, :, last_query_index, :].mean(dim=1).squeeze()
            )
            best_candidate_id_abs = torch.tensor(
                best_candidate_id, device=last_attn_scores.device
            ) + prev_input_len
            self.attn_scores_final = torch.cat(
                (
                    last_attn_scores[:prev_input_len],
                    last_attn_scores[best_candidate_id_abs],
                ),
                dim=0,
            )

        # Update target KV cache: select accepted branch positions
        start = self.current_length_data[0].item() - draft_input_ids.size(1)
        select_indices = torch.tensor(best_candidate_id) + start

        for data in self.past_key_values_data:
            tgt = data[..., select_indices.to(data.device), :]
            dst = data[..., start : start + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)

        self.current_length_data.fill_(start + tgt.shape[-2])

        # Append accepted tokens + next token to input_ids
        if len(best_candidate) > 0:
            accepted_tensor = torch.tensor(
                best_candidate, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
            input_ids = torch.cat([input_ids, accepted_tensor], dim=-1)
        input_ids = torch.cat([input_ids, next_token.to(input_ids.device)], dim=-1)

        evicted = 0  # Cloud doesn't evict; edge handles draft cache eviction

        # Clear attn_scores after use
        self.attn_scores = None

        return input_ids, best_candidate, accept_length, next_token, evicted

    def compute_retrieval_chunk_indices(self, total_seq_len: int) -> List[int]:
        """
        Compute top-k chunk indices from self.attn_scores_final.
        Called by the service layer after verify() when retrieval is requested.
        """
        if self.attn_scores_final is None:
            return []

        attn = self.attn_scores_final
        n_chunks = (total_seq_len + self.retrieval_chunk_size - 1) // self.retrieval_chunk_size
        if n_chunks == 0:
            return []

        chunks = []
        current_start = 0
        chunk_idx = 0
        while current_start < total_seq_len:
            end_pos = min(current_start + self.retrieval_chunk_size, total_seq_len)
            chunks.append((chunk_idx, current_start, end_pos))
            chunk_idx += 1
            current_start = end_pos

        chunks_tensor = torch.tensor(
            [[start, end] for (_, start, end) in chunks],
            dtype=torch.long,
            device=attn.device,
        )
        starts = chunks_tensor[:, 0]
        ends = chunks_tensor[:, 1]

        cum_attn = torch.cumsum(attn, dim=0)
        lower = torch.where(
            starts > 0,
            cum_attn[starts - 1],
            torch.zeros_like(starts, dtype=attn.dtype),
        )
        ends_minus_one = torch.clamp(ends - 1, max=cum_attn.size(0) - 1)
        chunk_sums = cum_attn[ends_minus_one] - lower

        lengths = (ends - starts).float()
        chunk_means = chunk_sums / lengths

        k = min(self.retrieve_top_k, chunk_means.size(0))
        topk = torch.topk(chunk_means, k=k)
        selected_indices = topk.indices.tolist()

        # Reset for next cycle
        self.attn_scores_final = None

        return selected_indices
