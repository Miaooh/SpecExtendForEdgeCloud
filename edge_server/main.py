"""
Edge Server Entry Point.

Loads the small draft model, connects to the cloud target model,
and runs speculative decoding with the edge generating draft trees
and the cloud verifying them.
"""

import argparse
import json
import sys
import torch
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classic.modeling_llama_kv_draft import LlamaForCausalLM as KVLlamaForCausalLM_retrieval
from edge_server.engine import EdgeDraftEngine
from edge_server.client import EdgeClient


def load_texts_from_jsonl(path: str, max_samples: int = None):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode error at line {i}: {e}")
                continue
            text = obj.get("text")
            if text is None:
                continue
            texts.append(text)
    return texts


def run_generation(
    engine: EdgeDraftEngine,
    client: EdgeClient,
    input_ids: torch.LongTensor,
    nodes: int,
    threshold: float,
    max_depth: int,
    max_new_tokens: int,
    eos_token_id: int,
):
    """
    Main generation loop: draft on edge, verify via cloud stream.
    """
    current_input_ids = input_ids.clone()
    current_seq_len = input_ids.shape[1]
    new_token = 0
    accept_length_list = []
    first_request = True

    # Use queue-based pattern for bidirectional stream
    from queue import Queue
    import threading

    request_queue = Queue()
    response_queue = Queue()

    def request_iterator():
        while True:
            req = request_queue.get()
            if req is None:
                break
            yield req

    def rpc_thread():
        try:
            for resp in client.speculate_stream(request_iterator()):
                response_queue.put(resp)
        except Exception as e:
            response_queue.put(e)

    t = threading.Thread(target=rpc_thread, daemon=True)
    t.start()

    # First draft: prefill state
    draft_input_ids, draft_position_ids, tree_attention_mask, parent = engine.draft(
        current_input_ids,
        nodes=nodes,
        threshold=threshold,
        max_depth=max_depth,
    )

    # For the first request, the cloud will prepend its own first token.
    # For subsequent requests, we must prepend the last token (root) ourselves.
    request_retrieval = engine.should_request_retrieval()
    req = client.build_draft_request(
        draft_token_ids=draft_input_ids,
        position_ids=draft_position_ids,
        tree_attention_mask=tree_attention_mask,
        parent_last=parent,
        current_seq_len=current_seq_len,
        request_retrieval=request_retrieval,
        timestep=engine.timestep,
        prefix_token_ids=current_input_ids.squeeze(0) if first_request else None,
    )
    first_request = False
    request_queue.put(req)

    while True:
        resp = response_queue.get()
        if isinstance(resp, Exception):
            raise resp

        (
            accepted_token_ids,
            accept_length,
            next_token,
            retrieval_chunk_indices,
            need_stop,
        ) = client.parse_verify_result(resp)

        accept_length_list.append(accept_length)

        # Update local state
        current_input_ids = engine.advance_state(
            current_input_ids,
            accepted_token_ids,
            next_token,
            retrieval_chunk_indices=retrieval_chunk_indices if engine.use_retrieval_cache else None,
        )
        current_seq_len = current_input_ids.shape[1]
        new_token += accept_length + 1

        if need_stop or eos_token_id in current_input_ids[0, input_ids.shape[1]:].tolist():
            break
        if new_token > max_new_tokens:
            break

        # Generate next draft tree
        draft_input_ids, draft_position_ids, tree_attention_mask, parent = engine.draft(
            current_input_ids,
            nodes=nodes,
            threshold=threshold,
            max_depth=max_depth,
        )

        # Prepend root token (last token of current_input_ids) for cloud verification
        root_token = current_input_ids[:, -1:]
        draft_input_ids = torch.cat([root_token.to(draft_input_ids.device), draft_input_ids], dim=-1)
        position_ids = torch.cat([
            torch.tensor([draft_position_ids[0].item() - 1], dtype=torch.long, device=draft_position_ids.device),
            draft_position_ids
        ], dim=-1)
        tree_attention_mask = torch.cat([
            torch.zeros(1, tree_attention_mask.size(1), dtype=tree_attention_mask.dtype, device=tree_attention_mask.device),
            tree_attention_mask
        ], dim=0)
        tree_attention_mask = torch.cat([
            torch.ones(tree_attention_mask.size(0), 1, dtype=tree_attention_mask.dtype, device=tree_attention_mask.device),
            tree_attention_mask
        ], dim=-1)
        parent = torch.cat([
            torch.tensor([0], dtype=torch.long, device=parent.device),
            parent + 1
        ], dim=-1)

        request_retrieval = engine.should_request_retrieval()
        req = client.build_draft_request(
            draft_token_ids=draft_input_ids,
            position_ids=position_ids,
            tree_attention_mask=tree_attention_mask,
            parent_last=parent,
            current_seq_len=current_seq_len,
            request_retrieval=request_retrieval,
            timestep=engine.timestep,
        )
        request_queue.put(req)

    request_queue.put(None)
    t.join(timeout=5.0)

    return current_input_ids, accept_length_list


def main():
    parser = argparse.ArgumentParser(description="SpecExtend Edge Server")
    parser.add_argument(
        "--draft_model_path",
        type=str,
        default="/home/xzh/models/vicuna-68m",
        help="Path to the small draft model (edge side)",
    )
    parser.add_argument(
        "--cloud_address",
        type=str,
        default="localhost:50051",
        help="Cloud gRPC server address",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file with 'text' field",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--use_specextend",
        action="store_true",
        help="Enable SpecExtend retrieval cache",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=50,
        help="Tree width (number of nodes per layer)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Tree expansion threshold",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Max tree depth",
    )
    parser.add_argument(
        "--retrieval_chunk_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--retrieve_top_k",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--retrieve_every_n_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[Edge] Loading draft model from {args.draft_model_path} ...")
    draft_model = KVLlamaForCausalLM_retrieval.from_pretrained(
        args.draft_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto" if args.device == "cuda" else None,
    ).eval()
    if args.device != "cuda":
        draft_model = draft_model.to(device)

    engine = EdgeDraftEngine(
        draft_model=draft_model,
        draft_model_path=args.draft_model_path,
        device=device,
        use_retrieval_cache=args.use_specextend,
        retrieval_chunk_size=args.retrieval_chunk_size,
        retrieve_top_k=args.retrieve_top_k,
        retrieve_every_n_steps=args.retrieve_every_n_steps,
    )

    print(f"[Edge] Connecting to cloud at {args.cloud_address} ...")
    client = EdgeClient(cloud_address=args.cloud_address)
    client.connect()

    texts = load_texts_from_jsonl(args.input_file, max_samples=args.max_samples)
    if not texts:
        print("[Edge] No valid texts loaded.")
        return

    for idx, text in enumerate(texts):
        print(f"\n[Edge] === Sample {idx + 1}/{len(texts)} ===")
        input_ids = engine.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True
        ).to(device)

        engine.init_generation(input_ids, max_new_tokens=args.max_new_tokens)

        final_ids, accept_lengths = run_generation(
            engine=engine,
            client=client,
            input_ids=input_ids,
            nodes=args.nodes,
            threshold=args.threshold,
            max_depth=args.max_depth,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=engine.tokenizer.eos_token_id,
        )

        output_text = engine.tokenizer.decode(
            final_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(f"[Edge] Output length: {final_ids.shape[1] - input_ids.shape[1]} tokens")
        if accept_lengths:
            avg_accept = sum(accept_lengths) / len(accept_lengths)
            print(f"[Edge] Avg accept length: {avg_accept:.3f}")

    client.close()
    print("[Edge] Done.")


if __name__ == "__main__":
    main()
