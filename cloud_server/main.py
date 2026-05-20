"""
Cloud Server Entry Point.

Loads the large target model, starts a gRPC server,
and waits for Edge clients to send draft trees for verification.
"""

import argparse
import grpc
import sys
import torch
from concurrent import futures
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.modeling_llama_kv_target import LlamaForCausalLM as KVLlamaForCausalLM
from cloud_server.engine import CloudVerifyEngine
from cloud_server.service import SpecExtendServicer

from protocol import draft_pb2_grpc


def serve(
    target_model_path: str,
    port: int,
    device: torch.device,
    use_retrieval_cache: bool = True,
    retrieval_chunk_size: int = 32,
    retrieve_top_k: int = 32,
    max_workers: int = 10,
):
    print(f"[Cloud] Loading target model from {target_model_path} ...")
    target_model = KVLlamaForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto" if str(device) == "cuda" else None,
    ).eval()
    if str(device) != "cuda":
        target_model = target_model.to(device)

    engine = CloudVerifyEngine(
        target_model=target_model,
        target_model_path=target_model_path,
        device=device,
        use_retrieval_cache=use_retrieval_cache,
        retrieval_chunk_size=retrieval_chunk_size,
    )
    # Attach retrieve_top_k if not already set
    engine.retrieve_top_k = retrieve_top_k

    servicer = SpecExtendServicer(engine)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    draft_pb2_grpc.add_SpecExtendServiceServicer_to_server(servicer, server)
    bind_address = f"[::]:{port}"
    server.add_insecure_port(bind_address)

    print(f"[Cloud] gRPC server listening on {bind_address}")
    server.start()
    print("[Cloud] Waiting for edge connections...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[Cloud] Shutting down...")
        server.stop(grace_period=5)


def main():
    parser = argparse.ArgumentParser(description="SpecExtend Cloud Server")
    parser.add_argument(
        "--target_model_path",
        type=str,
        required=True,
        help="Path to the large target model (cloud side)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="gRPC server port",
    )
    parser.add_argument(
        "--use_specextend",
        action="store_true",
        help="Enable SpecExtend retrieval cache computation",
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
        "--max_workers",
        type=int,
        default=10,
        help="gRPC thread pool size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    serve(
        target_model_path=args.target_model_path,
        port=args.port,
        device=device,
        use_retrieval_cache=args.use_specextend,
        retrieval_chunk_size=args.retrieval_chunk_size,
        retrieve_top_k=args.retrieve_top_k,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
