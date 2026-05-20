# SpecExtendForEdgeCloud

Edge-Cloud deployment of [SpecExtend](https://github.com/jycha98/SpecExtend): a speculative decoding framework where the **draft model runs on the edge** and the **target model runs in the cloud**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EDGE (边端)                                     │
│  ┌─────────────────────┐      ┌─────────────────────┐                        │
│  │   Draft Model       │      │  Draft KV Cache     │                        │
│  │   (Vicuna-68M)      │◄────►│  (Full + Working)   │                        │
│  │   + Tree Generator  │      │  + Chunk Manager    │                        │
│  └──────────┬──────────┘      └─────────────────────┘                        │
│             │                                                                │
│             │  gRPC bidirectional stream                                      │
│             ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    EdgeClient (gRPC)                        │            │
│  └─────────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ low-latency network
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLOUD (云端)                                    │
│  ┌─────────────────────┐      ┌─────────────────────┐                        │
│  │   Target Model      │      │  Target KV Cache    │                        │
│  │   (Vicuna-7B)       │◄────►│  (Preallocated)     │                        │
│  │   + Tree Attention  │      │                     │                        │
│  └──────────┬──────────┘      └─────────────────────┘                        │
│             │                                                                │
│             │  output: Tree Logits → Verify → Accept/Reject                  │
│             │  output: Attention Scores → Retrieval Chunk Selector           │
│             ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    SpecExtendService (gRPC)                 │            │
│  └─────────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
SpecExtendForEdgeCloud/
├── protocol/
│   ├── draft.proto              # gRPC protocol definition
│   ├── draft_pb2.py             # Generated protobuf Python code
│   └── draft_pb2_grpc.py        # Generated gRPC Python code
├── edge_server/
│   ├── engine.py                # EdgeDraftEngine: draft model + cache management
│   ├── client.py                # gRPC client to cloud
│   └── main.py                  # Edge entry point
├── cloud_server/
│   ├── engine.py                # CloudVerifyEngine: target model + verification
│   ├── service.py               # gRPC servicer implementation
│   └── main.py                  # Cloud entry point
├── shared/
│   ├── kv_cache.py              # KVCache implementation (reused from SpecExtend)
│   ├── opt_tree.py              # Tree search for draft generation (reused)
│   ├── triton_tree_attn.py      # Tree attention kernels (reused)
│   └── modeling_llama_kv_target.py  # Modified Llama target model (reused)
├── classic/
│   └── modeling_llama_kv_draft.py   # Modified Llama draft model (reused)
├── pyproject.toml               # uv-based dependency management
├── README.md
└── .gitignore
```

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd SpecExtendForEdgeCloud

# Sync dependencies (excluding optional flash-attn)
uv sync

# If you need FlashAttention (cloud side with CUDA)
uv sync --extra flash
```

## Usage

### 1. Start the Cloud Server

On the **cloud machine** with the large target model:

```bash
uv run python -m cloud_server.main \
  --target_model_path /path/to/vicuna-7b-v1.5-16k \
  --port 50051 \
  --use_specextend \
  --device cuda
```

### 2. Start the Edge Client

On the **edge machine** with the small draft model:

```bash
uv run python -m edge_server.main \
  --draft_model_path /home/xzh/models/vicuna-68m \
  --cloud_address <cloud-ip>:50051 \
  --input_file data/govreport/govreport_2K.jsonl \
  --use_specextend \
  --max_new_tokens 256 \
  --device cuda
```

### Arguments

**Edge (`edge_server.main`)**
| Argument | Default | Description |
|----------|---------|-------------|
| `--draft_model_path` | `/home/xzh/models/vicuna-68m` | Path to draft model |
| `--cloud_address` | `localhost:50051` | Cloud gRPC endpoint |
| `--input_file` | *required* | Input JSONL with `"text"` field |
| `--use_specextend` | `False` | Enable retrieval cache |
| `--nodes` | `50` | Tree width |
| `--threshold` | `0.7` | Tree expansion threshold |
| `--max_depth` | `10` | Max tree depth |
| `--retrieval_chunk_size` | `32` | Chunk size for retrieval |
| `--retrieve_top_k` | `32` | Top-k chunks to retrieve |
| `--retrieve_every_n_steps` | `4` | Retrieval period |

**Cloud (`cloud_server.main`)**
| Argument | Default | Description |
|----------|---------|-------------|
| `--target_model_path` | *required* | Path to target model |
| `--port` | `50051` | gRPC listen port |
| `--use_specextend` | `False` | Compute retrieval attention scores |
| `--retrieval_chunk_size` | `32` | Chunk size for retrieval |
| `--retrieve_top_k` | `32` | Top-k chunks to select |
| `--max_workers` | `10` | gRPC thread pool size |

## How It Works

1. **Prefill (both sides)**: Edge and Cloud independently run prefill on the input prompt to initialize their respective KV caches.
2. **Draft (Edge)**: The edge runs the small draft model to generate a token tree using the `Tree` search algorithm.
3. **Send to Cloud**: The draft tree (token IDs, position IDs, attention mask, parent indices) is serialized and sent via gRPC.
4. **Tree Decoding (Cloud)**: The cloud runs the large target model once over the entire draft tree.
5. **Verify (Cloud)**: Draft tokens are verified against target logits. The longest matching branch is accepted, plus one resampled token.
6. **Retrieval (Cloud, optional)**: If `use_specextend` is enabled, the cloud computes attention scores and selects top-k chunks.
7. **Return Result**: Cloud sends back accepted tokens, resampled token, and chunk indices.
8. **Update (Edge)**: The edge updates its local sequence and rebuilds the working draft KV cache using the retrieved chunks.
9. **Repeat**: Go back to step 2 until EOS or max tokens.

## Key Design Decisions

- **Bidirectional gRPC streaming**: `SpeculateDraft` is a `stream-stream` RPC. This allows the edge to pipeline draft generation while waiting for network latency.
- **Chunk indices over attention scores**: To minimize bandwidth, the cloud computes top-k chunk indices locally and only sends the integer list back. The edge never sees raw attention scores.
- **Independent KV caches**: Both sides maintain their own KV caches. Only token IDs and control signals cross the network—never large KV tensors.
- **No hidden-state transfer**: The Classic draft model is fully independent, so no target hidden states need to cross the network (unlike EAGLE).

## Limitations

- **Batch size 1 only**: The original SpecExtend only supports batch size 1; this port preserves that constraint.
- **Classic draft only**: EAGLE-style drafts require target hidden states and are not supported in this edge-cloud split.
- **gRPC over insecure channel**: Current implementation uses `grpc.insecure_channel`. For production, add TLS/mTLS.

## License

This project inherits the license of the original SpecExtend repository.
