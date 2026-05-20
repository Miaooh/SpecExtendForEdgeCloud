#!/bin/bash
# Edge client startup script

DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/home/xzh/models/vicuna-68m}"
CLOUD_ADDRESS="${CLOUD_ADDRESS:-localhost:50051}"
INPUT_FILE="${INPUT_FILE:-data/govreport/govreport_2K.jsonl}"
DEVICE="${DEVICE:-cuda}"

uv run python -m edge_server.main \
  --draft_model_path "$DRAFT_MODEL_PATH" \
  --cloud_address "$CLOUD_ADDRESS" \
  --input_file "$INPUT_FILE" \
  --use_specextend \
  --max_new_tokens 256 \
  --device "$DEVICE"
