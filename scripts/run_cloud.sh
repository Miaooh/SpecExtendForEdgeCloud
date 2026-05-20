#!/bin/bash
# Cloud server startup script

TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/path/to/vicuna-7b-v1.5-16k}"
PORT="${PORT:-50051}"
DEVICE="${DEVICE:-cuda}"

uv run python -m cloud_server.main \
  --target_model_path "$TARGET_MODEL_PATH" \
  --port "$PORT" \
  --use_specextend \
  --device "$DEVICE"
