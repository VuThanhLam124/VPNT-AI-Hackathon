#!/bin/bash
set -e

# Local LLM (Transformers). Model id mặc định set trong `predict.py`:
#   LOCAL_LLM_MODEL_ID=Qwen/Qwen3-8B-Base
# Có thể override:
#   LOCAL_LLM_MODEL_ID=/code/models/Qwen3-8B-Base
#   LOCAL_LLM_4BIT=1
#   LOCAL_LLM_MAX_NEW_TOKENS=16

python predict.py --input "/code/private_test.json" --output "submission.csv"

echo "[DONE] Output: submission.csv"
