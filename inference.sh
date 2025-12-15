#!/bin/bash
set -e

# Ép chạy CPU để ổn định (không phụ thuộc GPU/VRAM).
export CUDA_VISIBLE_DEVICES=""
export BGE_DEVICE="cpu"

LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-15}"
LLM_MODEL="${LLM_MODEL:-auto}"
python predict.py --input "/code/private_test.json" --output "submission.csv" --llm_batch_size "${LLM_BATCH_SIZE}" --llm_model "${LLM_MODEL}"

echo "[DONE] Output: submission.csv"
