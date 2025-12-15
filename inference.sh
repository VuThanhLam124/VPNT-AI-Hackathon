#!/bin/bash
set -e

# VNPT-only:
# - Cần có `api-keys.json` trong /code (hoặc set API_KEYS_PATH trỏ tới file đó).
# - Cần có `kb_vnpt_embedding_index.pkl` trong /code (hoặc set VNPT_EMBED_INDEX_PATH).

python predict.py --input "/code/private_test.json" --output "submission.csv"

echo "[DONE] Output: submission.csv"
