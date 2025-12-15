# VNPT AI Hackathon - Track 2: The Builder

Repository này cung cấp pipeline end-to-end để trả lời câu hỏi trắc nghiệm và sinh `submission.csv` theo chuẩn BTC, chỉ sử dụng API do BTC cung cấp (VNPT LLM + VNPT Embedding).

## Pipeline Flow

1. Đọc input từ `/code/private_test.json` (BTC mount vào container khi chấm).
2. Tiền xử lý: tách context nhúng (nếu có), lọc câu nhạy cảm.
3. Retrieval: dùng Embedding API BTC để embed query và cosine-search trong `kb_vnpt_embedding_index.pkl`.
4. Answering: dùng LLM API BTC (Small/Large) để trả về đúng 1 chữ cái đáp án.
5. Xuất `submission.csv` (cột `qid,answer`) ra thư mục làm việc.

## Data Processing

- Nguồn KB: `data/converted/*.jsonl`.
- Loader trong `data_utils.py` hỗ trợ JSONL chuẩn, dòng có dấu phẩy cuối, và pseudo-array/multi-line; đảm bảo mỗi `{}` là 1 chunk.
- Dữ liệu mới như `stem.jsonl`, `atomic_weights.jsonl`, `temple.jsonl` được nạp thành các `doc_type` tương ứng để retrieval lọc theo domain.

## Resource Initialization

Nếu bạn thay đổi KB, hãy build lại VNPT embedding index:

```bash
python build_index.py --kb_dir data/converted --out kb_vnpt_embedding_index.pkl --resume --sleep_s 0.12
```

## Chạy local (không dùng Docker)

```bash
python predict.py \
  --input AInicorns_TheBuilder_public_v1.1/data/test.json \
  --output submission.csv \
  --api_keys api-keys.json \
  --vnpt_index kb_vnpt_embedding_index.pkl \
  --llm_model large \
  --top_k_retrieval 6
```

## Docker (theo hướng dẫn nộp bài)

- Entry-point của pipeline: `predict.py`
- Script chạy end-to-end: `inference.sh` (đọc `/code/private_test.json`, xuất `submission.csv`)

Build:
```bash
docker build -t team_submission .
```

Run (ví dụ local):
```bash
docker run --rm -v /path/to/private_test.json:/code/private_test.json team_submission
```

## Notebook thử nghiệm Kaggle

- File `kaggle_eval.ipynb` (trong repo) chứa quy trình chạy trên kernel Kaggle: cài dependency, mount `api-keys.json`, chuyển `val.csv/test.csv` sang JSON format và gọi `predict.py` để xuất `submission.csv`.
