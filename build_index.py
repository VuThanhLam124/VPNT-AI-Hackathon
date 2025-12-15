"""
Build VNPT embedding index cho KB bằng Embedding API BTC.

Output: `kb_vnpt_embedding_index.pkl`
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Optional

import numpy as np
from tqdm import tqdm

from api_client import APIClient
from data_utils import load_kb_corpus


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_dir", default="data/converted")
    parser.add_argument("--out", default="kb_vnpt_embedding_index.pkl")
    parser.add_argument("--encoding_format", default="float", choices=["float", "base64"])
    parser.add_argument(
        "--sleep_s",
        type=float,
        default=float(os.getenv("EMBED_SLEEP_S", "0.15")),
        help="Throttle giữa các request embedding (s). Quota BTC ~ 500 req/phút.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Nếu output đã tồn tại, tiếp tục embed phần còn thiếu (yêu cầu KB không đổi).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Giới hạn số chunk để test nhanh (0 = không giới hạn).",
    )
    parser.add_argument(
        "--max_chunk_chars",
        type=int,
        default=800,
        help="Truncate text mỗi chunk trước khi embed (giảm latency).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = APIClient(api_config_path=os.getenv("API_KEYS_PATH", "api-keys.json"))

    docs_raw = load_kb_corpus(args.kb_dir)
    docs = []
    texts = []
    for d in docs_raw:
        t = d.get("text", "")
        if not t or len(t) < 10:
            continue
        if args.max_chunk_chars and len(t) > args.max_chunk_chars:
            t = t[: args.max_chunk_chars]
        docs.append(
            {
                "text": t,
                "source_file": d.get("source_file"),
                "doc_type": d.get("doc_type"),
                "answer": d.get("answer"),
            }
        )
        texts.append(t)

    if args.limit and args.limit > 0:
        docs = docs[: args.limit]
        texts = texts[: args.limit]

    out_path = args.out
    start_idx = 0
    embeddings: list[Optional[np.ndarray]] = [None] * len(texts)
    if args.resume and os.path.exists(out_path):
        try:
            with open(out_path, "rb") as f:
                old = pickle.load(f)
            old_docs = old.get("docs", [])
            old_emb = old.get("embeddings", [])
            if isinstance(old_docs, list) and isinstance(old_emb, list) and len(old_docs) == len(old_emb) == len(docs):
                embeddings = old_emb
                for i, e in enumerate(embeddings):
                    if e is None:
                        start_idx = i
                        break
                else:
                    start_idx = len(embeddings)
            else:
                print("[WARN] Resume bị bỏ qua do KB khác với output hiện tại.")
        except Exception as e:
            print(f"[WARN] Không resume được: {e}")

    for i in tqdm(range(start_idx, len(texts))):
        vec = client.get_embedding(texts[i], encoding_format=args.encoding_format)
        embeddings[i] = vec
        if args.sleep_s and args.sleep_s > 0:
            time.sleep(args.sleep_s)
        if (i + 1) % 50 == 0 or (i + 1) == len(texts):
            with open(out_path, "wb") as f:
                pickle.dump(
                    {
                        "backend": "vnpt",
                        "model_name": "vnptai_hackathon_embedding",
                        "encoding_format": args.encoding_format,
                        "docs": docs,
                        "embeddings": embeddings,
                    },
                    f,
                )

    print(f"✓ Built VNPT embedding index with {len(docs)} docs -> {out_path}")


if __name__ == "__main__":
    main()

