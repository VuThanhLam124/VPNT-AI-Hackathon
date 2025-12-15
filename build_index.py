"""
Build embedding index cho KB.
Output mặc định: kb_bge_index.pkl (BGE-m3 offline)

Có thể build bằng Embedding API của BTC (VNPT) và lưu lại để reuse.
"""
import argparse
import os
import pickle
import time
from typing import Optional

import numpy as np
from tqdm import tqdm

from api_client import APIClient
from data_utils import load_kb_corpus
from generate_simple import build_kb_embedding_index, BGE_INDEX_PATH, MAX_CHUNK_CHARS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_dir", default="data/converted")
    parser.add_argument("--max_chunk_chars", type=int, default=MAX_CHUNK_CHARS)
    parser.add_argument(
        "--backend",
        choices=["bge", "vnpt"],
        default="bge",
        help="bge: embed offline bằng BAAI/bge-m3; vnpt: embed bằng Embedding API BTC.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Đường dẫn file index output (pickle). Nếu bỏ trống sẽ dùng mặc định theo backend.",
    )
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
        help="Nếu output đã tồn tại, tiếp tục embed phần còn thiếu.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Giới hạn số chunk để test nhanh (0 = không giới hạn).",
    )
    return parser.parse_args()


def build_vnpt_embedding_index(
    kb_dir: str,
    out_path: str,
    max_chunk_chars: int,
    encoding_format: str,
    sleep_s: float,
    resume: bool,
    limit: int,
):
    client = APIClient()
    docs_raw = load_kb_corpus(kb_dir)

    docs = []
    texts = []
    for d in docs_raw:
        t = d.get("text", "")
        if not t or len(t) < 10:
            continue
        if max_chunk_chars and len(t) > max_chunk_chars:
            t = t[:max_chunk_chars]
        docs.append(
            {
                "text": t,
                "source_file": d.get("source_file"),
                "doc_type": d.get("doc_type"),
                "answer": d.get("answer"),
            }
        )
        texts.append(t)

    if limit and limit > 0:
        docs = docs[:limit]
        texts = texts[:limit]

    start_idx = 0
    embeddings: list[Optional[np.ndarray]] = [None] * len(texts)
    if resume and os.path.exists(out_path):
        try:
            with open(out_path, "rb") as f:
                old = pickle.load(f)
            old_docs = old.get("docs", [])
            old_emb = old.get("embeddings", [])
            if isinstance(old_docs, list) and isinstance(old_emb, list) and len(old_docs) == len(old_emb):
                if len(old_docs) == len(docs):
                    embeddings = old_emb
                    for i, e in enumerate(embeddings):
                        if e is None:
                            start_idx = i
                            break
                    else:
                        start_idx = len(embeddings)
                else:
                    print("[WARN] Resume bị bỏ qua do số lượng docs khác nhau.")
        except Exception as e:
            print(f"[WARN] Không resume được: {e}")

    for i in tqdm(range(start_idx, len(texts))):
        vec = client.get_embedding(texts[i], encoding_format=encoding_format)
        embeddings[i] = vec
        if sleep_s and sleep_s > 0:
            time.sleep(sleep_s)

        # checkpoint mỗi 50 items
        if (i + 1) % 50 == 0 or (i + 1) == len(texts):
            with open(out_path, "wb") as f:
                pickle.dump(
                    {
                        "backend": "vnpt",
                        "model_name": "vnptai_hackathon_embedding",
                        "encoding_format": encoding_format,
                        "docs": docs,
                        "embeddings": embeddings,
                    },
                    f,
                )

    print(f"✓ Built VNPT embedding index with {len(docs)} docs -> {out_path}")


def main():
    args = parse_args()

    if args.backend == "bge":
        out_path = args.out or BGE_INDEX_PATH
        print("Building KB embedding index (BGE-m3)...")
        kb_emb = build_kb_embedding_index(
            kb_dir=args.kb_dir,
            max_chunk_chars=args.max_chunk_chars,
            save_path=out_path,
        )
        if kb_emb is None:
            print("✗ Không build được index (thiếu dependencies hoặc model).")
        else:
            print(f"✓ Built index with {len(kb_emb['docs'])} docs -> {out_path}")
        return

    out_path = args.out or "kb_vnpt_embedding_index.pkl"
    print("Building KB embedding index (VNPT Embedding API)...")
    build_vnpt_embedding_index(
        kb_dir=args.kb_dir,
        out_path=out_path,
        max_chunk_chars=args.max_chunk_chars,
        encoding_format=args.encoding_format,
        sleep_s=args.sleep_s,
        resume=args.resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
