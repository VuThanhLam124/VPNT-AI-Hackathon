"""
VNPT-only helpers:
- Load VNPT embedding index (prebuilt via Embedding API).
- Retrieve top-k documents by cosine similarity using query embedding from API.
- Lightweight domain/type filters.

Không sử dụng model/embedding ngoài API của BTC.
"""

from __future__ import annotations

import os
import pickle
import re
from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

VNPT_EMBED_INDEX_PATH = os.getenv("VNPT_EMBED_INDEX_PATH", "kb_vnpt_embedding_index.pkl")
VNPT_EMBED_SLEEP_S = float(os.getenv("VNPT_EMBED_SLEEP_S", "0.12"))


def load_vnpt_embedding_index(path: str = VNPT_EMBED_INDEX_PATH) -> Optional[dict]:
    """
    Load index được tạo bởi `build_index.py --backend vnpt`.

    Returns dict:
      - docs: list[dict]
      - matrix: np.ndarray shape [N, dim] (đã normalize)
    """
    if np is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        docs = data.get("docs", [])
        embs = data.get("embeddings", [])
        if not isinstance(docs, list) or not isinstance(embs, list) or len(docs) != len(embs):
            return None
        kept_docs = []
        kept_embs = []
        for d, e in zip(docs, embs):
            if e is None:
                continue
            kept_docs.append(d)
            kept_embs.append(np.asarray(e, dtype=np.float32))
        if not kept_docs:
            return None
        matrix = np.stack(kept_embs, axis=0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        return {
            "backend": "vnpt",
            "model_name": data.get("model_name", "vnptai_hackathon_embedding"),
            "docs": kept_docs,
            "matrix": matrix,
        }
    except Exception:
        return None


def retrieve_docs_vnpt_from_vector(vec: "np.ndarray", vnpt_index: dict, top_k: int = 5) -> list:
    """
    vec đã normalize (shape: [dim]).
    """
    sims = vnpt_index["matrix"] @ vec
    if top_k <= 0:
        return []
    top_k = min(int(top_k), len(sims))
    top_idx = np.argpartition(-sims, max(0, top_k - 1))[:top_k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    out = []
    for idx in top_idx:
        doc = dict(vnpt_index["docs"][int(idx)])
        doc["retrieval_score"] = float(sims[int(idx)])
        out.append(doc)
    return out


def retrieve_docs_vnpt(query: str, vnpt_index: dict, client, top_k: int = 5) -> list:
    """
    Retrieval bằng VNPT embedding index:
      - embed query bằng Embedding API BTC (1 request)
      - cosine search trong index local
    """
    if np is None or client is None or vnpt_index is None:
        return []
    vec = client.get_embedding(query, encoding_format="float")
    if vec is None:
        return []
    vec = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(vec))
    if n == 0:
        return []
    vec = vec / n
    if VNPT_EMBED_SLEEP_S and VNPT_EMBED_SLEEP_S > 0:
        import time

        time.sleep(VNPT_EMBED_SLEEP_S)
    return retrieve_docs_vnpt_from_vector(vec, vnpt_index, top_k=top_k)


def retrieve_docs_vnpt_filtered(query: str, vnpt_index: dict, client, allowed_doc_types: Optional[set], top_k: int = 5) -> list:
    raw = retrieve_docs_vnpt(query, vnpt_index, client, top_k=max(20, int(top_k) * 4))
    return _filter_docs_by_type(raw, allowed_doc_types)[: int(top_k)]


def _allowed_doc_types(domain: str) -> Optional[set]:
    """
    Giảm nhiễu retrieval bằng cách lọc theo doc_type.
    """
    if domain == "math":
        return {"stem", "qa", "kv"}
    if domain == "chemistry":
        return {"atomic", "stem", "qa", "kv"}
    if domain == "geography":
        return {"qa", "ward", "province", "unesco", "temple", "kv"}
    if domain == "tu_tuong_hcm":
        return {"hcm", "hcm_event", "qa", "kv"}
    if domain == "law":
        return {"qa", "kv"}
    if domain == "lich_su":
        return {"qa", "temple", "unesco", "kv", "hcm_event"}
    return {"qa", "unesco", "temple", "province", "kv"}


def _filter_docs_by_type(docs: list, allowed: Optional[set]) -> list:
    if allowed is None:
        return docs
    if not allowed:
        return []
    return [d for d in docs if d.get("doc_type") in allowed]


def _is_math_like(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if "$" in t or "\\" in t:
        return True
    if any(x in t for x in ["sin", "cos", "tan", "cot", "log", "đạo hàm", "tích phân", "phương trình", "xác suất"]):
        return True
    if any(sym in t for sym in ["=", "+", "-", "*", "/"]) and any(ch.isdigit() for ch in t):
        return True
    return False

