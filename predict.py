"""
Entry-point inference (VNPT-only):
- Retrieval: Embedding API BTC + cosine search trong `kb_vnpt_embedding_index.pkl`
- Answering: LLM API BTC (small/large)

Output: submission.csv (qid,answer)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional

from api_client import APIClient
from data_utils import detect_domain, extract_context_and_question, has_embedded_context, load_dataset
from generate_simple import _allowed_doc_types, _is_math_like, load_vnpt_embedding_index, retrieve_docs_vnpt_filtered


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/code/private_test.json")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument(
        "--api_keys",
        default=os.getenv("API_KEYS_PATH", "api-keys.json"),
        help="Path tới api-keys.json (VNPT LLM + Embedding).",
    )
    parser.add_argument(
        "--llm_model",
        choices=["auto", "small", "large"],
        default=os.getenv("VNPT_LLM_MODEL", "auto"),
        help="Chọn LLM API BTC: auto/small/large.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=int(os.getenv("VNPT_LLM_MAX_TOKENS", "64")),
        help="Giới hạn max_completion_tokens cho LLM API.",
    )
    parser.add_argument(
        "--vnpt_index",
        default=os.getenv("VNPT_EMBED_INDEX_PATH", "kb_vnpt_embedding_index.pkl"),
        help="VNPT embedding index pickle (build bằng build_index.py --backend vnpt).",
    )
    parser.add_argument(
        "--top_k_retrieval",
        type=int,
        default=int(os.getenv("TOP_K_RETRIEVAL", "6")),
        help="Số chunk lấy từ KB để làm context.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=os.getenv("ENABLE_THINKING", "0") == "1",
        help="Bật hướng dẫn 'suy luận thầm' (không yêu cầu giải thích).",
    )
    parser.add_argument(
        "--error_log",
        default=os.getenv("ERROR_LOG_PATH", "llm_errors.jsonl"),
        help="Đường dẫn log lỗi (jsonl).",
    )
    return parser.parse_args()


def _choice_label(i: int) -> str:
    return chr(65 + i)


def _parse_choice_letter(text: str, num_choices: int) -> Optional[str]:
    if not text:
        return None
    letters = {chr(ord("A") + i) for i in range(num_choices)}
    stripped = text.strip().upper()
    if stripped in letters:
        return stripped
    m = re.search(r"\b([A-J])\b", stripped)
    if m:
        cand = m.group(1)
        if cand in letters:
            return cand
    m = re.search(r"(?:^|[^A-Z])([A-J])(?:[^A-Z]|$)", stripped)
    if m:
        cand = m.group(1)
        if cand in letters:
            return cand
    return None


def _build_prompt(question: str, choices: List[str], context: str, enable_thinking: bool) -> str:
    lines = []
    lines.append("Bạn là hệ thống trả lời câu hỏi trắc nghiệm.")
    if enable_thinking:
        lines.append("Hãy suy luận thầm, KHÔNG viết ra suy luận.")
    lines.append("Chỉ trả về đúng 1 chữ cái (A, B, C, ...), không thêm giải thích.")
    if context:
        lines.append("")
        lines.append("<context>")
        lines.append(context)
        lines.append("</context>")
    lines.append("")
    lines.append("<question>")
    lines.append(question)
    lines.append("</question>")
    lines.append("")
    lines.append("<choices>")
    for i, c in enumerate(choices):
        lines.append(f"{_choice_label(i)}. {c}")
    lines.append("</choices>")
    lines.append("")
    lines.append("Trả lời:")
    return "\n".join(lines)


def _select_llm_use_large(mode: str, question: str) -> bool:
    if mode == "large":
        return True
    if mode == "small":
        return False
    # auto
    ql = (question or "").lower()
    if _is_math_like(ql):
        return True
    if len(ql) > 350:
        return True
    return True


def main():
    args = parse_args()

    data = load_dataset(args.input)
    print(f"Total: {len(data)} questions")

    vnpt_index = load_vnpt_embedding_index(args.vnpt_index)
    if vnpt_index is None:
        raise SystemExit(f"[ERROR] Không load được VNPT index: {args.vnpt_index}")

    client = APIClient(api_config_path=args.api_keys)

    errors: List[Dict] = []
    final_answers: Dict[str, str] = {}

    def record_error(item: Dict, stage: str, error: str, raw: str = ""):
        errors.append(
            {
                "qid": item.get("qid"),
                "stage": stage,
                "error": error,
                "question": item.get("question"),
                "choices": item.get("choices"),
                "context": item.get("context"),
                "raw": (raw or "")[:4000],
            }
        )

    for row in data:
        qid = row.get("qid")
        q_text = row.get("question", "") or ""
        choices = row.get("choices", []) or []

        if not isinstance(qid, str) or not isinstance(q_text, str) or not isinstance(choices, list) or not choices:
            record_error({"qid": qid, "question": q_text, "choices": choices, "context": ""}, "input", "invalid_format")
            # fallback
            final_answers[str(qid)] = "A"
            continue

        context, clean_q = extract_context_and_question(q_text)
        domain = detect_domain(clean_q)
        allowed = _allowed_doc_types(domain if not _is_math_like(clean_q) else "math")

        rag_context = ""
        if context and has_embedded_context(q_text):
            rag_context = context[:1200]
        else:
            docs = retrieve_docs_vnpt_filtered(
                clean_q,
                vnpt_index=vnpt_index,
                client=client,
                allowed_doc_types=allowed,
                top_k=max(1, int(args.top_k_retrieval)),
            )
            if docs:
                rag_context = "\n\n---\n\n".join((d.get("text") or "")[:800] for d in docs)

        prompt = _build_prompt(clean_q, [str(x) for x in choices], rag_context, enable_thinking=bool(args.enable_thinking))

        use_large = _select_llm_use_large(args.llm_model, clean_q)
        try:
            out = client.call_llm(prompt, use_large=use_large, max_tokens=int(args.max_tokens), temperature=0.0)
        except Exception as e:
            record_error(
                {"qid": qid, "question": clean_q, "choices": choices, "context": rag_context},
                "llm_call",
                f"{type(e).__name__}: {e}",
            )
            final_answers[qid] = "A"
            continue

        ans = _parse_choice_letter(out, num_choices=len(choices))
        if not ans:
            record_error(
                {"qid": qid, "question": clean_q, "choices": choices, "context": rag_context},
                "llm_parse",
                "cannot_parse_choice_letter",
                raw=out,
            )
            ans = "A"
        final_answers[qid] = ans

    if errors:
        with open(args.error_log, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[WARN] Wrote error log: {args.error_log} ({len(errors)} items)")

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer"])
        for row in data:
            writer.writerow([row.get("qid"), final_answers.get(row.get("qid"), "A")])
    print(f"[DONE] Saved {args.output}")


if __name__ == "__main__":
    main()

