"""
Entry-point inference: đọc input và xuất submission theo chuẩn BTC.

Mode: Local LLM (không dùng VNPT LLM API).
Model mặc định: Qwen/Qwen3-8B-Base (Transformers).

Lưu ý:
- Model Base không luôn tuân thủ format; script sẽ parse chữ cái A/B/C/D... từ output.
- Nếu lỗi/không parse được, sẽ log theo qid ra `llm_errors.jsonl` và exit(1).
"""

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional

from data_utils import extract_context_and_question, has_embedded_context, load_dataset, detect_domain
from generate_simple import (
    BGE_INDEX_PATH,
    build_bm25_index,
    load_saved_bge_index,
    retrieve_docs_bge_filtered,
    retrieve_docs_bm25_filtered,
    _allowed_doc_types,
    _is_math_like,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/code/private_test.json",
        help="Đường dẫn file test (BTC mount).",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Đường dẫn file output (qid,answer).",
    )
    parser.add_argument("--kb_dir", default="data/converted", help="Thư mục KB.")
    parser.add_argument(
        "--model_id",
        default=os.getenv("LOCAL_LLM_MODEL_ID", "Qwen/Qwen3-8B-Base"),
        help="HF model id hoặc local folder path. Có thể set env LOCAL_LLM_MODEL_ID.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "16")),
        help="Giới hạn token sinh thêm cho mỗi câu.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=os.getenv("LOCAL_LLM_4BIT", "1") == "1",
        help="Bật quantization 4-bit (bitsandbytes) nếu có CUDA.",
    )
    parser.add_argument(
        "--top_k_retrieval",
        type=int,
        default=int(os.getenv("TOP_K_RETRIEVAL", "4")),
        help="Số chunk lấy từ KB để làm context.",
    )
    parser.add_argument(
        "--bge_index",
        default=os.getenv("BGE_INDEX_PATH", BGE_INDEX_PATH),
        help="Đường dẫn BGE index (pickle).",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=os.getenv("ENABLE_THINKING", "1") == "1",
        help="Thêm hướng dẫn suy luận ngắn gọn trước khi trả lời.",
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
        lines.append("Suy luận ngắn gọn trước khi chốt đáp án.")
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


class LocalLLM:
    def __init__(self, model_id: str, load_in_4bit: bool):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.hf_token = os.getenv("HF_TOKEN")

        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = None
            torch_dtype = torch.float32

        quant_cfg = None
        if load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                try:
                    import bitsandbytes  # noqa: F401
                    has_bnb = True
                except Exception:
                    has_bnb = False

                if has_bnb:
                    quant_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
            except Exception:
                quant_cfg = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=self.hf_token,
            trust_remote_code=True,
        )
        model_kwargs = {
            "token": self.hf_token,
            "trust_remote_code": True,
            "device_map": device_map,
            "quantization_config": quant_cfg,
        }
        # Transformers >= 4.57 dùng `dtype` (torch_dtype deprecated).
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, **model_kwargs)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, **model_kwargs)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        torch = self.torch
        tok = self.tokenizer
        model = self.model

        # Qwen family thường hỗ trợ chat template; Base vẫn có thể dùng để ổn định định dạng prompt.
        if hasattr(tok, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(rendered, return_tensors="pt")
        else:
            inputs = tok(prompt, return_tensors="pt")

        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        return tok.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()

    data = load_dataset(args.input)
    print(f"Total: {len(data)} questions")

    bge_index = load_saved_bge_index(args.bge_index)
    bm25_index = None
    if bge_index is None:
        print(f"[WARN] Không load được BGE index ({args.bge_index}), fallback BM25.")
        bm25_index = build_bm25_index(args.kb_dir)
        if bm25_index is None:
            raise SystemExit(
                "[ERROR] Không có index retrieval khả dụng (BGE hoặc BM25). "
                "Kiểm tra lại `--bge_index` và `--kb_dir`, hoặc cài `rank-bm25`."
            )
    else:
        print(f"[INFO] Loaded BGE index: {args.bge_index} ({len(bge_index['docs'])} docs)")

    llm = LocalLLM(args.model_id, load_in_4bit=bool(args.load_in_4bit))
    print(f"[INFO] Local model: {args.model_id}")

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
            continue

        context, clean_q = extract_context_and_question(q_text)
        domain = detect_domain(clean_q)
        allowed = _allowed_doc_types(domain if not _is_math_like(clean_q) else "math")

        rag_context = ""
        if context and has_embedded_context(q_text):
            rag_context = context[:1200]
        else:
            docs = []
            if bge_index is not None:
                docs = retrieve_docs_bge_filtered(
                    clean_q,
                    kb_emb=bge_index,
                    allowed_doc_types=allowed,
                    top_k=max(1, int(args.top_k_retrieval)),
                )
            elif bm25_index is not None:
                docs = retrieve_docs_bm25_filtered(
                    clean_q,
                    bm25_index=bm25_index,
                    allowed_doc_types=allowed,
                    top_k=max(1, int(args.top_k_retrieval)),
                )
            if docs:
                rag_context = "\n\n---\n\n".join((d.get("text") or "")[:800] for d in docs)

        prompt = _build_prompt(clean_q, [str(x) for x in choices], rag_context, enable_thinking=bool(args.enable_thinking))
        try:
            out = llm.generate(prompt, max_new_tokens=int(args.max_new_tokens))
        except Exception as e:
            record_error(
                {"qid": qid, "question": clean_q, "choices": choices, "context": rag_context},
                "llm_call",
                f"{type(e).__name__}: {e}",
            )
            continue

        ans = _parse_choice_letter(out, num_choices=len(choices))
        if not ans:
            record_error(
                {"qid": qid, "question": clean_q, "choices": choices, "context": rag_context},
                "llm_parse",
                "cannot_parse_choice_letter",
                raw=out,
            )
            continue
        final_answers[qid] = ans

    if errors:
        with open(args.error_log, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[ERROR] Wrote error log: {args.error_log} ({len(errors)} items)")
        raise SystemExit(1)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer"])
        for row in data:
            writer.writerow([row["qid"], final_answers[row["qid"]]])
    print(f"[DONE] Saved {args.output}")


if __name__ == "__main__":
    main()
