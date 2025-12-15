"""
Entry-point inference: đọc private_test.json và xuất submission.csv
"""
import argparse
import os
import json
import time
import re
from typing import Dict, List, Optional, Tuple

from api_client import APIClient
from data_utils import (
    load_dataset,
    extract_context_and_question,
    has_embedded_context,
    is_sensitive_question,
    detect_domain,
)
from generate_simple import (
    build_bm25_index,
    retrieve_docs_bm25_filtered,
    _allowed_doc_types,
    _is_math_like,
    try_solve_stem,
    load_vnpt_embedding_index,
    retrieve_docs_vnpt_filtered,
    retrieve_docs_hybrid_filtered,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/code/private_test.json",
        help="Đường dẫn file test (mặc định: /code/private_test.json do BTC mount)",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Đường dẫn file output submission.csv",
    )
    parser.add_argument(
        "--kb_dir",
        default="data/converted",
        help="Thư mục KB",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=15,
        help="Số câu hỏi cho mỗi request LLM (để tránh rate-limit).",
    )
    parser.add_argument(
        "--llm_model",
        choices=["auto", "large", "small"],
        default="auto",
        help="Chọn LLM VNPT để trả lời batch (auto = trộn small/large theo độ khó).",
    )
    parser.add_argument(
        "--llm_n",
        type=int,
        default=1,
        help="Self-consistency: số completions trong 1 request (khuyến nghị 1-3; chỉ áp dụng cho LLM Large).",
    )
    parser.add_argument(
        "--llm_seed",
        type=int,
        default=42,
        help="Seed cho reproducible outputs (nếu API hỗ trợ).",
    )
    parser.add_argument(
        "--large_top_p",
        type=float,
        default=1.0,
        help="top_p cho LLM Large.",
    )
    parser.add_argument(
        "--large_top_k",
        type=int,
        default=20,
        help="top_k cho LLM Large.",
    )
    parser.add_argument(
        "--small_top_p",
        type=float,
        default=1.0,
        help="top_p cho LLM Small.",
    )
    parser.add_argument(
        "--small_top_k",
        type=int,
        default=20,
        help="top_k cho LLM Small.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="presence_penalty (-2..2).",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="frequency_penalty (-2..2).",
    )
    parser.add_argument(
        "--use_logprobs",
        action="store_true",
        help="Bật logprobs/top_logprobs (nếu server hỗ trợ).",
    )
    parser.add_argument(
        "--top_logprobs",
        type=int,
        default=5,
        help="Số token top_logprobs (0-20).",
    )
    parser.add_argument(
        "--use_tools",
        action="store_true",
        help="Bật function calling (tools/tool_choice) cho LLM Large.",
    )
    parser.add_argument(
        "--use_worker_small",
        action="store_true",
        help="Bật LLM Small làm worker (rewrite/intent) để cải thiện retrieval.",
    )
    parser.add_argument(
        "--no_worker_small",
        action="store_true",
        help="Tắt LLM Small làm worker (hữu ích để tái tạo mode cũ). Nếu đặt cờ này thì worker luôn tắt.",
    )
    parser.add_argument(
        "--worker_batch_size",
        type=int,
        default=30,
        help="Số câu cho mỗi request worker (LLM Small).",
    )
    return parser.parse_args()

def _choice_label(i: int) -> str:
    return chr(65 + i)


def _parse_json_answers(text: str) -> Dict[str, str]:
    """
    Kỳ vọng: {"answers":[{"qid":"...","answer":"A"}, ...]}
    """
    out: Dict[str, str] = {}
    try:
        obj = json.loads(text)
    except Exception:
        return out
    answers = obj.get("answers")
    if not isinstance(answers, list):
        return out
    for item in answers:
        if not isinstance(item, dict):
            continue
        qid = item.get("qid")
        ans = item.get("answer")
        if isinstance(qid, str) and isinstance(ans, str) and ans:
            out[qid] = ans.strip().upper()[:1]
    return out


def _compress_context(question: str, choices: List[str], context: str, max_chars: int = 1200) -> str:
    """
    Rút gọn context nhúng dài: chọn các câu có overlap token cao với câu hỏi + lựa chọn.
    """
    if not context or len(context) <= max_chars:
        return context
    tok_re = re.compile(r"[^\W_]+", re.UNICODE)

    def toks(s: str) -> set:
        return set(t.lower() for t in tok_re.findall(s or "") if len(t) > 1)

    query_tokens = toks(question)
    for c in choices:
        query_tokens |= toks(c or "")

    sents = []
    for line in context.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        sents.extend(parts)

    scored = []
    for s in sents:
        st = toks(s)
        if not st:
            continue
        overlap = len(st & query_tokens)
        if overlap == 0:
            continue
        scored.append((overlap, s))
    scored.sort(key=lambda x: x[0], reverse=True)

    kept = []
    total = 0
    for _, s in scored:
        if total + len(s) + 1 > max_chars and kept:
            break
        kept.append(s)
        total += len(s) + 1
    return "\n".join(kept) if kept else context[:max_chars]


def _build_evidence_pack(question: str, choices: List[str], context: str, max_chars: int = 900) -> str:
    """
    Nén context theo evidence: chọn các câu liên quan nhất tới question + choices.
    Output dạng danh sách dòng "E1: ...", giúp LLM bám evidence, giảm nhiễu.
    """
    if not context:
        return ""
    if len(context) <= max_chars:
        # vẫn gắn nhãn để prompt dễ chỉ evidence
        return "\n".join([f"E1: {context.strip()}"])[:max_chars]

    tok_re = re.compile(r"[^\W_]+", re.UNICODE)

    def toks(s: str) -> List[str]:
        return [t.lower() for t in tok_re.findall(s or "") if len(t) > 1]

    stop = {
        "và","là","của","cho","trong","một","những","các","theo","với","được","từ","đến",
        "nào","sau","đây","đó","này","khi","vì","do","ở","trên","dưới","không","có",
        "được","bị","lại","vẫn","cũng","đã","đang","sẽ",
    }

    q_tokens = [t for t in toks(question) if t not in stop]
    c_tokens = []
    for c in choices:
        c_tokens.append([t for t in toks(c) if t not in stop])

    # sentence split
    sents: List[str] = []
    for line in context.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p = p.strip()
            if p:
                sents.append(p)

    if not sents:
        return f"E1: {context.strip()}"[:max_chars]

    ctx_norm = _normalize_text(context)
    scored = []
    for idx, s in enumerate(sents):
        st = set(toks(s))
        q_ov = sum(1 for t in set(q_tokens) if t in st)
        best_c = 0
        exact_hits = 0
        for j, ct in enumerate(c_tokens):
            ov = sum(1 for t in set(ct) if t in st)
            best_c = max(best_c, ov)
            # exact short substring for choice
            c_raw = choices[j] if j < len(choices) else ""
            c_norm = _normalize_text(c_raw)
            if 8 <= len(c_norm) <= 80 and c_norm and c_norm in _normalize_text(s):
                exact_hits += 1
        score = 3 * q_ov + 2 * best_c + 8 * exact_hits
        # numbers: keep sentences with numbers if question mentions %
        if "%" in (question or "") and re.search(r"\d", s):
            score += 2
        if score > 0:
            scored.append((score, idx, s))

    if not scored:
        return f"E1: {context.strip()}"[:max_chars]

    scored.sort(reverse=True, key=lambda x: x[0])

    kept: List[str] = []
    used_idx = set()
    total = 0

    # Ensure at least one sentence mentioning each choice keyword if present
    for j, c in enumerate(choices):
        c_norm = _normalize_text(c)
        if not (8 <= len(c_norm) <= 80):
            continue
        if c_norm and c_norm in ctx_norm:
            # pick the first sentence containing it
            for idx, s in enumerate(sents):
                if idx in used_idx:
                    continue
                if c_norm in _normalize_text(s):
                    used_idx.add(idx)
                    kept.append(s)
                    break

    for _, idx, s in scored:
        if idx in used_idx:
            continue
        line = s
        add_len = len(line) + 6
        if total + add_len > max_chars and kept:
            break
        kept.append(line)
        used_idx.add(idx)
        total += add_len

    # Build labeled evidence
    out_lines = []
    for i, s in enumerate(kept[:20], 1):
        out_lines.append(f"E{i}: {s}")
    joined = "\n".join(out_lines)
    return joined[:max_chars]


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    # remove most punctuation but keep % and numbers/letters
    s = re.sub(r"[^\w\s%À-ỹ]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_worker_items(text: str) -> Dict[str, Dict[str, str]]:
    """
    Kỳ vọng: {"items":[{"qid":"...","rewrite":"...","intent":"..."}]}
    """
    out: Dict[str, Dict[str, str]] = {}
    try:
        obj = json.loads(text)
    except Exception:
        return out
    items = obj.get("items")
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        qid = it.get("qid")
        rewrite = it.get("rewrite")
        intent = it.get("intent")
        if isinstance(qid, str):
            out[qid] = {
                "rewrite": rewrite.strip() if isinstance(rewrite, str) else "",
                "intent": intent.strip().lower() if isinstance(intent, str) else "",
            }
    return out


def run_inference_llm_batch(
    input_path: str,
    output_path: str,
    kb_dir: str,
    batch_size: int,
    llm_mode: str,
    use_worker_small: bool = False,
    worker_batch_size: int = 30,
    llm_n: int = 1,
    llm_seed: int = 42,
    large_top_p: float = 1.0,
    large_top_k: int = 20,
    small_top_p: float = 1.0,
    small_top_k: int = 20,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    use_logprobs: bool = False,
    top_logprobs: int = 5,
    use_tools: bool = False,
):
    data = load_dataset(input_path)
    print(f"Total: {len(data)} questions")

    bm25_index = build_bm25_index(kb_dir)
    if bm25_index is None:
        print("[ERROR] Không build được BM25 index. Hãy cài `rank-bm25` và đảm bảo KB hợp lệ.")
        raise SystemExit(2)

    client = APIClient()
    vnpt_index = load_vnpt_embedding_index()
    if vnpt_index is not None:
        print(f"[INFO] Using VNPT embedding index: {len(vnpt_index['docs'])} docs")

    errors: List[Dict] = []

    def _record_error(item: Dict, stage: str, error: str, raw: str = ""):
        errors.append(
            {
                "qid": item.get("qid"),
                "stage": stage,
                "error": error,
                "question": item.get("question"),
                "choices": item.get("choices"),
                "context": item.get("context"),
                "domain": item.get("domain"),
                "raw": (raw or "")[:4000],
            }
        )

    def _write_errors(path: str):
        if not errors:
            return
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[ERROR] Wrote error log: {path} ({len(errors)} items)")

    # Worker (LLM Small): rewrite query + intent to help retrieval (batch to respect quota)
    worker_map: Dict[str, Dict[str, str]] = {}
    if use_worker_small:
        worker_inputs: List[Dict] = []
        for row in data:
            qid = row["qid"]
            q_text = row["question"]
            choices = row["choices"]
            context, clean_q = extract_context_and_question(q_text)
            if context and has_embedded_context(q_text):
                continue
            if try_solve_stem(clean_q, choices):
                continue
            if is_sensitive_question(clean_q):
                continue
            worker_inputs.append({"qid": qid, "question": clean_q})

        def _flush_worker(batch: List[Dict], depth: int = 0):
            if not batch:
                return
            prompt = (
                "Bạn là worker model để hỗ trợ retrieval.\n"
                "Với mỗi câu hỏi, hãy:\n"
                "1) Viết lại query rất ngắn (tối đa 12 từ), chỉ giữ thực thể + keyword.\n"
                "2) Gán intent một trong: math, chemistry, law, tu_tuong_hcm, geography, reading, other.\n\n"
                "Output BẮT BUỘC là JSON:\n"
                "{\"items\":[{\"qid\":\"...\",\"rewrite\":\"...\",\"intent\":\"...\"}]}\n"
                "Không thêm chữ nào ngoài JSON.\n\n"
                "Dữ liệu:\n"
            )
            for it in batch:
                prompt += f"\n[qid={it['qid']}]\n{it['question']}\n"
            resp = client.call_chat(
                messages=[{"role": "system", "content": "Chỉ trả về JSON, không thêm chữ."}, {"role": "user", "content": prompt}],
                use_large=False,
                max_tokens=2048,
                temperature=0.0,
                top_p=small_top_p,
                top_k=small_top_k,
                n=1,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                response_format={"type": "json_object"},
                seed=int(llm_seed),
                logprobs=use_logprobs,
                top_logprobs=top_logprobs,
                timeout_s=60,
            )
            if isinstance(resp, dict) and isinstance(resp.get("error"), dict):
                err = resp.get("error") or {}
                resp = f"__VNPT_ERROR__{err.get('code','200')}__{err.get('type','')}"
            elif isinstance(resp, dict):
                # map to content string for parser
                choices = resp.get("choices")
                if isinstance(choices, list) and choices:
                    msg = (choices[0] or {}).get("message", {})
                    resp = (msg.get("content") or "") if isinstance(msg, dict) else ""
                else:
                    resp = ""
            if isinstance(resp, str) and resp.startswith("__VNPT_ERROR__400") and len(batch) > 1:
                # split nhẹ để tránh 1 câu bị block làm hỏng cả worker batch
                if depth < 1 and len(batch) > 2:
                    mid = len(batch) // 2
                    _flush_worker(batch[:mid], depth=depth + 1)
                    _flush_worker(batch[mid:], depth=depth + 1)
                    return
                return
            if isinstance(resp, str) and resp.startswith("__VNPT_ERROR__401"):
                time.sleep(65)
                return
            parsed = _parse_worker_items(resp)
            worker_map.update(parsed)

        # batch worker inputs
        for i in range(0, len(worker_inputs), max(1, worker_batch_size)):
            _flush_worker(worker_inputs[i : i + max(1, worker_batch_size)])
            time.sleep(0.2)

    pending_large: List[Dict] = []
    pending_small: List[Dict] = []
    final_answers: Dict[str, str] = {}
    stats = {
        "solved_llm": 0,
        "llm_errors": 0,
    }

    def _build_tools():
        # Tools do user định nghĩa; VNPT không có fixed list.
        return [
            {
                "type": "function",
                "function": {
                    "name": "kb_search_batch",
                    "description": "Tìm evidence từ KB cho nhiều qid một lần. Trả về items với evidence ngắn.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "qid": {"type": "string"},
                                        "query": {"type": "string"},
                                        "domain": {"type": "string"},
                                        "top_k": {"type": "integer"},
                                    },
                                    "required": ["qid", "query"],
                                },
                            }
                        },
                        "required": ["items"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_evidence",
                    "description": "Rút gọn context thành evidence pack E1..En dựa trên question+choices.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "choices": {"type": "array", "items": {"type": "string"}},
                            "context": {"type": "string"},
                            "max_chars": {"type": "integer"},
                        },
                        "required": ["question", "choices", "context"],
                    },
                },
            },
        ]

    def _run_tool(name: str, args: Dict[str, object]) -> Dict[str, object]:
        if name == "extract_evidence":
            q = str(args.get("question") or "")
            ch = args.get("choices") or []
            if not isinstance(ch, list):
                ch = []
            ctx = str(args.get("context") or "")
            mc = int(args.get("max_chars") or 900)
            return {"evidence": _build_evidence_pack(q, [str(x) for x in ch], ctx, max_chars=mc)}
        if name == "kb_search_batch":
            items = args.get("items") or []
            if not isinstance(items, list):
                items = []
            out_items = []
            for it in items[:50]:
                if not isinstance(it, dict):
                    continue
                qid = str(it.get("qid") or "")
                query = str(it.get("query") or "")
                dom = str(it.get("domain") or "general")
                topk = int(it.get("top_k") or 4)
                allowed = _allowed_doc_types(dom if not _is_math_like(query) else "math")
                docs = retrieve_docs_hybrid_filtered(
                    query,
                    vnpt_index=vnpt_index,
                    client=client,
                    bm25_index=bm25_index,
                    allowed_doc_types=allowed,
                    top_k=max(1, min(6, topk)),
                )
                joined = "\n\n---\n\n".join((d.get("text") or "")[:800] for d in (docs or []))
                evidence = _build_evidence_pack(query, [], joined, max_chars=700) if joined else ""
                out_items.append({"qid": qid, "evidence": evidence})
            return {"items": out_items}
        return {"error": f"Unknown tool: {name}"}

    def _call_large_with_tools(prompt_text: str) -> str:
        tools = _build_tools()
        messages = [
            {
                "role": "system",
                "content": "Bạn được phép gọi tools để tìm evidence. Sau khi đủ thông tin, trả về JSON answers.",
            },
            {"role": "user", "content": prompt_text},
        ]
        for _ in range(3):
            resp = client.call_chat(
                messages=messages,
                use_large=True,
                max_tokens=1536,
                temperature=0.0,
                top_p=large_top_p,
                top_k=large_top_k,
                n=1,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                # response_format có thể không tương thích với tool_calls trên một số gateway -> để None
                response_format=None,
                tools=tools,
                tool_choice="auto",
                timeout_s=120,
            )
            if isinstance(resp, dict) and isinstance(resp.get("error"), dict):
                err = resp.get("error") or {}
                return f"__VNPT_ERROR__{err.get('code','200')}__{err.get('type','')}"
            choices = resp.get("choices") if isinstance(resp, dict) else None
            if not isinstance(choices, list) or not choices:
                return ""
            ch0 = choices[0] if isinstance(choices[0], dict) else {}
            msg = ch0.get("message", {}) if isinstance(ch0, dict) else {}
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
            if tool_calls:
                # Append the assistant tool-call message before tool results (spec-compatible)
                messages.append({"role": "assistant", "tool_calls": tool_calls, "content": None})
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_id = tc.get("id")
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        parsed_args = {}
                    result = _run_tool(str(name), parsed_args if isinstance(parsed_args, dict) else {})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                continue
            # normal content
            content = (msg.get("content") or "") if isinstance(msg, dict) else ""
            return str(content)
        return ""

    def should_use_large(item: Dict) -> bool:
        """
        Heuristic route: ưu tiên Large cho bài toán, dài, nhiều lựa chọn,
        hoặc các domain dễ sai (HCM, luật, hóa, văn học).
        """
        q = item["question"]
        domain = item.get("domain")
        if _is_math_like(q):
            return True
        if domain in {"tu_tuong_hcm", "law", "chemistry"}:
            return True
        if len(q) > 260:
            return True
        ctx = item.get("context") or ""
        if len(ctx) > 900:
            return True
        if len(item["choices"]) > 5:
            return True
        # văn học / nghệ thuật
        if "nghệ thuật" in q.lower() or "biện pháp" in q.lower():
            return True
        return False

    def _flush_batch(pending: List[Dict], use_large_for_batch: bool, depth: int = 0):
        if not pending:
            return

        prompt = (
            "Bạn là hệ thống trả lời trắc nghiệm.\n"
            "Mục tiêu: chọn 1 đáp án đúng nhất cho MỖI câu.\n\n"
            "Nguồn thông tin:\n"
            "- Nếu có <context>: ưu tiên dùng đúng thông tin trong context.\n"
            "- Nếu không có/không đủ context: dùng kiến thức chung + suy luận + tính toán.\n\n"
            "Quy tắc chọn đáp án:\n"
            "- Nếu câu hỏi có 'Theo nội dung/Đoạn thông tin/Ngữ cảnh': chỉ kết luận từ context.\n"
            "  Nếu context không chứa thông tin liên quan và có lựa chọn 'Không đủ thông tin/Không thể kết luận': chọn lựa chọn đó.\n"
            "- Nếu có lựa chọn kiểu 'Cả A,B,C' hoặc 'Cả a, b, c': chọn nếu (và chỉ nếu) nhiều lựa chọn thành phần đều đúng.\n"
            "- Với %/so sánh số: trích số từ lựa chọn và chọn max/min theo câu hỏi.\n"
            "- Với toán/lý: tự tính ra kết quả rồi map sang lựa chọn khớp nhất.\n"
            "- Nếu context dài/nhiều đoạn: tập trung vào câu chứa đúng thực thể/keyword của câu hỏi.\n\n"
            "Yêu cầu output BẮT BUỘC (không thêm chữ nào khác):\n"
            "{\"answers\":[{\"qid\":\"...\",\"answer\":\"A\",\"evidence\":\"...\"}]}\n"
            "- answer là 1 chữ cái hợp lệ (A, B, C, ...).\n"
            "- evidence (tuỳ chọn) là 1 câu trích ngắn từ context để chứng minh.\n"
            "- Trả về đủ số lượng qid đã đưa vào.\n\n"
            "Dữ liệu:\n"
        )
        for item in pending:
            prompt += f"\n[qid={item['qid']}]\n"
            if item.get("context"):
                prompt += f"<evidence>\n{item['context']}\n</evidence>\n"
            prompt += f"<question>\n{item['question']}\n</question>\n"
            prompt += "<choices>\n"
            for i, c in enumerate(item["choices"]):
                prompt += f"{_choice_label(i)}. {c}\n"
            prompt += "</choices>\n"

        timeout_s = 120 if use_large_for_batch else 60
        batch_map: Dict[str, str] = {}
        resp = ""
        # Strict: chỉ gọi 1 lần, không retry/fallback; lỗi thì log.
        try:
            if use_large_for_batch and llm_n and llm_n > 1:
                resps = client.call_llm_n(
                    prompt,
                    use_large=True,
                    n=int(llm_n),
                    max_tokens=2048,
                    temperature=0.0,
                    top_p=large_top_p,
                    top_k=large_top_k,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    response_format={"type": "json_object"},
                    seed=int(llm_seed),
                    logprobs=use_logprobs,
                    top_logprobs=top_logprobs,
                    timeout_s=timeout_s,
                )
                if resps and isinstance(resps[0], str) and resps[0].startswith("__VNPT_ERROR__"):
                    resp = resps[0]
                    batch_map = {}
                else:
                    maps = [_parse_json_answers(r) for r in (resps or [])]
                    voted: Dict[str, str] = {}
                    for item in pending:
                        qid = item["qid"]
                        votes = []
                        for m in maps:
                            a = m.get(qid)
                            if a:
                                votes.append(a)
                        if votes:
                            from collections import Counter
                            c = Counter(votes)
                            voted[qid] = c.most_common(1)[0][0]
                    batch_map = voted
                    resp = "\n\n".join([r for r in (resps or []) if isinstance(r, str)])[:4000]
            else:
                if use_large_for_batch and use_tools:
                    resp = _call_large_with_tools(prompt)
                else:
                    resp = client.call_llm(
                        prompt,
                        use_large=use_large_for_batch,
                        max_tokens=2048,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                        timeout_s=timeout_s,
                    )
                batch_map = _parse_json_answers(resp)
        except Exception as e:
            stats["llm_errors"] += 1
            for item in pending:
                _record_error(item, "llm_call_exception", f"{type(e).__name__}: {e}")
            pending.clear()
            return

        # Nếu bị block/400 cho cả batch: split để cô lập câu gây lỗi
        if isinstance(resp, str) and (resp.startswith("__VNPT_ERROR__400") or resp.startswith("__VNPT_ERROR__200__BadRequestError")) and len(pending) > 1:
            stats["llm_errors"] += 1
            if depth < 1 and len(pending) > 2:
                mid = len(pending) // 2
                left = pending[:mid]
                right = pending[mid:]
                pending.clear()
                _flush_batch(left, use_large_for_batch, depth=depth + 1)
                _flush_batch(right, use_large_for_batch, depth=depth + 1)
                return
            for item in pending:
                _record_error(item, "llm_call", f"bad_request_batch:{resp}", raw=str(resp))
            pending.clear()
            return
        if isinstance(resp, str) and resp.startswith("__VNPT_ERROR__"):
            stats["llm_errors"] += 1
            for item in pending:
                _record_error(item, "llm_call", f"vnpt_error:{resp}", raw=str(resp))
            pending.clear()
            return

        for item in pending:
            qid = item["qid"]
            ans = batch_map.get(qid)
            if ans and 0 <= (ord(ans) - ord("A")) < len(item["choices"]):
                final_answers[qid] = ans
                stats["solved_llm"] += 1
            else:
                stats["llm_errors"] += 1
                _record_error(item, "llm_invalid_answer", "missing_or_out_of_range", raw=str(resp))
        pending.clear()

        # Nhẹ nhàng tránh burst (không cần sleep dài vì batch đã giảm req)
        time.sleep(0.2)

    def flush_all():
        if llm_mode == "auto":
            _flush_batch(pending_small, use_large_for_batch=False)
            _flush_batch(pending_large, use_large_for_batch=True)
            return
        if llm_mode == "small":
            _flush_batch(pending_small, use_large_for_batch=False)
            return
        _flush_batch(pending_large, use_large_for_batch=True)

    for row in data:
        qid = row.get("qid")
        q_text = row.get("question", "") or ""
        choices = row.get("choices", []) or []
        try:
            context, clean_q = extract_context_and_question(q_text)
            domain = detect_domain(clean_q)
            math_like = _is_math_like(clean_q)
            allowed = _allowed_doc_types(domain if not math_like else "math")

            # Worker rewrite giúp retrieval (chỉ khi llm_mode=auto)
            rewrite = ""
            w = worker_map.get(str(qid)) or worker_map.get(qid) or {}
            rewrite = (w.get("rewrite") or "").strip()

            rag_context = ""
            if context and has_embedded_context(q_text):
                rag_context = _build_evidence_pack(clean_q, choices, context, max_chars=900)
            elif allowed:
                query_for_retrieval = clean_q
                if rewrite:
                    query_for_retrieval = f"{rewrite} | {clean_q[:180]}"
                docs = retrieve_docs_hybrid_filtered(
                    query_for_retrieval,
                    vnpt_index=vnpt_index,
                    client=client,
                    bm25_index=bm25_index,
                    allowed_doc_types=allowed,
                    top_k=4,
                )
                if docs:
                    joined = "\n\n---\n\n".join(d["text"][:800] for d in docs)
                    rag_context = _build_evidence_pack(clean_q, choices, joined, max_chars=900)

            item = {
                "qid": qid,
                "question": clean_q,
                "choices": choices,
                "context": rag_context,
                "domain": domain,
            }
        except Exception as e:
            item = {
                "qid": qid,
                "question": q_text,
                "choices": choices,
                "context": "",
                "domain": None,
            }
            _record_error(item, "build_item", f"{type(e).__name__}: {e}")
            continue

        # Route model
        if llm_mode == "large":
            pending_large.append(item)
        elif llm_mode == "small":
            pending_small.append(item)
        else:
            # auto: câu dễ -> Small, câu khó -> Large
            if should_use_large(item):
                pending_large.append(item)
            else:
                pending_small.append(item)

        # flush theo batch_size
        if llm_mode in ("small", "auto") and len(pending_small) >= batch_size:
            _flush_batch(pending_small, use_large_for_batch=False)
        if llm_mode in ("large", "auto") and len(pending_large) >= batch_size:
            _flush_batch(pending_large, use_large_for_batch=True)

    flush_all()

    # Strict: không fallback. Nếu có lỗi/thiếu đáp án -> log và crash.
    for row in data:
        if row["qid"] in final_answers:
            continue
        q_text = row.get("question", "") or ""
        context, clean_q = extract_context_and_question(q_text)
        item = {
            "qid": row.get("qid"),
            "question": clean_q,
            "choices": row.get("choices", []),
            "context": context if has_embedded_context(q_text) else "",
            "domain": detect_domain(clean_q),
        }
        _record_error(item, "missing_answer", "no_answer_generated")

    if errors:
        _write_errors(os.getenv("ERROR_LOG_PATH", "llm_errors.jsonl"))
        raise SystemExit(1)

    # Save CSV
    import csv

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer"])
        for row in data:
            writer.writerow([row["qid"], final_answers[row["qid"]]])

    print(f"[DONE] Saved {output_path}")
    print(f"[STATS] {stats}")


def main():
    args = parse_args()

    # Worker mặc định tắt (để giống mode cũ); bật khi user yêu cầu rõ ràng.
    use_worker_small = False if args.no_worker_small else bool(args.use_worker_small)
    run_inference_llm_batch(
        input_path=args.input,
        output_path=args.output,
        kb_dir=args.kb_dir,
        batch_size=max(1, args.llm_batch_size),
        llm_mode=args.llm_model,
        use_worker_small=use_worker_small,
        worker_batch_size=max(1, args.worker_batch_size),
        llm_n=max(1, int(args.llm_n)),
        llm_seed=int(args.llm_seed),
        large_top_p=float(args.large_top_p),
        large_top_k=int(args.large_top_k),
        small_top_p=float(args.small_top_p),
        small_top_k=int(args.small_top_k),
        presence_penalty=float(args.presence_penalty),
        frequency_penalty=float(args.frequency_penalty),
        use_logprobs=bool(args.use_logprobs),
        top_logprobs=int(args.top_logprobs),
        use_tools=bool(args.use_tools),
    )


if __name__ == "__main__":
    main()
