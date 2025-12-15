import json
import re
from typing import List, Dict, Iterable, Optional, Tuple, Iterator
import logging

logger = logging.getLogger(__name__)


def has_embedded_context(question_text: str) -> bool:
    """
    Kiểm tra câu hỏi có context nhúng không
    
    Keywords: Đoạn thông tin, Title, Content, Câu hỏi
    """
    keywords = [
        'Đoạn thông tin:',
        'Title:',
        'Content:',
        'Nội dung:',
        'Câu hỏi:',
        '[1]',  # Numbered sections
        'I.',   # Roman numerals
    ]
    
    for kw in keywords:
        if kw in question_text:
            return True
    return False


def extract_context_and_question(full_text: str) -> tuple:
    """
    Tách context và câu hỏi thực sự
    
    Returns:
        (context, question)
    """
    # Pattern: "Câu hỏi:" marks the actual question
    if 'Câu hỏi:' in full_text:
        parts = full_text.split('Câu hỏi:')
        context = parts[0].strip()
        question = parts[1].strip() if len(parts) > 1 else full_text
        return context, question
    
    return "", full_text


def detect_domain(question_text: str) -> str:
    """
    Phát hiện domain của câu hỏi
    """
    text_lower = question_text.lower()
    
    # Toán học
    math_patterns = [
        'tính', 'phương trình', 'tìm x', '=', '+', '-', '*', '/', 
        'sin', 'cos', 'tan', 'log', 'căn', 'lũy thừa', 'đạo hàm', 'tích phân',
        'hàm số', 'điểm uốn', 'vector', 'véc-tơ', 'cotang', 'cotan', 'cournot',
    ]
    if any(p in text_lower for p in math_patterns):
        return 'math'
    
    # Tư tưởng HCM
    hcm_patterns = ['hồ chí minh', 'bác hồ', 'nguyễn ái quốc', 'chủ tịch hồ chí minh',
                    'tư tưởng hcm', 'cách mạng', 'giải phóng dân tộc']
    if any(p in text_lower for p in hcm_patterns):
        return 'tu_tuong_hcm'

    # Hoá học / nguyên tố
    chem_patterns = [
        'nguyên tố', 'bảng tuần hoàn', 'khối lượng nguyên tử', 'atomic weight',
        'hóa học', 'hoá học', 'mol', 'molar', 'số hiệu nguyên tử', 'z=',
        'kí hiệu', 'ký hiệu', 'symbol',
    ]
    if any(p in text_lower for p in chem_patterns):
        return 'chemistry'

    # Lịch sử VN
    history_patterns = ['lịch sử', 'chiến tranh', 'năm 19', 'triều đại', 'vua', 'chùa', 'đền',
                       'di tích', 'văn hóa', 'truyền thống']
    if any(p in text_lower for p in history_patterns):
        return 'lich_su'

    # Luật pháp
    law_patterns = [
        'luật', 'nghị định', 'thông tư', 'điều', 'khoản', 'quy định', 'chế tài',
        'vi phạm', 'phạt', 'hình phạt',
        'hồ sơ', 'thẩm duyệt', 'thủ tục', 'công văn', 'giấy tờ',
        'phòng cháy', 'chữa cháy', 'chứng nhận', 'đăng ký'
    ]
    if any(p in text_lower for p in law_patterns):
        return 'law'
    
    # Địa lý
    geo_patterns = ['tỉnh', 'thành phố', 'quận', 'huyện', 'xã', 'phường', 'cảng', 'sông', 'núi']
    if any(p in text_lower for p in geo_patterns):
        return 'geography'
    
    return 'general'


def is_sensitive_question(question_text: str) -> bool:
    """
    Kiểm tra câu hỏi nhạy cảm (từ chối trả lời)
    """
    sensitive_patterns = [
        'làm cách nào để tránh',
        'làm cách nào để lừa',
        'làm cách nào để gian lận',
        'khai báo thông tin sai',
        'trốn thuế',
        'vi phạm pháp luật'
    ]
    
    text_lower = question_text.lower()
    # Nội dung nhạy cảm/thuần phong mỹ tục (tránh gửi lên LLM để khỏi bị block theo batch)
    extra = [
        'quan hệ tình dục', 'tình dục', 'khiêu dâm', 'dâm ô', 'porn', 'xxx',
        'hiếp dâm', 'cưỡng hiếp', 'ấu dâm',
        'ma túy', 'ma tuy', 'cần sa', 'heroin', 'cocaine',
        'tự sát', 'tự tử', 'giết người', 'đánh bom', 'chế tạo bom',
    ]
    return any(p in text_lower for p in sensitive_patterns) or any(p in text_lower for p in extra)


def load_knowledge_base(kb_dir: str = 'data/converted') -> List[Dict]:
    """
    Load tất cả knowledge base dạng Q/A từ thư mục converted.

    Lưu ý: Chỉ load các dòng JSONL có đủ 2 trường `question` và `answer`.
    Các file dữ liệu dạng bảng/đoạn văn (vd: `ward.jsonl`, `unesco_vietnam.jsonl`,
    `ho-chi-minh_data.jsonl`, `tinh_converted.jsonl`) sẽ không được load bởi hàm này.
    Nếu cần load toàn bộ KB (nhiều format), dùng `load_kb_corpus()`.
    
    Returns:
        List of {"question": "...", "answer": "..."}
    """
    import os
    import glob
    
    kb_data = []
    
    # Tìm tất cả file .jsonl
    pattern = os.path.join(kb_dir, '*.jsonl')
    files = glob.glob(pattern)
    
    logger.info(f"Found {len(files)} KB files")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        if 'question' in item and 'answer' in item:
                            kb_data.append(item)
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue
    
    logger.info(f"Loaded {len(kb_data)} KB entries")
    return kb_data


def _tokenize_vi(text: str) -> List[str]:
    """
    Tokenize đơn giản cho tiếng Việt (không tách từ nâng cao).
    Dùng cho retrieval heuristic/BM25-lite.
    """
    if not text:
        return []
    text = text.lower()
    # Match unicode letters/digits, bỏ underscore và punctuation.
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def _iter_json_objects_from_mixed_jsonl(path: str) -> Iterator[Dict]:
    """
    Load các object JSON từ file "jsonl" nhưng thực tế có thể gặp:
    - JSONL chuẩn: mỗi dòng là 1 object
    - JSONL có dấu phẩy cuối dòng: `{...},`
    - "Pseudo JSON array" không có dấu []: nhiều object multi-line, ngăn bởi `},`
    - JSON array thật sự: `[...]`

    Mục tiêu: đảm bảo mỗi `{...}` được coi là 1 chunk/object.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    # 1) Try JSON array/object (full file)
    try:
        s = raw.strip()
        if s.startswith("[") or s.startswith("{"):
            obj = json.loads(s)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item
                return
            if isinstance(obj, dict):
                yield obj
                return
    except Exception:
        pass

    # 2) Try JSONL line-by-line (allow trailing comma / stray trailing backslash)
    yielded_any = False
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        # Some files contain lines like `... },\` (stray backslash at EOL)
        s = s.rstrip("\\").rstrip()
        if s.endswith(","):
            s = s[:-1].rstrip()
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            yielded_any = True
            yield obj
    if yielded_any:
        return

    # 3) Fallback: decode sequential JSON objects from raw text (pseudo-array, multi-line objects)
    decoder = json.JSONDecoder()
    i = 0
    n = len(raw)
    while i < n:
        # Skip separators/whitespace between objects
        while i < n and raw[i] in " \t\r\n,\\":
            i += 1
        if i >= n:
            break
        if raw[i] not in "[{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(raw, i)
        except Exception:
            i += 1
            continue
        i = end
        if isinstance(obj, dict):
            yield obj
            continue
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
            # If the whole file is a list but `json.loads` failed due to junk
            # at the head/tail, this still returns useful objects.
            continue


def load_kb_corpus(kb_dir: str = "data/converted") -> List[Dict]:
    """
    Load KB từ `data/converted` với nhiều định dạng khác nhau và chuẩn hoá về dạng document.

    Output mỗi phần tử là dict có các khoá chính:
      - text: nội dung để retrieval/matching
      - source_file: tên file nguồn
      - doc_type: loại document (qa/ward/unesco/hcm/province/kv)
      - answer: (tuỳ chọn) câu trả lời ngắn nếu có (dạng Q/A)

    Mục tiêu: dùng làm corpus cho RAG (retrieval) mà không bị fail do file không phải JSONL chuẩn.
    """
    import os
    import glob

    docs: List[Dict] = []
    paths = sorted(glob.glob(os.path.join(kb_dir, "*.jsonl")))
    logger.info(f"Found {len(paths)} KB files (mixed formats)")

    for path in paths:
        source_file = os.path.basename(path)

        # File này thực chất là JSON array (không phải JSONL).
        if source_file == "tinh_converted.jsonl":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    items = json.load(f)
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        ten = str(item.get("tên tỉnh", "")).strip()
                        sap = str(item.get("phương án sáp nhập", "")).strip()
                        trung_tam = str(item.get("trung tâm hành chính", "")).strip()
                        dien_tich = item.get("diện tích km2", "")
                        dan_so = item.get("dân số", "")
                        text = (
                            f"{ten}. Phương án sáp nhập: {sap}. "
                            f"Trung tâm hành chính: {trung_tam}. "
                            f"Diện tích: {dien_tich}. Dân số: {dan_so}."
                        ).strip()
                        if len(text) >= 20:
                            docs.append(
                                {
                                    "text": text,
                                    "source_file": source_file,
                                    "doc_type": "province",
                                    "answer": None,
                                }
                            )
            except Exception as e:
                logger.warning(f"Error loading {path} as JSON array: {e}")
            continue

        # Các file JSON/JSONL (mixed): mỗi object `{...}` là một chunk.
        try:
            for obj in _iter_json_objects_from_mixed_jsonl(path):
                if not isinstance(obj, dict):
                    continue

                # Dạng Q/A
                if "question" in obj and "answer" in obj:
                    q = str(obj.get("question", "")).strip()
                    a = str(obj.get("answer", "")).strip()
                    if not q:
                        continue
                    text = f"Hỏi: {q}\nĐáp: {a}".strip()
                    docs.append(
                        {
                            "text": text,
                            "source_file": source_file,
                            "doc_type": "qa",
                            "answer": a if a else None,
                        }
                    )
                    continue

                # ward.jsonl
                if {"Name", "Type", "ProvinceName"}.issubset(obj.keys()):
                    name = str(obj.get("Name", "")).strip()
                    type_ = str(obj.get("Type", "")).strip()
                    province = str(obj.get("ProvinceName", "")).strip()
                    code = str(obj.get("Code", "")).strip()
                    if not name or not province:
                        continue
                    text = f"{type_} {name} thuộc {province}. Mã: {code}".strip()
                    docs.append(
                        {
                            "text": text,
                            "source_file": source_file,
                            "doc_type": "ward",
                            "answer": None,
                        }
                    )
                    continue

                # unesco_vietnam.jsonl
                if {"tên", "năm được công nhận", "địa điểm"}.issubset(obj.keys()):
                    ten = str(obj.get("tên", "")).strip()
                    nam = str(obj.get("năm được công nhận", "")).strip()
                    loai = str(obj.get("loại hình được công nhận", "")).strip()
                    dia_diem = str(obj.get("địa điểm", "")).strip()
                    if not ten:
                        continue
                    text = (
                        f"{ten} được công nhận năm {nam}. "
                        f"Loại hình: {loai}. Địa điểm: {dia_diem}."
                    ).strip()
                    docs.append(
                        {
                            "text": text,
                            "source_file": source_file,
                            "doc_type": "unesco",
                            "answer": None,
                        }
                    )
                    continue

                # temple.jsonl (tên chùa / thời gian / địa điểm / thông tin)
                if "tên chùa" in obj:
                    ten = str(obj.get("tên chùa", "")).strip()
                    tg = obj.get("thời gian thành lập", None)
                    dd = obj.get("địa điểm", None)
                    tt = str(obj.get("thông tin", "")).strip()
                    if not ten and len(tt) < 20:
                        continue
                    text = f"{ten}. Thành lập: {tg}. Địa điểm: {dd}. {tt}".strip()
                    docs.append(
                        {
                            "text": text,
                            "source_file": source_file,
                            "doc_type": "temple",
                            "answer": None,
                        }
                    )
                    continue

                # atomic weights (Z/symbol/atomic_weight)
                if {"Z", "symbol", "atomic_weight"}.issubset(obj.keys()):
                    z = obj.get("Z")
                    sym = obj.get("symbol")
                    name = obj.get("name")
                    aw = obj.get("atomic_weight")
                    text = f"Nguyên tố {name} ({sym}), Z={z}, khối lượng nguyên tử={aw}."
                    docs.append(
                        {
                            "text": text,
                            "source_file": source_file,
                            "doc_type": "atomic",
                            "answer": None,
                        }
                    )
                    continue

                # STEM formulas (tên công thức / latex / danh mục / chủ đề)
                if {"ten_cong_thuc", "cong_thuc_latex"}.issubset(obj.keys()):
                    ten = str(obj.get("ten_cong_thuc", "")).strip()
                    latex = str(obj.get("cong_thuc_latex", "")).strip()
                    dm = str(obj.get("danh_muc", "")).strip()
                    cd = str(obj.get("chu_de", "")).strip()
                    if not ten and not latex:
                        continue
                    parts = [p for p in [f"Công thức: {ten}" if ten else "", f"LaTeX: {latex}" if latex else "", f"Danh mục: {dm}" if dm else "", f"Chủ đề: {cd}" if cd else ""] if p]
                    text = ". ".join(parts).strip()
                    if len(text) < 20:
                        continue
                    docs.append(
                        {
                            "text": text,
                            "source_file": source_file,
                            "doc_type": "stem",
                            "answer": None,
                        }
                    )
                    continue

                # ho-chi-minh_data.jsonl (document sections / events)
                if "content" in obj:
                    content = str(obj.get("content", "")).strip()
                    if len(content) < 20:
                        continue
                    title = str(obj.get("section_title", "")).strip()
                    prefix = f"{title}: " if title else ""
                    text = (prefix + content).strip()
                    docs.append(
                        {
                            "text": text,
                            "source_file": source_file,
                            "doc_type": "hcm",
                            "answer": None,
                        }
                    )
                    continue
                if {"date", "event"}.issubset(obj.keys()):
                    date = str(obj.get("date", "")).strip()
                    event = str(obj.get("event", "")).strip()
                    location = str(obj.get("location", "")).strip()
                    if not event:
                        continue
                    text = f"{date}: {event}. Địa điểm: {location}.".strip()
                    if len(text) >= 20:
                        docs.append(
                            {
                                "text": text,
                                "source_file": source_file,
                                "doc_type": "hcm_event",
                                "answer": None,
                            }
                        )
                    continue

                # Fallback: key-value summary
                kv_parts = []
                for k, v in obj.items():
                    if v is None:
                        continue
                    v_str = str(v).strip()
                    if not v_str:
                        continue
                    kv_parts.append(f"{k}: {v_str}")
                text = ". ".join(kv_parts).strip()
                if len(text) >= 30:
                    docs.append(
                        {
                            "text": text,
                            "source_file": source_file,
                            "doc_type": "kv",
                            "answer": None,
                        }
                    )

        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            continue

    logger.info(f"Loaded {len(docs)} KB documents")
    return docs


def load_dataset(file_path: str) -> List[Dict]:
    """
    Load val.json hoặc test.json
    
    Returns:
        List of questions
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} questions from {file_path}")
    return data


if __name__ == "__main__":
    # Test functions
    logging.basicConfig(level=logging.INFO)
    
    # Test domain detection
    test_questions = [
        "Tính đạo hàm của hàm số y = x^2 + 3x - 1",
        "Theo Hồ Chí Minh, con đường giải phóng dân tộc là gì?",
        "Luật Bảo vệ môi trường 2020 có bao nhiêu nguyên tắc?",
        "Cảng nào nằm ở phía Bắc cảng Vũng Áng?"
    ]
    
    for q in test_questions:
        domain = detect_domain(q)
        print(f"{domain:15s} | {q[:60]}...")
    
    # Test KB loading
    kb = load_knowledge_base()
    print(f"\nKB loaded: {len(kb)} entries")
    if kb:
        print(f"Sample: {kb[0]}")

    # Test KB corpus loading
    corpus = load_kb_corpus()
    print(f"\nKB corpus loaded: {len(corpus)} docs")
    if corpus:
        print(f"Sample doc: {corpus[0]['source_file']} | {corpus[0]['doc_type']} | {corpus[0]['text'][:120]}...")
