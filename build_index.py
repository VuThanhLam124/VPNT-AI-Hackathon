"""
Build embedding index cho KB bằng BGE-m3 offline.
Output mặc định: kb_bge_m3_index.pkl
"""
import argparse
from generate_simple import build_kb_embedding_index, BGE_INDEX_PATH, MAX_CHUNK_CHARS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_dir", default="data/converted")
    parser.add_argument("--max_chunk_chars", type=int, default=MAX_CHUNK_CHARS)
    parser.add_argument(
        "--out",
        default="",
        help="Đường dẫn file index output (pickle). Nếu bỏ trống sẽ dùng mặc định.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
        print(f"✓ Built index với {len(kb_emb['docs'])} docs -> {out_path}")


if __name__ == "__main__":
    main()
