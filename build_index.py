"""Build the Chroma vector database used by app.py.

Steps (matching the course starter notebook):
  1. Pick a dataset   - the Anime Compass catalog (18k titles, CC0 MyAnimeList data)
  2. Chunk + embed    - sentence windows of each synopsis, embedded with a
                        title/genre header in front, all-MiniLM-L6-v2
  3. Store in Chroma  - persistent collection written to ./chromadb, plus
                        data/anime_meta.json (one row of display fields per anime)

Usage:
    python build_index.py                       # download the catalog from Hugging Face
    python build_index.py --catalog-file x.json # use a local copy instead
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from search_core import (
    CHUNK_CHARS,
    DEFAULT_DB_DIR,
    DEFAULT_META_PATH,
    MODEL_NAME,
    build_chunks,
    sentence_transformer_embedder,
    write_index,
)

DATASET_REPO = "Seiitsu/anime-compass-data"
DATASET_FILE = "anime_catalog.json"
# GitHub rejects files over 100 MB without Git LFS.
MAX_FILE_MB = 95


def load_catalog(path: Path | None) -> list[dict]:
    if path is None:
        from huggingface_hub import hf_hub_download

        print(f"Downloading {DATASET_FILE} from the {DATASET_REPO} dataset ...")
        path = Path(hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset"))
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dir_size_mb(path: Path) -> dict[str, float]:
    return {
        str(file.relative_to(path)): file.stat().st_size / 1e6 for file in sorted(path.rglob("*")) if file.is_file()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog-file", type=Path, help="local anime_catalog.json (default: download)")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--meta-path", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument("--max-chars", type=int, default=CHUNK_CHARS, help="max characters per chunk")
    args = parser.parse_args(argv)

    catalog = load_catalog(args.catalog_file)
    chunks, records = build_chunks(catalog, max_chars=args.max_chars)
    titles = len(records)
    per_title = Counter(chunk.metadata["anime_id"] for chunk in chunks)
    lengths = [len(chunk.embed_text) for chunk in chunks]
    print(f"Catalog rows: {len(catalog):,}  ->  titles indexed: {titles:,}  ->  chunks: {len(chunks):,}")
    print(
        f"Chunks per title: mean {statistics.mean(per_title.values()):.2f}, max {max(per_title.values())}; "
        f"chunk length: median {statistics.median(lengths):.0f} chars, max {max(lengths)}"
    )
    print("Example chunk (as embedded):\n  " + chunks[0].embed_text[:400].replace("\n", "\n  "))

    if args.db_dir.exists():
        shutil.rmtree(args.db_dir)
    print(f"Loading {MODEL_NAME} ...")
    embed = sentence_transformer_embedder()

    started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done == total or done % 5120 < 256:
            rate = done / max(time.perf_counter() - started, 1e-6)
            print(f"  embedded {done:,}/{total:,} chunks ({rate:,.0f}/s)", flush=True)

    collection = write_index(chunks, records, embed, db_dir=args.db_dir, meta_path=args.meta_path, progress=progress)
    print(f"Stored {collection.count():,} chunks in {args.db_dir} in {time.perf_counter() - started:.0f}s")

    sizes = dir_size_mb(args.db_dir)
    sizes[str(args.meta_path.name)] = args.meta_path.stat().st_size / 1e6
    for name, size in sizes.items():
        print(f"  {name}: {size:.1f} MB")
    too_big = [name for name, size in sizes.items() if size > MAX_FILE_MB]
    if too_big:
        print(f"WARNING: over {MAX_FILE_MB} MB, needs Git LFS to push to GitHub: {too_big}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
