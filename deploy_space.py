"""Publish the app and its prebuilt Chroma index to a Hugging Face Space.

Usage:
    HF_TOKEN=hf_xxx python deploy_space.py --space <user>/anime-compass-search

The token needs write access. The Space uses the Gradio SDK on free CPU hardware.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
FILES = [
    "app.py",
    "search_core.py",
    "build_index.py",
    "requirements.txt",
    "README.md",
    "chromadb/**",
    "data/**",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", default="Seiitsu/anime-compass-search", help="<owner>/<space-name>")
    args = parser.parse_args()

    if not (ROOT / "chromadb" / "chroma.sqlite3").exists():
        raise SystemExit("chromadb/ is missing: run `python build_index.py` first.")

    api = HfApi()
    print(f"Logged in to Hugging Face as {api.whoami()['name']}")
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.space,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=FILES,
        delete_patterns=["chromadb/**"],  # drop stale HNSW segment folders from earlier builds
        commit_message="Deploy Anime Compass Search",
    )
    print(f"Pushed {commit.oid[:7]}")
    print(f"Space: https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
