"""Sanity-check the built index with the real embedding model.

Runs plot-description queries whose answer is well known, prints the top
results, and fails if too many of the expected titles are missing from the
top 5. Used by the GitHub Action after build_index.py.
"""

from __future__ import annotations

import sys
import time

from search_core import AnimeSearchEngine, Filters, sentence_transformer_embedder

# (query, substring expected in one of the top-5 titles)
CASES = [
    ("humanity lives behind giant walls to hide from man-eating giants", "Shingeki no Kyojin"),
    ("a student finds a notebook that kills anyone whose name is written in it", "Death Note"),
    ("a boy and a girl in different towns mysteriously swap bodies while a comet approaches", "Kimi no Na wa"),
    ("two brothers use alchemy to try to get their bodies back after a failed transmutation", "Fullmetal Alchemist"),
    ("bounty hunters travel the solar system on a spaceship called the Bebop", "Cowboy Bebop"),
    ("a short boy dreams of becoming a great volleyball player like the little giant", "Haikyuu"),
    ("a self-proclaimed mad scientist sends text messages to the past with a microwave", "Steins;Gate"),
    ("a girl moves to the countryside and meets a giant forest spirit at a bus stop", "Tonari no Totoro"),
]
MIN_PASS = 6


def main() -> int:
    started = time.perf_counter()
    engine = AnimeSearchEngine.load(sentence_transformer_embedder())
    print(f"Loaded {engine.count():,} chunks / {len(engine.catalog):,} titles in {time.perf_counter() - started:.1f}s\n")

    passed = 0
    for query, expected in CASES:
        started = time.perf_counter()
        hits = engine.search(query, k=5, popularity_boost=0.05)
        elapsed = (time.perf_counter() - started) * 1000
        found = any(expected.lower() in hit.title.lower() for hit in hits)
        passed += found
        print(f"[{'PASS' if found else 'MISS'}] {query}  ({elapsed:.0f} ms)")
        for i, hit in enumerate(hits, 1):
            print(f"    {i}. {hit.title}  sim={hit.similarity:.3f}  score={hit.score:.3f}")

    print("\nFiltered: 'giant robots' as movies from 2000 on with score >= 7.5")
    for hit in engine.search(
        "giant robots", k=5, filters=Filters(types=["Movie"], year_min=2000, min_score=7.5), popularity_boost=0.05
    ):
        print(f"    {hit.title} ({hit.anime['type']}, {hit.anime['year']}, {hit.anime['score']})")
        assert hit.anime["type"] == "Movie" and hit.anime["year"] >= 2000 and hit.anime["score"] >= 7.5

    anime = engine.resolve_title("Death Note")
    print(f"\nMore like {anime['title']}:")
    for hit in engine.similar_to(anime["anime_id"], k=5, popularity_boost=0.05):
        print(f"    {hit.title}  sim={hit.similarity:.3f}")

    print(f"\n{passed}/{len(CASES)} expected titles found in the top 5 (need {MIN_PASS}).")
    return 0 if passed >= MIN_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
