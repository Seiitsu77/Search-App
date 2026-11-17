"""Measure search quality on queries with known answers.

Each case is a query plus the title(s) it should find. Matching is by
substring, so a sequel or movie of the right franchise counts as a hit.
Queries come in two styles:
  synopsis - phrased roughly the way the show's synopsis describes it
  fan      - the casual way a viewer would describe it
Searches use the app's default settings (format filter, popularity bonus).
Reports hit@1, hit@5, hit@10 and MRR@10, overall and per style.

Usage:
    python evaluate.py --out new.json --baseline baseline.json

With --baseline, prints a side-by-side comparison and exits 1 when MRR@10
falls more than --max-drop below the baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from search_core import DEFAULT_EXCLUDED_TYPES, POPULARITY_WEIGHT, AnimeSearchEngine, Filters

CASES: list[tuple[str, str, list[str]]] = [
    # -- synopsis-style ---------------------------------------------------------
    ("synopsis", "humanity lives behind giant walls to hide from man-eating giants", ["Shingeki no Kyojin"]),
    ("synopsis", "a student finds a notebook that kills anyone whose name is written in it", ["Death Note"]),
    ("synopsis", "two brothers use alchemy to try to get their bodies back after a failed transmutation", ["Fullmetal Alchemist"]),
    ("synopsis", "a self-proclaimed mad scientist sends text messages to the past with a microwave", ["Steins;Gate"]),
    ("synopsis", "a short boy dreams of becoming a great volleyball player like the little giant", ["Haikyuu"]),
    ("synopsis", "a boy who can see spirits and youkai inherits a book of their names", ["Natsume Yuujinchou"]),
    ("synopsis", "a mother raises two children who can turn into wolves", ["Ookami Kodomo"]),
    ("synopsis", "a salaryman is reincarnated in another world as a slime", ["Tensei shitara Slime"]),
    ("synopsis", "a boy becomes a demon slayer to turn his sister back into a human", ["Kimetsu no Yaiba"]),
    ("synopsis", "childhood friends reunite years later to help the ghost of a girl who died", ["Ano Hi Mita Hana"]),
    ("synopsis", "two girls travel through the ruins of civilization after the end of the world", ["Shoujo Shuumatsu Ryokou"]),
    ("synopsis", "a pianist stopped playing after his mother died until he meets a free-spirited violinist", ["Shigatsu wa Kimi no Uso"]),
    ("synopsis", "a wandering swordsman who was once a feared assassin vows never to kill again", ["Rurouni Kenshin"]),
    ("synopsis", "a girl goes camping alone near Mount Fuji and meets another girl who loves the outdoors", ["Yuru Camp"]),
    ("synopsis", "a boy is summoned by his father to pilot a giant robot and fight beings called angels", ["Evangelion"]),
    ("synopsis", "bounty hunters on a spaceship in 2071 after Earth's surface became uninhabitable", ["Cowboy Bebop"]),
    ("synopsis", "a high school girl in a rural town and a boy in Tokyo swap bodies in their dreams", ["Kimi no Na wa"]),
    ("synopsis", "a rubber boy sets out to sea to find the legendary treasure and become king of the pirates", ["One Piece"]),
    ("synopsis", "players are trapped in a virtual reality MMORPG where dying in the game means dying in real life", ["Sword Art Online"]),
    ("synopsis", "a girl works at a bathhouse for spirits to save her parents who were turned into pigs", ["Sen to Chihiro"]),
    ("synopsis", "orphans at an idyllic orphanage discover they are being raised as food for demons", ["Yakusoku no Neverland"]),
    ("synopsis", "a hero can defeat any enemy with a single punch and is bored because nobody challenges him", ["One Punch Man"]),
    # -- fan-style --------------------------------------------------------------
    ("fan", "space cowboys and jazz music", ["Cowboy Bebop"]),
    ("fan", "depressing mecha anime about kids piloting robots", ["Evangelion"]),
    ("fan", "cozy camping anime with cute girls", ["Yuru Camp"]),
    ("fan", "sad anime about the ghost of a dead childhood friend", ["Ano Hi Mita Hana"]),
    ("fan", "time travel thriller with a microwave", ["Steins;Gate"]),
    ("fan", "cute girls start a band in the high school light music club", ["K-On"]),
    ("fan", "cooking battles at an elite culinary school", ["Shokugeki no Souma"]),
    ("fan", "isekai where the main character is an undead skeleton overlord", ["Overlord"]),
    ("fan", "basketball anime with an invisible phantom sixth man", ["Kuroko no Basket"]),
    ("fan", "mages and their heroic spirits fight a battle royale for the holy grail", ["Fate/"]),
    ("fan", "a spy, an assassin and a telepathic girl pretend to be a family", ["Spy x Family"]),
    ("fan", "romantic comedy where two geniuses each try to make the other confess first", ["Kaguya-sama"]),
    ("fan", "dark fantasy about a mercenary with a giant sword seeking revenge", ["Berserk"]),
    ("fan", "a teen detective is shrunk into a child by a poison", ["Meitantei Conan"]),
    ("fan", "a psychic middle schooler suppresses his powers until his emotions hit 100 percent", ["Mob Psycho"]),
    ("fan", "a former child soldier writes letters for people while learning what love means", ["Violet Evergarden"]),
]
K = 10


def first_hit_rank(titles: list[str], targets: list[str]) -> int | None:
    for rank, title in enumerate(titles, 1):
        if any(target.lower() in title.lower() for target in targets):
            return rank
    return None


def summarize(ranks: list[int | None]) -> dict[str, float]:
    n = max(len(ranks), 1)
    return {
        "hit@1": sum(1 for r in ranks if r and r <= 1) / n,
        "hit@5": sum(1 for r in ranks if r and r <= 5) / n,
        "hit@10": sum(1 for r in ranks if r and r <= 10) / n,
        "mrr@10": sum(1 / r for r in ranks if r) / n,
    }


def evaluate(engine: AnimeSearchEngine) -> dict:
    filters = Filters(types=[t for t in engine.all_types() if t not in DEFAULT_EXCLUDED_TYPES])
    rows = []
    for style, query, targets in CASES:
        started = time.perf_counter()
        hits = engine.search(query, k=K, filters=filters, popularity_boost=POPULARITY_WEIGHT)
        elapsed = (time.perf_counter() - started) * 1000
        titles = [hit.title for hit in hits]
        rows.append({"style": style, "query": query, "rank": first_hit_rank(titles, targets), "top3": titles[:3], "ms": elapsed})

    result = {
        "model": engine.model_name,
        "overall": summarize([r["rank"] for r in rows]),
        "by_style": {
            style: summarize([r["rank"] for r in rows if r["style"] == style]) for style in ("synopsis", "fan")
        },
        "median_ms": sorted(r["ms"] for r in rows)[len(rows) // 2],
        "cases": rows,
    }
    return result


def fmt_rank(rank: int | None) -> str:
    return str(rank) if rank else "-"


def print_report(result: dict, baseline: dict | None) -> None:
    label = result["model"] + (f" (query prefix: {result['query_prefix']!r})" if result.get("query_prefix") is not None else "")
    print(f"\nModel: {label}   median query time: {result['median_ms']:.0f} ms\n")
    print(f"{'rank':>5} {'was':>4}  query  ->  top 3")
    old_ranks = {c["query"]: c["rank"] for c in (baseline or {}).get("cases", [])}
    for case in result["cases"]:
        was = fmt_rank(old_ranks.get(case["query"])) if baseline else ""
        print(f"{fmt_rank(case['rank']):>5} {was:>4}  [{case['style']}] {case['query']}")
        print(f"{'':12}-> {' | '.join(case['top3'])}")

    print()
    header = f"{'':10}" + "".join(f"{m:>10}" for m in ("hit@1", "hit@5", "hit@10", "mrr@10"))
    print(header)
    for name in ("overall", "synopsis", "fan"):
        new = result["overall"] if name == "overall" else result["by_style"][name]
        print(f"{name:10}" + "".join(f"{new[m]:>10.3f}" for m in ("hit@1", "hit@5", "hit@10", "mrr@10")))
        if baseline:
            old = baseline["overall"] if name == "overall" else baseline["by_style"][name]
            print(f"{'  baseline':10}" + "".join(f"{old[m]:>10.3f}" for m in ("hit@1", "hit@5", "hit@10", "mrr@10")))
    if baseline:
        print(f"\nBaseline model: {baseline['model']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="write results as JSON")
    parser.add_argument("--baseline", type=Path, help="earlier results JSON to compare against")
    parser.add_argument("--max-drop", type=float, default=0.02, help="allowed MRR@10 drop vs the baseline")
    parser.add_argument("--query-prefix", help="override the model's query instruction ('' for none)")
    args = parser.parse_args()

    engine = AnimeSearchEngine.load(query_prefix=args.query_prefix)
    result = evaluate(engine)
    result["query_prefix"] = args.query_prefix

    # Sanity checks on filtering and "more like this".
    for hit in engine.search(
        "giant robots", k=5, filters=Filters(types=["Movie"], year_min=2000, min_score=7.5), popularity_boost=0.05
    ):
        assert hit.anime["type"] == "Movie" and hit.anime["year"] >= 2000 and hit.anime["score"] >= 7.5, hit
    death_note = engine.resolve_title("Death Note")
    result["more_like_death_note"] = [h.title for h in engine.similar_to(death_note["anime_id"], k=5)]

    baseline = json.loads(args.baseline.read_text()) if args.baseline and args.baseline.exists() else None
    print_report(result, baseline)
    print(f"More like Death Note: {' | '.join(result['more_like_death_note'])}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    if baseline and result["overall"]["mrr@10"] < baseline["overall"]["mrr@10"] - args.max_drop:
        print(
            f"\nFAIL: MRR@10 {result['overall']['mrr@10']:.3f} is below the baseline "
            f"{baseline['overall']['mrr@10']:.3f} by more than {args.max_drop}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
