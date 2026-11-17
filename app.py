"""Anime Compass Search: a semantic search engine over 18k anime synopses.

Gradio front end for the Chroma index built by build_index.py. Run locally with
`python app.py`; on Hugging Face Spaces this file is the entry point.
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr

from search_core import (
    CHUNK_CHARS,
    DEFAULT_DB_DIR,
    DEFAULT_META_PATH,
    MODEL_NAME,
    AnimeSearchEngine,
    Filters,
    Hit,
    sentence_transformer_embedder,
)

POPULARITY_WEIGHT = 0.05
DEFAULT_EXCLUDED_TYPES = {"Music", "Unknown"}

EXAMPLES = [
    "space bounty hunters drifting through the solar system to a jazz soundtrack",
    "a high school volleyball team with a short middle blocker who can jump really high",
    "teenagers forced to pilot giant robots while dealing with depression",
    "girls go camping in winter and cook food outdoors",
    "a scientist builds a microwave that sends messages to the past",
    "two girls wander a ruined post-apocalyptic city on a small tank",
    "a cooking competition where every dish is a battle",
    "a lonely ghost girl who can't move on until her childhood friends reunite",
]

CSS = """
.ac-hero h1 { margin-bottom: 0.2rem; }
.ac-hero p { opacity: 0.8; margin-top: 0; }
.ac-status { font-size: 0.9rem; opacity: 0.8; margin: 0.25rem 0 0.75rem; }
.ac-card { display: flex; gap: 14px; padding: 12px; margin-bottom: 12px;
  border: 1px solid var(--border-color-primary); border-radius: 12px;
  background: var(--block-background-fill); }
.ac-poster { width: 86px; min-width: 86px; height: 122px; border-radius: 8px; object-fit: cover;
  background: var(--background-fill-secondary); }
.ac-poster-empty { display: flex; align-items: center; justify-content: center; font-size: 1.8rem; }
.ac-body { flex: 1; min-width: 0; }
.ac-title { font-weight: 650; font-size: 1.05rem; line-height: 1.25; }
.ac-title a { color: inherit; text-decoration: none; }
.ac-title a:hover { text-decoration: underline; }
.ac-rank { opacity: 0.55; margin-right: 4px; }
.ac-meta { font-size: 0.82rem; opacity: 0.75; margin: 2px 0 6px; }
.ac-pill { display: inline-block; font-size: 0.72rem; padding: 1px 8px; margin: 0 4px 4px 0;
  border-radius: 999px; background: var(--background-fill-secondary);
  border: 1px solid var(--border-color-primary); }
.ac-match { display: flex; align-items: center; gap: 8px; font-size: 0.78rem; opacity: 0.85; margin: 2px 0 6px; }
.ac-bar { flex: 0 0 120px; height: 6px; border-radius: 3px; background: var(--background-fill-secondary); overflow: hidden; }
.ac-bar > span { display: block; height: 100%; background: var(--color-accent); }
.ac-snippet { font-size: 0.9rem; line-height: 1.45; margin: 0; }
.ac-empty { padding: 24px; text-align: center; opacity: 0.7; }
@media (max-width: 640px) {
  .ac-poster { width: 64px; min-width: 64px; height: 92px; }
  .ac-bar { flex-basis: 70px; }
}
"""


def load_engine() -> AnimeSearchEngine:
    db_dir = Path(os.environ.get("CHROMA_DIR", DEFAULT_DB_DIR))
    meta_path = Path(os.environ.get("ANIME_META", DEFAULT_META_PATH))
    if not (db_dir / "chroma.sqlite3").exists() or not meta_path.exists():
        # Fresh checkout without the committed index: build it (downloads data, ~5 min on CPU).
        import build_index

        build_index.main(["--db-dir", str(db_dir), "--meta-path", str(meta_path)])
    return AnimeSearchEngine.load(sentence_transformer_embedder(), db_dir=db_dir, meta_path=meta_path)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt_members(members: int) -> str:
    if members >= 1_000_000:
        return f"{members / 1_000_000:.1f}M members"
    if members >= 1_000:
        return f"{members / 1_000:.0f}K members"
    return f"{members} members"


def render_hit(rank: int, hit: Hit) -> str:
    anime = hit.anime
    title = html.escape(hit.title)
    url = f"https://myanimelist.net/anime/{hit.anime_id}"
    image = str(anime.get("image_url") or "")
    poster = (
        f'<img class="ac-poster" src="{html.escape(image, quote=True)}" alt="" loading="lazy" referrerpolicy="no-referrer">'
        if image.startswith("https://")
        else '<div class="ac-poster ac-poster-empty">🎬</div>'
    )
    meta = [anime.get("type") or ""]
    if anime.get("year"):
        meta.append(str(anime["year"]))
    if anime.get("episodes"):
        meta.append(f'{anime["episodes"]} ep' + ("s" if anime["episodes"] != 1 else ""))
    if anime.get("score"):
        meta.append(f'★ {anime["score"]:.2f}')
    if anime.get("members"):
        meta.append(_fmt_members(int(anime["members"])))
    if anime.get("studios"):
        meta.append(", ".join(anime["studios"]))
    pills = "".join(f'<span class="ac-pill">{html.escape(g)}</span>' for g in anime.get("genres", [])[:8])
    percent = max(0.0, min(hit.similarity, 1.0)) * 100
    snippet = html.escape(hit.snippet) if hit.snippet else "<em>No synopsis available.</em>"
    return f"""
<div class="ac-card">
  {poster}
  <div class="ac-body">
    <div class="ac-title"><span class="ac-rank">{rank}.</span><a href="{url}" target="_blank" rel="noopener">{title}</a></div>
    <div class="ac-meta">{html.escape(" · ".join(m for m in meta if m))}</div>
    <div>{pills}</div>
    <div class="ac-match"><span>Match {percent:.0f}%</span><span class="ac-bar"><span style="width:{percent:.0f}%"></span></span></div>
    <p class="ac-snippet">{snippet}</p>
  </div>
</div>"""


def render_hits(hits: list[Hit], status: str) -> str:
    body = "".join(render_hit(i, hit) for i, hit in enumerate(hits, 1))
    if not hits:
        body = '<div class="ac-empty">No matches. Try a different description or loosen the filters.</div>'
    return f'<div class="ac-status">{status}</div>{body}'


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def build_demo(engine: AnimeSearchEngine) -> gr.Blocks:
    types = engine.all_types()
    years = [a["year"] for a in engine.catalog.values() if a.get("year")]
    year_lo, year_hi = (min(years), max(years)) if years else (1917, 2025)
    default_types = [t for t in types if t not in DEFAULT_EXCLUDED_TYPES]

    def make_filters(type_sel, genre_sel, year_from, year_to, min_score) -> Filters:
        return Filters(
            types=type_sel if type_sel and len(type_sel) < len(types) else None,
            genres=genre_sel or None,
            year_min=int(year_from) if year_from and int(year_from) > year_lo else None,
            year_max=int(year_to) if year_to and int(year_to) < year_hi else None,
            min_score=float(min_score) if min_score else None,
        )

    def run_search(query, type_sel, genre_sel, year_from, year_to, min_score, k, popular):
        query = (query or "").strip()
        if not query:
            return render_hits([], "Type a description of the anime you're looking for.")
        if not type_sel:
            return render_hits([], "Select at least one format.")
        filters = make_filters(type_sel, genre_sel, year_from, year_to, min_score)
        hits = engine.search(query, k=int(k), filters=filters, popularity_boost=POPULARITY_WEIGHT if popular else 0.0)
        return render_hits(hits, f"Top {len(hits)} titles for <b>“{html.escape(query)}”</b>")

    def run_similar(title, type_sel, genre_sel, year_from, year_to, min_score, k, popular):
        if not title or not str(title).strip():
            return render_hits([], "Pick or type an anime title.")
        if not type_sel:
            return render_hits([], "Select at least one format.")
        anime = engine.resolve_title(str(title))
        if anime is None:
            return render_hits([], f"Couldn't find a title matching “{html.escape(str(title))}”.")
        filters = make_filters(type_sel, genre_sel, year_from, year_to, min_score)
        hits = engine.similar_to(
            anime["anime_id"], k=int(k), filters=filters, popularity_boost=POPULARITY_WEIGHT if popular else 0.0
        )
        return render_hits(hits, f"Titles whose synopses read most like <b>{html.escape(anime['title'])}</b>")

    with gr.Blocks(title="Anime Compass Search") as demo:
        gr.HTML(
            f"""<div class="ac-hero"><h1>🧭 Anime Compass Search</h1>
<p>Describe a plot, a vibe or a theme in your own words. This semantic search engine finds anime whose
synopses mean the same thing, even when they share no keywords. It searches {len(engine.catalog):,} titles
split into {engine.count():,} synopsis chunks, embedded with <code>{MODEL_NAME.split("/")[-1]}</code> and
stored in a Chroma vector database.</p></div>"""
        )

        with gr.Tabs():
            with gr.Tab("🔎 Search"):
                with gr.Row(equal_height=True):
                    query = gr.Textbox(
                        label="What are you in the mood for?",
                        placeholder="e.g. a samurai with a cross-shaped scar who has sworn never to kill again",
                        scale=5,
                        autofocus=True,
                    )
                    search_btn = gr.Button("Search", variant="primary", scale=1)
                gr.Examples(EXAMPLES, inputs=query, label="Try one")
            with gr.Tab("✨ More like this"):
                with gr.Row(equal_height=True):
                    title = gr.Dropdown(
                        choices=engine.popular_titles(3000),
                        value=None,
                        label="Pick an anime (or type any title)",
                        allow_custom_value=True,
                        filterable=True,
                        scale=5,
                    )
                    similar_btn = gr.Button("Find similar", variant="primary", scale=1)

        with gr.Accordion("Filters", open=False):
            with gr.Row():
                type_sel = gr.CheckboxGroup(types, value=default_types, label="Format")
            with gr.Row():
                genre_sel = gr.Dropdown(
                    engine.all_genres(), multiselect=True, label="Must include genres", scale=2
                )
                min_score = gr.Slider(0, 9, value=0, step=0.5, label="Minimum MAL score (0 = any)", scale=1)
            with gr.Row():
                year_from = gr.Slider(year_lo, year_hi, value=year_lo, step=1, label="From year")
                year_to = gr.Slider(year_lo, year_hi, value=year_hi, step=1, label="To year")
            with gr.Row():
                k = gr.Slider(5, 30, value=10, step=1, label="Number of results")
                popular = gr.Checkbox(
                    value=True,
                    label="Favor well-known titles",
                    info="Adds a small bonus by MyAnimeList membership to break near-ties.",
                )

        results = gr.HTML(render_hits([], "Results will appear here."))

        filter_inputs = [type_sel, genre_sel, year_from, year_to, min_score, k, popular]
        search_inputs = [query, *filter_inputs]
        similar_inputs = [title, *filter_inputs]
        search_btn.click(run_search, search_inputs, results)
        query.submit(run_search, search_inputs, results)
        similar_btn.click(run_similar, similar_inputs, results)
        title.select(run_similar, similar_inputs, results)

        gr.Markdown(
            f"""
**How it works.** Every synopsis is split into sentence windows of up to {CHUNK_CHARS} characters that overlap
by one sentence. Each window is embedded with a short header (title, format, year, genres, studio) in front,
so a passage from the middle of a synopsis still knows which show it belongs to. The vectors live in a
persistent Chroma collection (cosine distance). A query is embedded the same way, Chroma returns the nearest
chunks, and the results keep each anime's best-matching chunk. "Match" is that chunk's cosine similarity.
"More like this" averages a title's chunk vectors and searches with the result.

Data: the CC0 [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)
via the [Anime Compass](https://github.com/Seiitsu77/anime-compass) catalog; adult titles excluded.
Code: [github.com/Seiitsu77/Search-App](https://github.com/Seiitsu77/Search-App).
"""
        )
    return demo


if __name__ == "__main__":
    build_demo(load_engine()).launch(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="pink"), css=CSS)
