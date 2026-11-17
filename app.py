"""Anime Compass Search: a semantic search engine over 18k anime synopses.

Streamlit front end for the Chroma index built by build_index.py.
Run locally with `streamlit run app.py`; Streamlit Community Cloud uses the same file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st

from search_core import (
    CHUNK_CHARS,
    DEFAULT_DB_DIR,
    DEFAULT_META_PATH,
    MODEL_NAME,
    AnimeSearchEngine,
    Filters,
    Hit,
    minilm_embedder,
)

st.set_page_config(page_title="Anime Compass Search", page_icon="🧭", layout="wide")

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


@st.cache_resource(show_spinner="Loading the Chroma index and embedding model…")
def get_engine() -> AnimeSearchEngine:
    db_dir = Path(os.environ.get("CHROMA_DIR", DEFAULT_DB_DIR))
    meta_path = Path(os.environ.get("ANIME_META", DEFAULT_META_PATH))
    if not (db_dir / "chroma.sqlite3").exists() or not meta_path.exists():
        # Fresh checkout without the committed index: build it (downloads data, several minutes on CPU).
        import build_index

        build_index.main(["--db-dir", str(db_dir), "--meta-path", str(meta_path)])
    return AnimeSearchEngine.load(minilm_embedder(), db_dir=db_dir, meta_path=meta_path)


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|~$<>:])")


def md(text: str) -> str:
    """Escape text for Streamlit markdown (also stops $…$ LaTeX and :color[] directives)."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def members_label(members: int) -> str:
    if members >= 1_000_000:
        return f"{members / 1_000_000:.1f}M members"
    if members >= 1_000:
        return f"{members / 1_000:.0f}K members"
    return f"{members} members"


def render_hit(rank: int, hit: Hit) -> None:
    anime = hit.anime
    with st.container(border=True):
        poster, body = st.columns([1, 7], gap="medium")
        with poster:
            image = str(anime.get("image_url") or "")
            if image.startswith("https://"):
                st.image(image, width=96)
            else:
                st.markdown("### 🎬")
        with body:
            url = f"https://myanimelist.net/anime/{hit.anime_id}"
            st.markdown(f"**{rank}\\. [{md(hit.title)}]({url})**")
            meta = [anime.get("type") or ""]
            if anime.get("year"):
                meta.append(str(anime["year"]))
            if anime.get("episodes"):
                meta.append(f'{anime["episodes"]} ep' + ("s" if anime["episodes"] != 1 else ""))
            if anime.get("score"):
                meta.append(f'★ {anime["score"]:.2f}')
            if anime.get("members"):
                meta.append(members_label(int(anime["members"])))
            if anime.get("studios"):
                meta.append(", ".join(anime["studios"]))
            st.caption(md(" · ".join(m for m in meta if m)))
            if anime.get("genres"):
                st.markdown(" ".join(f":violet-badge[{md(g)}]" for g in anime["genres"][:8]))
            similarity = max(0.0, min(hit.similarity, 1.0))
            st.progress(similarity, text=f"Match {similarity:.0%}")
            st.write(hit.snippet if hit.snippet else "_No synopsis available._")


def render_hits(hits: list[Hit]) -> None:
    if not hits:
        st.info("No matches. Try a different description or loosen the filters in the sidebar.")
    for rank, hit in enumerate(hits, 1):
        render_hit(rank, hit)


def use_example() -> None:
    choice = st.session_state.get("example")
    if choice:
        st.session_state["query"] = choice
    st.session_state["example"] = None


engine = get_engine()

# -- Sidebar: filters ----------------------------------------------------------

types = engine.all_types()
years = [a["year"] for a in engine.catalog.values() if a.get("year")]
year_lo, year_hi = (min(years), max(years)) if years else (1917, 2025)

with st.sidebar:
    st.header("Filters")
    type_sel = st.multiselect("Format", types, default=[t for t in types if t not in DEFAULT_EXCLUDED_TYPES])
    genre_sel = st.multiselect("Must include genres", engine.all_genres())
    year_from, year_to = st.slider("Year", year_lo, year_hi, (year_lo, year_hi))
    min_score = st.slider("Minimum MAL score", 0.0, 9.0, 0.0, 0.5, help="0 means any score.")
    k = st.slider("Number of results", 5, 30, 10)
    popular = st.toggle(
        "Favor well-known titles",
        value=True,
        help="Adds a small bonus by MyAnimeList membership to break near-ties.",
    )

filters = Filters(
    types=type_sel if len(type_sel) < len(types) else None,
    genres=genre_sel or None,
    year_min=year_from if year_from > year_lo else None,
    year_max=year_to if year_to < year_hi else None,
    min_score=min_score or None,
)
boost = POPULARITY_WEIGHT if popular else 0.0

# -- Main ----------------------------------------------------------------------

st.title("🧭 Anime Compass Search")
st.markdown(
    "Describe a plot, a vibe or a theme in your own words. This semantic search engine finds anime whose "
    "synopses *mean* the same thing, even when they share no keywords. It searches "
    f"**{len(engine.catalog):,}** titles split into **{engine.count():,}** synopsis chunks, embedded with "
    f"`{MODEL_NAME}` and stored in a Chroma vector database."
)

search_tab, similar_tab = st.tabs(["🔎 Search", "✨ More like this"])

with search_tab:
    query = st.text_input(
        "What are you in the mood for?",
        key="query",
        placeholder="e.g. a samurai with a cross-shaped scar who has sworn never to kill again",
    )
    st.pills("Try one", EXAMPLES, key="example", on_change=use_example)
    if not type_sel:
        st.warning("Select at least one format in the sidebar.")
    elif query.strip():
        hits = engine.search(query, k=k, filters=filters, popularity_boost=boost)
        st.markdown(f"Top {len(hits)} titles for **“{md(query.strip())}”**")
        render_hits(hits)

with similar_tab:
    title = st.selectbox(
        "Pick an anime (or type any title)",
        engine.popular_titles(5000),
        index=None,
        placeholder="Start typing a title…",
        accept_new_options=True,
    )
    if title and not type_sel:
        st.warning("Select at least one format in the sidebar.")
    elif title:
        anime = engine.resolve_title(title)
        if anime is None:
            st.warning(f"Couldn't find a title matching “{md(title)}”.")
        else:
            hits = engine.similar_to(anime["anime_id"], k=k, filters=filters, popularity_boost=boost)
            st.markdown(f"Titles whose synopses read most like **{md(anime['title'])}**")
            render_hits(hits)

with st.expander("How it works"):
    st.markdown(
        f"""
1. **Dataset.** The [Anime Compass](https://github.com/Seiitsu77/anime-compass) catalog, built from the CC0
   [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)
   (MyAnimeList). Adult titles are excluded.
2. **Chunk + embed.** Each synopsis is split into sentence windows of up to {CHUNK_CHARS} characters that
   overlap by one sentence. Each window is embedded with a one-line header (title, format, year, genres,
   studio) in front, so a passage from the middle of a synopsis still knows which show it belongs to.
3. **Chroma.** The vectors live in a persistent Chroma collection (cosine distance) committed with the app.
4. **Search.** The query is embedded with the same model, and Chroma returns the nearest chunks. Each
   anime keeps its best-matching chunk, shown as the matched passage. "Match" is its cosine similarity.
   **More like this** averages a title's chunk vectors and searches with the result.

Code: [github.com/Seiitsu77/Search-App](https://github.com/Seiitsu77/Search-App)
"""
    )
