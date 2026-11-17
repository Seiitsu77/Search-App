---
title: Anime Compass Search
emoji: 🧭
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Semantic search over 18k anime synopses with Chroma
---

# 🧭 Anime Compass Search

A semantic search engine for anime. Describe a plot, a vibe or a theme in your own words
("space bounty hunters with a jazz soundtrack", "girls go camping in winter") and it finds anime
whose synopses *mean* the same thing, even when they share no keywords with your query.

Built for AIPI 510 Interactive #2 (*Build Your Own Semantic Search Engine*), following the course
starter notebook: pick a dataset → chunk + embed → store in a Chroma vector DB → search app →
publish on Hugging Face Spaces.

**Live app:** https://huggingface.co/spaces/Seiitsu/anime-compass-search

## 1. Dataset

The catalog from my [Anime Compass](https://github.com/Seiitsu77/anime-compass) recommender, which is
built from the CC0 [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)
(MyAnimeList data). It is downloaded from the
[`Seiitsu/anime-compass-data`](https://huggingface.co/datasets/Seiitsu/anime-compass-data) dataset on
Hugging Face: about 18k titles with synopsis, genres, studio, format, year, score and membership
counts. Adult titles are excluded.

## 2. Chunking + embedding

- **Cleaning:** source credits such as `[Written by MAL Rewrite]` and "no synopsis yet" placeholders are removed.
- **Chunking:** each synopsis is split into sentences and grouped into windows of at most 700
  characters. Consecutive windows overlap by one sentence, so an idea that spans a boundary
  survives in at least one chunk.
- **Contextual header:** before embedding, each window gets a one-line header such as
  `Cowboy Bebop (TV, 1998). Genres: Action, Sci-Fi. Studio: Sunrise.` A passage from the middle
  of a synopsis therefore still embeds as part of the right show, and titles without a synopsis
  stay findable by title, genre and studio.
- **Model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  (384-dimensional, normalized vectors).

## 3. Chroma vector DB

`chromadb/` is a persistent Chroma collection (`anime_chunks`, cosine distance). Each record holds
one chunk's vector and text, with `{anime_id, chunk}` as metadata. Per-anime display fields (title,
poster, score, genres and so on) are stored once in `data/anime_meta.json`. Repeating them as Chroma
metadata on every chunk roughly doubled the database size.

## 4. Search app (Gradio)

- **Search:** the query is embedded with the same model, and Chroma returns the nearest chunks.
  Each anime keeps its best chunk, and that chunk is shown as the matched passage with its
  cosine similarity.
- **Filters:** format, genres, year range and minimum MAL score. These become a Chroma `where`
  clause on `anime_id`.
- **More like this:** pick a title. The app averages that title's chunk vectors and searches
  with the result.
- **Favor well-known titles** (on by default) adds `0.05 × log-scaled membership` to the similarity,
  which breaks near-ties in favour of shows people have heard of.

## 5. Hugging Face Space

The Space runs `app.py` with the prebuilt `chromadb/` on free CPU hardware.

## Files

```
chromadb/            persistent Chroma DB (chroma.sqlite3 + HNSW index)
data/anime_meta.json one row of display/filter fields per anime
app.py               Gradio app (loads the Chroma DB)
search_core.py       cleaning, chunking, Chroma storage, search/ranking
build_index.py       downloads the catalog, chunks, embeds, writes chromadb/
smoke_test.py        real-model checks (e.g. a "notebook that kills" query should find Death Note)
deploy_space.py      uploads the app + index to the Hugging Face Space
tests/               offline unit tests (tiny hashing embedder, no model download)
requirements.txt
```

## Run it yourself

```bash
pip install -r requirements.txt
python build_index.py   # ~18k titles → chunks → chromadb/ (a few minutes on CPU)
python smoke_test.py    # optional sanity check
python app.py           # http://127.0.0.1:7860
```

A GitHub Action (`.github/workflows/build-index.yml`) rebuilds the index whenever the build code
changes, runs the tests and the smoke test, and commits the result. When the repository has an
`HF_TOKEN` secret, the Action also redeploys the Space.

Deploy manually with `HF_TOKEN=<write token> python deploy_space.py --space <user>/anime-compass-search`.
