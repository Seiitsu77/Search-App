# 🧭 Anime Compass Search

A semantic search engine for anime. Describe a plot, a vibe or a theme in your own words
("space bounty hunters with a jazz soundtrack", "girls go camping in winter") and it finds anime
whose synopses *mean* the same thing, even when they share no keywords with your query.

Built for AIPI 510 Interactive #2 (*Build Your Own Semantic Search Engine*), following the course
starter notebook: pick a dataset → chunk + embed → store in a Chroma vector DB → search app
(Streamlit) → publish (Streamlit Community Cloud).

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
- **Model:** [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  (384-dimensional, normalized vectors), the starter notebook's model. It runs through the ONNX
  build that ships with Chroma, which uses the same mean pooling, normalization and 256-token
  limit as sentence-transformers. That avoids installing PyTorch, so the Streamlit Cloud install
  stays small.

## 3. Chroma vector DB

`chromadb/` is a persistent Chroma collection (`anime_chunks`, cosine distance). Each record holds
one chunk's vector and text, with `{anime_id, chunk}` as metadata. Per-anime display fields (title,
poster, score, genres and so on) are stored once in `data/anime_meta.json`. Repeating them as Chroma
metadata on every chunk roughly doubled the database size.

## 4. Search app (Streamlit)

- **Search:** the query is embedded with the same model, and Chroma returns the nearest chunks.
  Each anime keeps its best chunk, and that chunk is shown as the matched passage with its
  cosine similarity.
- **Filters (sidebar):** format, genres, year range and minimum MAL score. These become a Chroma
  `where` clause on `anime_id`.
- **More like this:** pick a title. The app averages that title's chunk vectors and searches
  with the result.
- **Favor well-known titles** (on by default) adds `0.05 × log-scaled membership` to the similarity,
  which breaks near-ties in favour of shows people have heard of.

## 5. Deploy on Streamlit Community Cloud (free)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and click **Create app**.
2. Pick the repository `Seiitsu77/Search-App`, the branch that contains `chromadb/`, and the main
   file `app.py`. Optionally set a custom URL such as `anime-compass-search`.
3. Click **Deploy**. The first start downloads the embedding model (~90 MB) and then serves
   queries from the committed `chromadb/`.

## Files

```
chromadb/            persistent Chroma DB (chroma.sqlite3 + HNSW index)
data/anime_meta.json one row of display/filter fields per anime
app.py               Streamlit app (loads the Chroma DB)
search_core.py       cleaning, chunking, Chroma storage, search/ranking
build_index.py       downloads the catalog, chunks, embeds, writes chromadb/
smoke_test.py        real-model checks (e.g. a "notebook that kills" query should find Death Note)
tests/               offline unit tests (tiny hashing embedder, no model download)
requirements.txt
.streamlit/config.toml
```

## Run it yourself

```bash
pip install -r requirements.txt
python build_index.py   # ~18k titles → chunks → chromadb/ (a few minutes on CPU)
python smoke_test.py    # optional sanity check
streamlit run app.py    # http://localhost:8501
```

A GitHub Action (`.github/workflows/build-index.yml`) rebuilds the index whenever the build code
changes, runs the tests and the smoke test, and commits the result. Streamlit Cloud then picks up
the new commit automatically.
