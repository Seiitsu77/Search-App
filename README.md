# 🧭 Anime Compass Search

A semantic search engine for anime. Describe a plot, a vibe or a theme in your own words
("space cowboys and jazz music", "a mother raises two children who can turn into wolves") and it finds anime
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
- **Model:** [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)
  (384-dimensional, CLS pooling, normalized). It is trained for retrieval: queries get the
  instruction `Represent this sentence for searching relevant passages: ` and documents are
  embedded as-is. It replaced the starter notebook's `all-MiniLM-L6-v2` after scoring better
  on the evaluation below. The official ONNX export runs on onnxruntime, so the app does not
  need PyTorch and the Streamlit Cloud install stays small. The index records which model built
  it, and the app always embeds queries with that same model.

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
3. Click **Deploy**. The first start downloads the embedding model (~130 MB) and then serves
   queries from the committed `chromadb/`.

## Evaluation

`evaluate.py` runs 38 queries with known answers. 22 are phrased like a synopsis ("orphans
discover they are being raised as food for demons") and 16 like a viewer would put it ("space
cowboys and jazz music"). It searches with the app's default settings and reports hit@k (the
right show is in the top k) and MRR@10. Every index rebuild scores the old index and the new
one, and the new index is committed only if MRR@10 does not drop. The latest scores are in
`data/evaluation.json`.

| Model | hit@1 | hit@5 | hit@10 | MRR@10 | MRR@10 synopsis-style | MRR@10 fan-style |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (starter notebook) | 0.447 | 0.632 | 0.658 | 0.517 | **0.587** | 0.420 |
| bge-small-en-v1.5 without query instruction | 0.500 | 0.605 | 0.658 | 0.549 | 0.532 | 0.573 |
| **bge-small-en-v1.5 (deployed)** | **0.526** | **0.658** | **0.684** | **0.579** | 0.570 | **0.591** |

BGE's gain comes almost entirely from fan-style queries (MRR 0.42 → 0.59). On synopsis-style
queries the two models are about even. Some shows are still missed by both models, for
example Kimetsu no Yaiba, AnoHana, Evangelion and Yakusoku no Neverland. Median query time is
about 0.1 s on a CPU.

## Files

```
chromadb/            persistent Chroma DB (chroma.sqlite3 + HNSW index)
data/anime_meta.json one row of display/filter fields per anime
data/evaluation.json latest evaluation scores
app.py               Streamlit app (loads the Chroma DB)
search_core.py       cleaning, chunking, Chroma storage, search/ranking
build_index.py       downloads the catalog, chunks, embeds, writes chromadb/
evaluate.py          search-quality evaluation (38 queries with known answers)
tests/               offline unit tests (tiny hashing embedder, no model download)
requirements.txt
.streamlit/config.toml
```

## Run it yourself

```bash
pip install -r requirements.txt
python build_index.py   # ~18k titles → chunks → chromadb/ (a few minutes on CPU)
python evaluate.py      # optional: score search quality
streamlit run app.py    # http://localhost:8501
```

A GitHub Action (`.github/workflows/build-index.yml`) rebuilds the index whenever the build code
changes, runs the tests, scores old and new index, and commits the result. Streamlit Cloud then picks up
the new commit automatically.
