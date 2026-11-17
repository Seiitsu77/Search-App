"""Shared logic for the Anime Compass semantic search app.

Everything the index builder (build_index.py) and the Streamlit app (app.py) have
in common lives here: cleaning catalog rows, chunking synopses, writing chunks
to Chroma, and turning chunk hits back into ranked anime.

The embedding model is passed in as a plain callable (list of texts -> 2-D
array of unit vectors), so tests can swap in a tiny offline embedder.
build_index.py and app.py both use minilm_embedder(), so documents and
queries are always embedded by the same model.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "anime_chunks"
DEFAULT_DB_DIR = Path(__file__).resolve().parent / "chromadb"
# Per-anime display/filter fields live once in this file instead of being
# repeated in Chroma metadata on every chunk (which roughly doubled the index).
DEFAULT_META_PATH = Path(__file__).resolve().parent / "data" / "anime_meta.json"

# Chunking: sentence windows of at most ~CHUNK_CHARS characters, overlapping by
# one sentence. The embedder reads at most 256 word pieces; a 700-character
# window plus the one-line header comes to roughly 150-200.
CHUNK_CHARS = 700
OVERLAP_SENTENCES = 1

# Adult titles are not part of this demo.
EXCLUDED_GENRES = {"hentai", "erotica"}

Embedder = Callable[[Sequence[str]], np.ndarray]

_SOURCE_NOTE = re.compile(r"\s*[\(\[]\s*(source|written by)[^)\]]*[\)\]]\s*", re.IGNORECASE)
_NO_SYNOPSIS = re.compile(r"^no (synopsis|summary)( information)? has been added", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])")


# --------------------------------------------------------------------------
# Cleaning and chunking
# --------------------------------------------------------------------------


def clean_synopsis(text: Any) -> str:
    """Drop source credits, placeholders and stray whitespace from a synopsis."""
    synopsis = str(text or "").replace("\r", "\n").strip()
    if not synopsis or _NO_SYNOPSIS.match(synopsis):
        return ""
    synopsis = _SOURCE_NOTE.sub(" ", synopsis)
    synopsis = re.sub(r"[ \t]+", " ", synopsis)
    synopsis = re.sub(r"\s*\n\s*", "\n", synopsis)
    return synopsis.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation and paragraph breaks."""
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            sentences.extend(part.strip() for part in _SENTENCE_END.split(paragraph) if part.strip())
    return sentences


def chunk_text(text: str, max_chars: int = CHUNK_CHARS, overlap: int = OVERLAP_SENTENCES) -> list[str]:
    """Group sentences into windows of at most ``max_chars`` that overlap by ``overlap`` sentences.

    A single sentence longer than ``max_chars`` becomes its own chunk rather
    than being cut mid-sentence; the model truncates anything past its limit.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        end = start
        length = 0
        while end < len(sentences) and (end == start or length + 1 + len(sentences[end]) <= max_chars):
            length += len(sentences[end]) + (1 if end > start else 0)
            end += 1
        chunks.append(" ".join(sentences[start:end]))
        if end >= len(sentences):
            break
        # Step forward, keeping `overlap` sentences of context, but always advance.
        start = max(start + 1, end - overlap)
    return chunks


def is_excluded(item: Mapping[str, Any]) -> bool:
    genres = {str(genre).lower() for genre in item.get("genres") or []}
    return bool(genres & EXCLUDED_GENRES) or str(item.get("rating") or "").lower().startswith("rx")


def _year(item: Mapping[str, Any]) -> int:
    try:
        return int(item.get("start_year") or 0)
    except (TypeError, ValueError):
        return 0


def _number(value: Any, cast: Callable[[Any], Any], default: Any) -> Any:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return cast(value)
    except (TypeError, ValueError):
        return default


def anime_header(item: Mapping[str, Any]) -> str:
    """One line of context prepended to every chunk so each chunk knows its show."""
    title = str(item.get("title") or "").strip()
    bits = [part for part in (str(item.get("type") or "").strip(), str(_year(item) or "")) if part]
    header = f"{title} ({', '.join(bits)})." if bits else f"{title}."
    genres = [str(g) for g in item.get("genres") or [] if g]
    if genres:
        header += f" Genres: {', '.join(genres)}."
    studios = [str(s) for s in item.get("studios") or [] if s]
    if studios:
        header += f" Studio: {', '.join(studios[:3])}."
    return header


def anime_record(item: Mapping[str, Any]) -> dict[str, Any]:
    """Per-anime display and filter fields, kept once in a JSON file beside the index."""
    anime_id = int(item["id"])
    return {
        "anime_id": anime_id,
        "title": str(item.get("title") or ""),
        "type": str(item.get("type") or "Unknown"),
        "year": _year(item),
        "episodes": _number(item.get("episodes"), int, 0),
        "score": round(_number(item.get("score"), float, 0.0), 2),
        "members": _number(item.get("members"), int, 0),
        "genres": [str(g) for g in item.get("genres") or [] if g],
        "studios": [str(s) for s in (item.get("studios") or [])[:3] if s],
        "image_url": str(item.get("image_url") or ""),
    }


@dataclass
class Chunk:
    id: str
    document: str  # what Chroma stores and the app shows: one synopsis window
    embed_text: str  # what gets embedded: "<header>\n<synopsis window>"
    metadata: dict[str, Any] = field(default_factory=dict)


def build_chunks(
    catalog: Iterable[Mapping[str, Any]], max_chars: int = CHUNK_CHARS
) -> tuple[list[Chunk], list[dict[str, Any]]]:
    """Turn catalog rows into synopsis chunks plus one record per anime.

    Each chunk is embedded with a one-line header (title, type, year, genres,
    studio) in front of its synopsis window, so a window from the middle of a
    synopsis still embeds as part of the right show. Chroma stores only the
    window itself; the header can be rebuilt from the anime record, and storing
    it on every chunk would add ~20% to the database for no new information.
    Titles with no synopsis get one header-only chunk, so they stay findable by
    title, genre and studio.
    """
    chunks: list[Chunk] = []
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in catalog:
        if item.get("id") is None or not item.get("title") or is_excluded(item):
            continue
        anime_id = int(item["id"])
        if anime_id in seen:
            continue
        seen.add(anime_id)

        header = anime_header(item)
        records.append(anime_record(item))
        pieces = chunk_text(clean_synopsis(item.get("synopsis")), max_chars=max_chars) or [""]
        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{anime_id}-{index}",
                    document=piece,
                    embed_text=f"{header}\n{piece}" if piece else header,
                    metadata={"anime_id": anime_id, "chunk": index},
                )
            )
    return chunks, records


# --------------------------------------------------------------------------
# Chroma storage
# --------------------------------------------------------------------------


def open_client(db_dir: Path | str = DEFAULT_DB_DIR):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(db_dir), settings=Settings(anonymized_telemetry=False))


def write_index(
    chunks: Sequence[Chunk],
    records: Sequence[Mapping[str, Any]],
    embedder: Embedder,
    db_dir: Path | str = DEFAULT_DB_DIR,
    meta_path: Path | str = DEFAULT_META_PATH,
    batch_size: int = 256,
    progress: Callable[[int, int], None] | None = None,
):
    """Embed chunks into a fresh persistent Chroma collection and save the anime records."""
    client = open_client(db_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - collection did not exist yet
        pass
    collection = client.create_collection(
        COLLECTION_NAME,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"model": MODEL_NAME, "chunk_chars": CHUNK_CHARS},
    )
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = np.asarray(embedder([c.embed_text for c in batch]), dtype=np.float32)
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.document for c in batch],
            metadatas=[c.metadata for c in batch],
            embeddings=vectors.tolist(),
        )
        if progress:
            progress(min(start + batch_size, len(chunks)), len(chunks))

    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(list(records), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return collection


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


@dataclass
class Hit:
    anime: dict[str, Any]
    similarity: float
    score: float
    snippet: str

    @property
    def anime_id(self) -> int:
        return int(self.anime["anime_id"])

    @property
    def title(self) -> str:
        return str(self.anime["title"])


@dataclass
class Filters:
    types: Sequence[str] | None = None
    genres: Sequence[str] | None = None
    year_min: int | None = None
    year_max: int | None = None
    min_score: float | None = None

    def active(self) -> bool:
        return bool(self.types or self.genres or self.year_min or self.year_max or self.min_score)

    def accepts(self, anime: Mapping[str, Any]) -> bool:
        if self.types and anime["type"] not in self.types:
            return False
        if self.genres and not set(self.genres) <= set(anime["genres"]):
            return False
        year = int(anime.get("year") or 0)
        if self.year_min and year < int(self.year_min):
            return False
        if self.year_max and (not year or year > int(self.year_max)):
            return False
        if self.min_score and float(anime.get("score") or 0) < float(self.min_score):
            return False
        return True


class AnimeSearchEngine:
    """Query the Chroma chunk collection and rank whole anime."""

    def __init__(self, collection, catalog: Sequence[Mapping[str, Any]], embedder: Embedder):
        self.collection = collection
        self.embedder = embedder
        self.catalog: dict[int, dict[str, Any]] = {int(a["anime_id"]): dict(a) for a in catalog}
        self._by_popularity = sorted(self.catalog.values(), key=lambda a: -int(a.get("members") or 0))

    @classmethod
    def load(
        cls,
        embedder: Embedder,
        db_dir: Path | str = DEFAULT_DB_DIR,
        meta_path: Path | str = DEFAULT_META_PATH,
    ) -> "AnimeSearchEngine":
        collection = open_client(db_dir).get_collection(COLLECTION_NAME, embedding_function=None)
        catalog = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        return cls(collection, catalog, embedder)

    # -- catalog helpers ---------------------------------------------------

    def count(self) -> int:
        return self.collection.count()

    def all_genres(self) -> list[str]:
        return sorted({genre for anime in self.catalog.values() for genre in anime["genres"]})

    def all_types(self) -> list[str]:
        counts: dict[str, int] = {}
        for anime in self.catalog.values():
            counts[anime["type"]] = counts.get(anime["type"], 0) + 1
        return sorted(counts, key=lambda t: -counts[t])

    def popular_titles(self, n: int = 2000) -> list[str]:
        return [str(a["title"]) for a in self._by_popularity[:n]]

    def resolve_title(self, text: str) -> dict[str, Any] | None:
        """Find a title by exact, prefix, substring or fuzzy match (most popular wins ties)."""
        query = text.strip().lower()
        if not query:
            return None
        for test in (
            lambda t: t == query,
            lambda t: t.startswith(query),
            lambda t: query in t,
        ):
            for anime in self._by_popularity:
                if test(str(anime["title"]).lower()):
                    return anime
        titles = {str(a["title"]).lower(): a for a in reversed(self._by_popularity)}
        close = get_close_matches(query, list(titles), n=1, cutoff=0.6)
        return titles[close[0]] if close else None

    # -- ranking -----------------------------------------------------------

    def _where(self, filters: Filters, exclude_ids: set[int]) -> tuple[dict[str, Any] | None, int]:
        """Translate filters into a Chroma where-clause on anime_id.

        Returns the clause (None when nothing is excluded) and how many titles pass.
        Uses whichever of $in / $nin gives the shorter list.
        """
        if not filters.active() and not exclude_ids:
            return None, len(self.catalog)
        allowed = [i for i, a in self.catalog.items() if i not in exclude_ids and filters.accepts(a)]
        if len(allowed) == len(self.catalog):
            return None, len(allowed)
        if not allowed:
            return {"anime_id": {"$in": [-1]}}, 0
        if len(allowed) <= len(self.catalog) - len(allowed):
            return ({"anime_id": {"$in": allowed}} if len(allowed) > 1 else {"anime_id": allowed[0]}), len(allowed)
        allowed_set = set(allowed)
        blocked = [i for i in self.catalog if i not in allowed_set]
        return ({"anime_id": {"$nin": blocked}} if len(blocked) > 1 else {"anime_id": {"$ne": blocked[0]}}), len(allowed)

    def _rank(
        self,
        query_vector: np.ndarray,
        k: int,
        filters: Filters | None,
        popularity_boost: float,
        exclude_ids: set[int] | None = None,
    ) -> list[Hit]:
        filters = filters or Filters()
        exclude_ids = exclude_ids or set()
        where, n_allowed = self._where(filters, exclude_ids)
        if n_allowed == 0:
            return []
        # Several chunks can belong to one anime, so ask for a generous pool of
        # chunks and keep each anime's best one.
        pool = min(max(k * 6, 60), self.collection.count())
        result = self.collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
            n_results=pool,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        best: dict[int, Hit] = {}
        for document, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        ):
            anime = self.catalog.get(int(metadata["anime_id"]))
            if anime is None:
                continue
            similarity = 1.0 - float(distance)
            # log10(members) runs from 0 to ~6.6 on this catalog; scale to 0..1.
            prior = min(math.log10(int(anime.get("members") or 0) + 1) / 6.5, 1.0)
            score = similarity + popularity_boost * prior
            current = best.get(anime["anime_id"])
            if current is None or score > current.score:
                best[anime["anime_id"]] = Hit(anime, similarity, score, str(document or ""))
        return sorted(best.values(), key=lambda hit: -hit.score)[:k]

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Filters | None = None,
        popularity_boost: float = 0.0,
    ) -> list[Hit]:
        query = query.strip()
        if not query:
            return []
        vector = np.asarray(self.embedder([query]), dtype=np.float32)[0]
        return self._rank(vector, k, filters, popularity_boost)

    def similar_to(
        self,
        anime_id: int,
        k: int = 10,
        filters: Filters | None = None,
        popularity_boost: float = 0.0,
    ) -> list[Hit]:
        """Titles whose chunks sit closest to the mean of this title's chunk embeddings."""
        rows = self.collection.get(where={"anime_id": int(anime_id)}, include=["embeddings"])
        embeddings = rows.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return []
        centroid = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            return []
        return self._rank(centroid / norm, k, filters, popularity_boost, exclude_ids={int(anime_id)})


def minilm_embedder() -> Embedder:
    """all-MiniLM-L6-v2 through Chroma's bundled ONNX runtime.

    Same model, pooling (attention-masked mean), normalization and 256-token
    limit as sentence-transformers, without installing PyTorch. That keeps the
    Streamlit Cloud install small and fast. The model (~90 MB) downloads once
    to ~/.cache/chroma on first use.
    """
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    model = ONNXMiniLM_L6_V2()

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(model(list(texts)), dtype=np.float32)

    return embed
