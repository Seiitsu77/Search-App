from __future__ import annotations

from search_core import (
    AnimeSearchEngine,
    Filters,
    build_chunks,
    chunk_text,
    clean_synopsis,
    split_sentences,
    write_index,
)

CATALOG = [
    {
        "id": 16498,
        "title": "Shingeki no Kyojin",
        "type": "TV",
        "start_year": 2013,
        "score": 8.5,
        "members": 3500000,
        "genres": ["Action", "Drama", "Shounen"],
        "studios": ["Wit Studio"],
        "synopsis": (
            "Centuries ago, mankind was slaughtered to near extinction by monstrous humanoid creatures "
            "called Titans. Humanity now lives behind enormous walls. Eren Yeager vows to wipe the Titans "
            "out after one breaches the wall.\n\n[Written by MAL Rewrite]"
        ),
    },
    {
        "id": 1535,
        "title": "Death Note",
        "type": "TV",
        "start_year": 2006,
        "score": 8.6,
        "members": 3700000,
        "genres": ["Supernatural", "Suspense", "Shounen"],
        "studios": ["Madhouse"],
        "synopsis": "A shinigami drops a notebook. Whoever's name is written in the notebook dies. Light Yagami finds it.",
    },
    {
        "id": 32281,
        "title": "Kimi no Na wa.",
        "type": "Movie",
        "start_year": 2016,
        "score": 8.8,
        "members": 2700000,
        "genres": ["Drama", "Romance", "Supernatural"],
        "studios": ["CoMix Wave Films"],
        "synopsis": "Mitsuha and Taki mysteriously swap bodies while a comet approaches Earth.",
    },
    {
        "id": 99999,
        "title": "Adult Title",
        "type": "OVA",
        "genres": ["Hentai"],
        "synopsis": "Should never be indexed.",
    },
    {"id": 5, "title": "No Synopsis Show", "type": "TV", "genres": ["Comedy"], "studios": ["P.A. Works"], "synopsis": ""},
]


def test_clean_synopsis_removes_credits_and_placeholders():
    assert clean_synopsis("Text here. (Source: ANN)") == "Text here."
    assert clean_synopsis("Story.\n\n[Written by MAL Rewrite]") == "Story."
    assert clean_synopsis("No synopsis information has been added to this title.") == ""
    assert clean_synopsis(None) == ""


def test_split_sentences_handles_paragraphs_and_quotes():
    text = 'He said "Run!" Then he ran. Dr. Who? Maybe.\nNew paragraph here.'
    assert split_sentences(text)[-1] == "New paragraph here."
    assert split_sentences("") == []


def test_chunk_text_respects_limit_and_overlaps():
    sentences = [f"Sentence number {i} is here." for i in range(20)]
    chunks = chunk_text(" ".join(sentences), max_chars=100, overlap=1)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    # Each chunk starts with the last sentence of the previous one.
    for previous, current in zip(chunks, chunks[1:]):
        assert current.startswith(split_sentences(previous)[-1])
    # Every sentence survives.
    joined = " ".join(chunks)
    assert all(sentence in joined for sentence in sentences)


def test_chunk_text_keeps_overlong_sentence_whole():
    long_sentence = "word " * 200 + "end."
    assert chunk_text(long_sentence.strip(), max_chars=50) == [long_sentence.strip()]


def test_build_chunks_excludes_adult_and_keeps_empty_synopsis():
    chunks, records = build_chunks(CATALOG)
    ids = {chunk.metadata["anime_id"] for chunk in chunks}
    assert ids == {record["anime_id"] for record in records}
    assert 99999 not in ids
    assert 5 in ids
    first = next(chunk for chunk in chunks if chunk.metadata["anime_id"] == 16498)
    assert first.embed_text.startswith(
        "Shingeki no Kyojin (TV, 2013). Genres: Action, Drama, Shounen. Studio: Wit Studio.\nCenturies ago"
    )
    assert first.document.startswith("Centuries ago")
    assert "MAL Rewrite" not in first.embed_text
    assert first.metadata == {"anime_id": 16498, "chunk": 0}
    empty = next(chunk for chunk in chunks if chunk.metadata["anime_id"] == 5)
    assert empty.document == ""
    assert "P.A. Works" in empty.embed_text
    record = next(r for r in records if r["anime_id"] == 5)
    assert record["genres"] == ["Comedy"] and record["score"] == 0.0 and record["year"] == 0


def test_end_to_end_search_and_filters(tmp_path, embedder):
    chunks, records = build_chunks(CATALOG, max_chars=120)
    assert len(chunks) > 4  # the Titan synopsis spans several chunks
    write_index(chunks, records, embedder, db_dir=tmp_path / "db", meta_path=tmp_path / "meta.json", batch_size=2)
    engine = AnimeSearchEngine.load(embedder, db_dir=tmp_path / "db", meta_path=tmp_path / "meta.json")

    assert engine.count() == len(chunks)
    assert set(engine.catalog) == {16498, 1535, 32281, 5}

    hits = engine.search("titans walls humanity", k=3)
    assert hits[0].title == "Shingeki no Kyojin"
    # One hit per anime even though several chunks match.
    assert len({hit.anime_id for hit in hits}) == len(hits)
    assert "MAL Rewrite" not in hits[0].snippet

    assert engine.search("notebook name dies", k=1)[0].title == "Death Note"
    assert engine.search("   ") == []

    movies = engine.search("notebook name dies", k=5, filters=Filters(types=["Movie"]))
    assert [hit.title for hit in movies] == ["Kimi no Na wa."]
    romance = engine.search("titans", k=5, filters=Filters(genres=["Romance"]))
    assert [hit.title for hit in romance] == ["Kimi no Na wa."]
    recent = engine.search("titans", k=5, filters=Filters(year_min=2010, min_score=8.6))
    assert [hit.title for hit in recent] == ["Kimi no Na wa."]
    old = engine.search("titans", k=5, filters=Filters(year_max=2010))
    assert [hit.title for hit in old] == ["Death Note"]  # unknown years are excluded by a max year
    assert engine.search("titans", k=5, filters=Filters(genres=["Romance", "Action"])) == []


def test_similar_and_title_resolution(tmp_path, embedder):
    write_index(*build_chunks(CATALOG), embedder, db_dir=tmp_path / "db", meta_path=tmp_path / "meta.json")
    engine = AnimeSearchEngine.load(embedder, db_dir=tmp_path / "db", meta_path=tmp_path / "meta.json")

    assert engine.resolve_title("death note")["anime_id"] == 1535
    assert engine.resolve_title("shingeki")["anime_id"] == 16498
    assert engine.resolve_title("kimi no na wa")["anime_id"] == 32281
    assert engine.resolve_title("deth note")["anime_id"] == 1535
    assert engine.resolve_title("zzzzzz") is None

    similar = engine.similar_to(1535, k=5)
    assert similar and all(hit.anime_id != 1535 for hit in similar)
    assert {hit.anime_id for hit in engine.similar_to(1535, k=5, filters=Filters(types=["TV"]))} == {16498, 5}
    assert engine.popular_titles(2) == ["Death Note", "Shingeki no Kyojin"]
    assert "Action" in engine.all_genres() and "TV" in engine.all_types()


# -- embedding models -----------------------------------------------------------

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import search_core  # noqa: E402
from search_core import OnnxClsEncoder, query_embedder  # noqa: E402


class _Named:
    def __init__(self, name):
        self.name = name


class StubSession:
    """Mimics onnxruntime.InferenceSession: CLS row = [real token count, 1, 0], other rows noise."""

    def __init__(self):
        self.feeds = []

    def get_inputs(self):
        return [_Named("input_ids"), _Named("attention_mask"), _Named("token_type_ids")]

    def get_outputs(self):
        return [_Named("last_hidden_state"), _Named("pooler_output")]

    def run(self, names, feeds):
        assert names == ["last_hidden_state"]
        self.feeds.append(feeds)
        batch, seq = feeds["input_ids"].shape
        hidden = np.full((batch, seq, 3), 7.0, dtype=np.float32)
        hidden[:, 0, 0] = feeds["attention_mask"].sum(axis=1)
        hidden[:, 0, 1] = 1.0
        hidden[:, 0, 2] = 0.0
        return [hidden]


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    vocab = {"[PAD]": 0, "[CLS]": 1, "[UNK]": 2, "a": 3, "b": 4, "c": 5}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A", special_tokens=[("[CLS]", 1)])
    return tokenizer


def test_onnx_cls_encoder_pools_normalizes_and_keeps_order():
    session = StubSession()
    encoder = OnnxClsEncoder(session, tiny_tokenizer(), batch_size=2)
    texts = ["a b c a b", "a", "a b c", "b c"]  # 5, 1, 3, 2 words
    vectors = encoder(texts)

    assert vectors.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)
    # Row i must come from text i: the first component encodes its token count (+1 for CLS).
    for text, vector in zip(texts, vectors):
        expected = np.array([len(text.split()) + 1, 1.0, 0.0])
        np.testing.assert_allclose(vector, expected / np.linalg.norm(expected), rtol=1e-6)
    # Batched by length (2 batches of 2), padded within a batch, token types all zero.
    assert len(session.feeds) == 2
    for feeds in session.feeds:
        assert feeds["input_ids"].shape == feeds["attention_mask"].shape
        assert not feeds["token_type_ids"].any()
    assert sorted(feeds["input_ids"].shape[1] for feeds in session.feeds) == [3, 6]


def test_query_embedder_prefixes_queries_only(monkeypatch):
    seen = []

    def fake_encoder(texts):
        seen.extend(texts)
        return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setitem(search_core._ENCODERS, "BAAI/bge-small-en-v1.5", fake_encoder)
    query_embedder("BAAI/bge-small-en-v1.5")(["cozy camping"])
    query_embedder("BAAI/bge-small-en-v1.5", prefix="")(["no prefix"])
    search_core.document_embedder("BAAI/bge-small-en-v1.5")(["a document"])
    assert seen == [
        "Represent this sentence for searching relevant passages: cozy camping",
        "no prefix",
        "a document",
    ]
    with pytest.raises(ValueError):
        search_core.document_embedder("not-a-model")


def test_index_records_its_model(tmp_path, embedder):
    write_index(
        *build_chunks(CATALOG), embedder, db_dir=tmp_path / "db", meta_path=tmp_path / "meta.json", model_name="BAAI/bge-small-en-v1.5"
    )
    engine = AnimeSearchEngine.load(embedder, db_dir=tmp_path / "db", meta_path=tmp_path / "meta.json")
    assert engine.model_name == "BAAI/bge-small-en-v1.5"


def test_evaluation_scoring():
    from evaluate import CASES, first_hit_rank, summarize

    assert first_hit_rank(["Haikyuu!! Movie", "x"], ["Haikyuu"]) == 1
    assert first_hit_rank(["x", "Fate/Zero"], ["Fate/"]) == 2
    assert first_hit_rank(["x"], ["Death Note"]) is None
    scores = summarize([1, 3, None, 10])
    assert scores == {"hit@1": 0.25, "hit@5": 0.5, "hit@10": 0.75, "mrr@10": pytest.approx((1 + 1 / 3 + 0.1) / 4)}
    assert len({query for _, query, _ in CASES}) == len(CASES)
