"""Offline tests for the RAG assistant (no API key needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_assistant import chunk_text, cosine_similarity, retrieve


def test_chunking_covers_all_words():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    rejoined = " ".join(chunks)
    for i in range(1000):
        assert f"word{i}" in rejoined


def test_chunk_overlap():
    chunks = chunk_text("a b c d e f g h", chunk_size=4, overlap=2)
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_cosine_identical_is_one():
    v = [0.1, 0.5, 0.9]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-9


def test_retrieve_picks_most_similar():
    chunks = [{"source": "s", "text": t} for t in ("cats", "dogs", "cats and kittens")]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    top = retrieve([0.9, 0.1], chunks, embeddings, top_k=1)
    assert top[0][0]["text"] == "cats and kittens"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
