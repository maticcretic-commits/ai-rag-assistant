#!/usr/bin/env python3
"""
AI RAG Assistant — portfolio practice project.

Chat with your own documents: ingest .txt/.md/.pdf files, then ask
questions and get answers with cited sources.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # then add your OPENAI_API_KEY

Usage:
    python rag_assistant.py ingest documents/
    python rag_assistant.py ask "What is the refund policy?"

How it works:
    1. Documents are split into overlapping chunks.
    2. Each chunk is embedded with OpenAI text-embedding-3-small.
    3. Your question is embedded the same way; the most similar
       chunks (cosine similarity) are retrieved.
    4. The LLM answers using ONLY the retrieved chunks and cites them.

This is a learning project — TODO(learn) comments mark good experiments.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

INDEX_PATH = Path(".rag_index.json")

# TODO(learn): experiment with chunk sizes (400 vs 800 vs 1200 words) and
# see how answer quality changes on your sample questions.


def load_documents(folder):
    """Read .txt/.md files (and .pdf if pypdf is installed) from folder."""
    docs = []
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append({"source": str(path), "text": text})
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
                text = "\n".join(
                    page.extract_text() or "" for page in PdfReader(str(path)).pages
                )
                docs.append({"source": str(path), "text": text})
            except ImportError:
                print(f"Skipping {path}: pip install pypdf to read PDFs")
    return docs


def chunk_text(text, chunk_size=600, overlap=120):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        piece = " ".join(words[start:start + chunk_size])
        if piece.strip():
            chunks.append(piece)
        if start + chunk_size >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("The 'openai' package is missing: pip install -r requirements.txt")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY first (see .env.example).")
    return OpenAI(api_key=api_key)


def embed_texts(client, texts):
    """Return one embedding vector per text."""
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(question_embedding, chunks, embeddings, top_k=3):
    """Return the top_k chunks most similar to the question."""
    ranked = sorted(
        zip(chunks, embeddings),
        key=lambda pair: cosine_similarity(question_embedding, pair[1]),
        reverse=True,
    )
    return ranked[:top_k]


def answer_question(client, question, contexts):
    """Ask the LLM to answer using ONLY the retrieved context, with citations."""
    context_block = "\n\n".join(
        f"[{i + 1}] (source: {chunk['source']})\n{chunk['text']}"
        for i, (chunk, _score) in enumerate(contexts)
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers strictly from the "
                    "provided context. If the answer is not in the context, say "
                    "'I don't know based on the provided documents.' Cite every "
                    "factual claim with the source number in brackets, e.g. [1]."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


def cmd_ingest(args):
    docs = load_documents(args.folder)
    if not docs:
        sys.exit(f"No readable documents found in {args.folder}")
    client = get_client()
    chunks = []
    for doc in docs:
        for piece in chunk_text(doc["text"]):
            chunks.append({"source": doc["source"], "text": piece})
    print(f"Embedding {len(chunks)} chunks from {len(docs)} documents...")
    # TODO(learn): the API accepts batches — try embedding 50 chunks per call
    # and compare speed vs one call per chunk.
    embeddings = embed_texts(client, [c["text"] for c in chunks])
    INDEX_PATH.write_text(json.dumps({"chunks": chunks, "embeddings": embeddings}))
    print(f"Done. Index saved to {INDEX_PATH}")


def cmd_ask(args):
    if not INDEX_PATH.exists():
        sys.exit("No index found. Run 'ingest' first.")
    index = json.loads(INDEX_PATH.read_text())
    client = get_client()
    question_embedding = embed_texts(client, [args.question])[0]
    contexts = retrieve(question_embedding, index["chunks"], index["embeddings"])
    print("\nSources used:")
    for i, (chunk, _score) in enumerate(contexts):
        print(f"  [{i + 1}] {chunk['source']}")
    print()
    print(answer_question(client, args.question, contexts))


def main():
    parser = argparse.ArgumentParser(description="RAG assistant over your documents")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="build the search index from a folder")
    p_ingest.add_argument("folder", help="folder with .txt/.md/.pdf files")
    p_ingest.set_defaults(func=cmd_ingest)
    p_ask = sub.add_parser("ask", help="ask a question over the indexed documents")
    p_ask.add_argument("question", help="your question in quotes")
    p_ask.set_defaults(func=cmd_ask)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
