# AI RAG Assistant

A portfolio practice project: a **Retrieval-Augmented Generation (RAG) assistant**
that answers questions over your own documents — with cited sources.

This is the project type at the top of the freelance pay scale for AI automation
work: custom LLM/RAG assistants typically go for **$1,500–$5,000 per project**
on platforms like Upwork, well above basic chatbot or workflow gigs.

> Built as a learning/demo project — not client work.

## How it works

1. **Ingest** — `.txt`, `.md`, and `.pdf` files are split into overlapping chunks.
2. **Embed** — each chunk becomes a vector via OpenAI `text-embedding-3-small`.
3. **Retrieve** — your question is embedded the same way; the most similar
   chunks are found with cosine similarity.
4. **Answer** — the LLM answers *only* from the retrieved chunks and cites
   every claim, e.g. `[1]`. If the answer isn't in your documents, it says so
   instead of hallucinating.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

## Usage

```bash
# Build the index from the sample docs (or point at your own folder)
python rag_assistant.py ingest documents/

# Ask questions
python rag_assistant.py ask "What is the refund policy?"
python rag_assistant.py ask "How much does the Business plan cost?"
```

Example output:

```
Sources used:
  [1] documents/sample-faq.txt
  [2] documents/sample-faq.txt

We offer a 30-day money-back guarantee on all paid plans [1]. ...
```

## Run the tests (no API key needed)

```bash
python tests/test_rag.py
```

## What I'd build next (learning roadmap)

- [ ] Swap the JSON index for a real vector DB (Chroma / Pinecone)
- [ ] Add a web UI (Streamlit or a simple chat page)
- [ ] Support .docx and web-page ingestion
- [ ] Add conversation memory for multi-turn chat
- [ ] Evaluate answer quality with a small test-question set

## ❤️ Support My Work

> If you find this project useful, please consider supporting my work with a Bitcoin donation:
>
> **₿ `BC1Q6Q75K8ZJXVW7W02LMDPRPY6XX6QK4LZZ2RMVAY`**
