"""
MILESTONE 2 - Searchable Knowledge Base (ChromaDB)
---------------------------------------------------
Store chunks as embeddings, then search them by meaning.
"""
import os

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

_embedder = None


def embed(texts):
    """Text -> vectors. The model loads once, the first time it is needed."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder.encode(texts).tolist()


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection("safety_docs")


def add_document(chunks, source):
    col = get_collection()

    # remove old chunks of the same file (safe to upload twice)
    old = col.get(where={"source": source})["ids"]
    if old:
        col.delete(ids=old)

    col.add(
        ids=[f"{source}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
        metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


def search(question, top_k=3):
    col = get_collection()
    if col.count() == 0:
        return []

    res = col.query(query_embeddings=embed([question]), n_results=min(top_k, col.count()))

    results = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({"text": text, "source": meta["source"], "chunk": meta["chunk"],
                        "distance": round(dist, 3)})
    return results


def list_documents():
    """{file name: number of chunks}"""
    counts = {}
    for meta in get_collection().get()["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return counts


def total_chunks():
    return get_collection().count()


if __name__ == "__main__":
    from documents import process_file

    add_document(process_file("data/sample_safety_manual.txt"), "sample_safety_manual.txt")
    for r in search("Do I need a helmet?"):
        print(r["distance"], r["text"][:80])
