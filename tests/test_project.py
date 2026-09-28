"""
Simple tests - run with:  pytest -q
Heavy parts (embedding model, Gemini) are replaced with small fakes,
so tests are fast, free and work offline.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb  # noqa: E402

import agent  # noqa: E402
import knowledge_base as kb  # noqa: E402
import report  # noqa: E402
import storage  # noqa: E402
from documents import clean_text, make_chunks  # noqa: E402
from vision import is_violation, summarize  # noqa: E402


def fake_embed(texts):
    """Tiny fake embedding: counts a few keywords."""
    words = ["helmet", "hardhat", "fire", "vest", "gloves", "report"]
    return [[t.lower().count(w) + 0.01 for w in words] for t in texts]


def test_violation_names():
    assert is_violation("no_helmet") and is_violation("NO-Hardhat")
    assert not is_violation("helmet") and not is_violation("Person") and not is_violation("none")


def test_summarize():
    dets = [{"class": "helmet", "violation": False}, {"class": "no_gloves", "violation": True}]
    s = summarize(dets, "site.jpg")
    assert s["violations"] == ["no_gloves"] and not s["compliant"]
    assert "no_gloves" in s["text"]


def test_documents():
    assert "Procedures" in clean_text("Safety Proce-\ndures\n\n 12 \n")
    chunks = make_chunks("Rule one.\n\nRule two.\n\n" + "x" * 2000)
    assert chunks[0] == "Rule one." and len(chunks) > 3


def test_knowledge_base(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "DB_PATH", str(tmp_path))
    monkeypatch.setattr(kb, "embed", fake_embed)
    kb.add_document(["Wear a helmet at all times.", "Fire drills every 3 months."], "manual.txt")
    kb.add_document(["Wear a helmet at all times.", "Fire drills every 3 months."], "manual.txt")  # twice = OK
    assert kb.total_chunks() == 2
    assert "helmet" in kb.search("helmet rule", top_k=1)[0]["text"]


def test_storage_and_kpis(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SCANS_FILE", str(tmp_path / "scans.csv"))
    monkeypatch.setattr(storage, "FEEDBACK_FILE", str(tmp_path / "feedback.csv"))
    storage.log_scan("a.jpg", {"violations": ["no_helmet"], "compliant": False})
    storage.log_scan("b.jpg", {"violations": [], "compliant": True})
    for liked in (1, 1, 1, 0):
        storage.log_feedback("q", liked)

    scans = storage.load_scans()
    k = storage.kpis(scans)
    assert k == {"scans": 2, "compliance": 50.0, "violations": 1, "satisfaction": 75.0}
    assert storage.violation_counts(scans)["no_helmet"] == 1

    pdf = report.build_pdf(k, storage.violation_counts(scans), scans, "Summary – with “quotes” ✓")
    assert pdf[:4] == b"%PDF"


def test_agent_routes(monkeypatch):
    monkeypatch.setattr(agent, "search", lambda q, top_k: [{"source": "m", "chunk": 0, "text": q}])
    monkeypatch.setattr(agent, "ask_llm", lambda prompt: None)          # no API key

    with_image = agent.ask_agent("Is he safe?", vision="VIOLATIONS: ['no_helmet']", violations=["no_helmet"])
    assert with_image["steps"] == ["use_image", "retrieve", "answer"]
    assert "no_helmet" in with_image["docs"][0]["text"]                 # image improved the search
    assert "GEMINI_API_KEY not set" in with_image["answer"]

    text_only = agent.ask_agent("What PPE is required?")
    assert text_only["steps"] == ["retrieve", "answer"]
