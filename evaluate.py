"""
MILESTONE 3 check - RAG retrieval accuracy (target: 85% or more)
For each question we know a word the correct chunk must contain.
Run:  python evaluate.py
"""
import os

import knowledge_base as kb
from documents import process_file

TESTS = [
    ("Do I need a helmet in the construction zone?", "hardhat"),
    ("When is a high visibility vest required?", "forklifts"),
    ("What eye protection is needed for grinding?", "goggles"),
    ("Which gloves are needed for sheet metal?", "cut-resistant"),
    ("When should a respirator be worn?", "respirator"),
    ("How often are fire drills held?", "three months"),
    ("Where are first aid kits kept?", "first aid kits"),
    ("How fast must an injury be reported?", "24 hours"),
    ("What is the penalty for a first PPE violation?", "verbal warning"),
    ("What compliance rate is the target?", "95 percent"),
]

manual = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_safety_manual.txt")
kb.add_document(process_file(manual), "sample_safety_manual.txt")

hits = 0
for question, keyword in TESTS:
    found = any(keyword in r["text"].lower() for r in kb.search(question, top_k=3))
    hits += found
    print("✅" if found else "❌", question)

print(f"\nRetrieval accuracy: {hits / len(TESTS) * 100:.0f}%  (target 85%)")
