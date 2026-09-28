"""
MILESTONE 3 - Multimodal RAG + LangGraph Agent
-----------------------------------------------
The agent answers a question using:
   (a) what the camera found   (vision summary text)
   (b) the safety documents    (ChromaDB search)

            image result given?
   START ──── yes ──> use_image ──┐
        └──── no ─────────────────┴──> retrieve ──> answer ──> END
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from knowledge_base import search
from llm import ask_llm

PROMPT = """You are a workplace safety assistant.
Answer using ONLY the documents below. If the answer is not there, say "Not found in documents."
Mention the source of each fact like [file #chunk].
If camera findings are given, say whether they break any rule.

Camera findings: {vision}

Documents:
{docs}

Question: {question}
Answer:"""


class State(TypedDict, total=False):
    question: str
    vision: str          # e.g. "site.jpg: VIOLATIONS: ['no_helmet']"
    violations: list
    search_query: str
    docs: list
    answer: str
    steps: list


# ── Step 1 (optional): use the camera result to improve the search ──
def use_image(state):
    query = state["question"] + " " + " ".join(set(state.get("violations", [])))
    return {"search_query": query, "steps": state.get("steps", []) + ["use_image"]}


# ── Step 2: find relevant document chunks ──
def retrieve(state):
    query = state.get("search_query") or state["question"]
    return {"docs": search(query, top_k=3), "steps": state.get("steps", []) + ["retrieve"]}


# ── Step 3: ask Gemini, grounded in those chunks ──
def answer(state):
    docs_text = "\n\n".join(f"[{d['source']} #{d['chunk']}] {d['text']}" for d in state["docs"])
    prompt = PROMPT.format(vision=state.get("vision") or "None",
                           docs=docs_text or "(no documents)",
                           question=state["question"])
    reply = ask_llm(prompt)

    if reply is None:   # no API key -> show the best passages instead of crashing
        reply = "⚠️ GEMINI_API_KEY not set. Most relevant passages:\n\n" + "\n\n".join(
            f"- [{d['source']} #{d['chunk']}] {d['text']}" for d in state["docs"])

    return {"answer": reply, "steps": state.get("steps", []) + ["answer"]}


def route(state):
    return "use_image" if state.get("vision") else "retrieve"


graph = StateGraph(State)
graph.add_node("use_image", use_image)
graph.add_node("retrieve", retrieve)
graph.add_node("answer", answer)
graph.add_conditional_edges(START, route, {"use_image": "use_image", "retrieve": "retrieve"})
graph.add_edge("use_image", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)
agent = graph.compile()


def ask_agent(question, vision="", violations=None):
    return agent.invoke({"question": question, "vision": vision,
                         "violations": violations or [], "steps": []})


if __name__ == "__main__":
    result = ask_agent("What PPE is required on site?")
    print("Steps:", result["steps"])
    print(result["answer"])
