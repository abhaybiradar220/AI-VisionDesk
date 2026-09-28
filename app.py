"""
AI-Powered Visual Data Analytics and Business Intelligence Platform - Streamlit dashboard (MILESTONE 4)
--------------------------------------------------
Run:  streamlit run app.py

Pages:
  🏠 Home          overview + status
  👷 PPE Detection image / video          (Milestone 1)
  📄 Documents     upload + search manuals (Milestone 2)
  🤖 Ask AI        LangGraph agent          (Milestone 3)
  📊 Dashboard     KPIs + charts            (Milestone 4)
  📋 Report        PDF download             (Milestone 4)

Remember: Streamlit re-runs this whole file on every click.
  - @st.cache_resource  -> load heavy things only once
  - st.session_state    -> remember results between clicks
  - save data only inside a button, or it is saved again on every rerun
"""
import os
import tempfile

import cv2
import streamlit as st

import knowledge_base as kb
import storage
import vision
from agent import ask_agent
from documents import process_file
from llm import MODEL, has_key
from report import build_pdf, write_summary

st.set_page_config(page_title="AI-Powered Visual Data Analytics and Business Intelligence Platform", page_icon="🦺", layout="wide")

# On Streamlit Cloud the API key comes from "Secrets"
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass


# ── load once ──────────────────────────────────────────────
@st.cache_resource
def get_model():
    return vision.load_model()


@st.cache_resource
def load_sample_manual():
    """If the knowledge base is empty (e.g. fresh cloud start), add a manual automatically."""
    if kb.total_chunks() == 0:
        for name in ["safety_manual.pdf", "sample_safety_manual.txt"]:
            path = os.path.join(vision.BASE_DIR, "data", name)
            if os.path.exists(path):
                kb.add_document(process_file(path), name)
                return name
    return None


def save_upload(uploaded):
    suffix = os.path.splitext(uploaded.name)[1]          # keep the real extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


model = get_model()
load_sample_manual()

# ── sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦺 AI-Powered Visual Data Analytics and Business Intelligence Platform")
    page = st.radio("Go to", ["🏠 Home", "👷 PPE Detection", "📄 Documents",
                              "🤖 Ask AI", "📊 Dashboard", "📋 Report"])
    st.divider()
    st.write(("✅" if model else "❌") + " PPE model (best.pt)")
    st.write(("✅" if has_key() else "⚠️") + f" Gemini ({MODEL})")
    st.write(f"📚 {kb.total_chunks()} chunks in knowledge base")


# ══════════════════════════════════════════════════════════
# 🏠 HOME
# ══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🦺 AI-Powered Visual Data Analytics and Business Intelligence Platform")
    st.subheader("VisionDesk AI - Multimodal Workplace Intelligence System")
    st.markdown("""
1. **👷 PPE Detection** – upload a site image or video, see violations
2. **📄 Documents** – upload safety manuals and inspection reports
3. **🤖 Ask AI** – ask questions; the AI uses the manual **and** the last image result
4. **📊 Dashboard** – compliance rate, violations, trends
5. **📋 Report** – download a PDF compliance report
""")
    if not model:
        st.warning("Put your trained **best.pt** in the project folder to enable detection.")
    if not has_key():
        st.info("Add **GEMINI_API_KEY** to the `.env` file for AI answers.")


# ══════════════════════════════════════════════════════════
# 👷 PPE DETECTION  (Milestone 1)
# ══════════════════════════════════════════════════════════
elif page == "👷 PPE Detection":
    st.header("👷 PPE Detection")
    if not model:
        st.error("best.pt not found in the project folder.")
        st.stop()

    kind = st.radio("Input", ["Image", "Video"], horizontal=True)
    conf = st.slider("Confidence", 0.1, 0.9, 0.4, 0.05,
                     help="Higher = fewer false alarms, but may miss some items")

    if kind == "Image":
        file = st.file_uploader("Upload a site image", type=["jpg", "jpeg", "png"])
        if file and st.button("Analyze image", type="primary"):
            path = save_upload(file)
            dets = vision.detect(model, path, conf)
            result = vision.summarize(dets, file.name)
            storage.log_scan(file.name, result, "image")           # saved only on click
            st.session_state["last"] = {"name": file.name, "result": result, "dets": dets,
                                        "image": vision.draw(cv2.imread(path), dets)}
    else:
        file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        every_n = st.slider("Check 1 of every N frames", 1, 30, 5)
        if file and st.button("Analyze video", type="primary"):
            with st.spinner("Analysing video..."):
                dets, preview, frames = vision.detect_video(model, save_upload(file), every_n, conf)
            result = vision.summarize(dets, file.name)
            storage.log_scan(file.name, result, "video")
            st.session_state["last"] = {"name": f"{file.name} ({frames} frames)", "result": result,
                                        "dets": dets, "image": preview}

    last = st.session_state.get("last")
    if last:
        col1, col2 = st.columns([3, 2])
        if last["image"] is not None:
            col1.image(cv2.cvtColor(last["image"], cv2.COLOR_BGR2RGB), caption=last["name"])
        with col2:
            r = last["result"]
            if r["compliant"]:
                st.success("✅ COMPLIANT - no violations")
            else:
                st.error("🚨 VIOLATIONS: " + ", ".join(sorted(set(r["violations"]))))
            st.metric("Objects detected", len(last["dets"]))
            st.metric("Violations", len(r["violations"]))
            st.write("**Counts:**", r["counts"])
            st.caption("This result is now available in 🤖 Ask AI.")


# ══════════════════════════════════════════════════════════
# 📄 DOCUMENTS  (Milestone 2)
# ══════════════════════════════════════════════════════════
elif page == "📄 Documents":
    st.header("📄 Document Knowledge Base")
    files = st.file_uploader("Upload manuals / inspection or incident reports",
                             type=["pdf", "txt", "docx"], accept_multiple_files=True)
    if files and st.button("Add to knowledge base", type="primary"):
        for f in files:
            chunks = process_file(save_upload(f))
            if chunks:
                kb.add_document(chunks, f.name)
                st.success(f"{f.name}: {len(chunks)} chunks added")
            else:
                st.warning(f"{f.name}: no text found (scanned PDF?)")

    st.subheader("Documents in the knowledge base")
    docs = kb.list_documents()
    if docs:
        st.table([{"document": name, "chunks": n} for name, n in docs.items()])
    else:
        st.info("No documents yet.")

    st.subheader("🔍 Test search")
    q = st.text_input("Search", placeholder="When must goggles be worn?")
    if q:
        for r in kb.search(q):
            st.info(f"**{r['source']} #{r['chunk']}** (distance {r['distance']})\n\n{r['text']}")


# ══════════════════════════════════════════════════════════
# 🤖 ASK AI  (Milestone 3 - LangGraph agent)
# ══════════════════════════════════════════════════════════
elif page == "🤖 Ask AI":
    st.header("🤖 Ask the Safety AI")

    last = st.session_state.get("last")
    use_image = False
    if last:
        use_image = st.checkbox(f"Include last detection result ({last['name']})", value=True)
        if use_image:
            st.caption(last["result"]["text"])

    question = st.text_input("Your question", placeholder="Is this worker compliant? Which rule applies?")

    if st.button("Ask", type="primary") and question:
        vis = last["result"] if (use_image and last) else None
        with st.spinner("Thinking..."):
            out = ask_agent(question,
                            vision=vis["text"] if vis else "",
                            violations=vis["violations"] if vis else [])
        st.session_state["answer"] = {"question": question, **out}
        st.session_state["voted"] = False

    ans = st.session_state.get("answer")
    if ans:
        st.markdown("### Answer")
        st.write(ans["answer"])
        with st.expander("How the agent worked"):
            st.write("**Steps:** " + " → ".join(ans["steps"]))
            for d in ans["docs"]:
                st.caption(f"[{d['source']} #{d['chunk']}] {d['text'][:200]}")

        if not st.session_state.get("voted"):
            st.write("Was this answer helpful?")
            c1, c2 = st.columns(2)
            if c1.button("👍 Yes"):
                storage.log_feedback(ans["question"], True)
                st.session_state["voted"] = True
                st.rerun()
            if c2.button("👎 No"):
                storage.log_feedback(ans["question"], False)
                st.session_state["voted"] = True
                st.rerun()
        else:
            st.caption("Thanks for your feedback!")


# ══════════════════════════════════════════════════════════
# 📊 DASHBOARD  (Milestone 4)
# ══════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.header("📊 Safety Dashboard")
    scans = storage.load_scans()
    k = storage.kpis(scans)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images / videos checked", k["scans"])
    c2.metric("Compliance rate", f"{k['compliance']}%" if k["compliance"] is not None else "-")
    c3.metric("Total violations", k["violations"])
    c4.metric("User satisfaction", f"{k['satisfaction']}%" if k["satisfaction"] is not None else "-",
              help="Target: 85% or more")

    if scans.empty:
        st.info("No data yet. Analyse some images in 👷 PPE Detection.")
    else:
        left, right = st.columns(2)
        left.subheader("Violations by type")
        counts = storage.violation_counts(scans)
        if len(counts):
            left.bar_chart(counts)
        else:
            left.success("No violations so far 🎉")
        right.subheader("Violations per day")
        right.line_chart(scans.set_index("time")["violations"].resample("D").sum())

        st.subheader("All scans")
        st.dataframe(scans.sort_values("time", ascending=False), hide_index=True)

    if st.button("🗑️ Clear history"):
        storage.clear_all()
        st.rerun()


# ══════════════════════════════════════════════════════════
# 📋 REPORT  (Milestone 4)
# ══════════════════════════════════════════════════════════
elif page == "📋 Report":
    st.header("📋 Compliance Report")
    scans = storage.load_scans()

    if st.button("Generate report", type="primary"):
        k = storage.kpis(scans)
        counts = storage.violation_counts(scans)
        with st.spinner("Writing summary..."):
            summary = write_summary(k, counts)
        st.session_state["report"] = {"summary": summary, "pdf": build_pdf(k, counts, scans, summary)}

    rep = st.session_state.get("report")
    if rep:
        st.subheader("Summary")
        st.write(rep["summary"])
        st.download_button("⬇️ Download PDF", rep["pdf"],
                           file_name="safety_compliance_report.pdf", mime="application/pdf")
        st.download_button("⬇️ Download data (CSV)", scans.to_csv(index=False),
                           file_name="scans.csv", mime="text/csv")
