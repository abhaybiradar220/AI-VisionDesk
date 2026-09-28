"""
MILESTONE 4 - Compliance report (PDF)
--------------------------------------
Numbers from the dashboard + a short summary written by Gemini.
"""
from datetime import datetime

from fpdf import FPDF

from knowledge_base import search
from llm import ask_llm


def safe(text):
    """fpdf's basic fonts can't print characters like – “ ” ✓, so replace them."""
    for bad, good in {"–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
                      "•": "-", "✓": "v", "≥": ">=", "…": "..."}.items():
        text = str(text).replace(bad, good)
    return text.encode("latin-1", "replace").decode("latin-1")


def write_summary(k, counts):
    """Ask Gemini for a short executive summary, grounded in the safety manual."""
    stats = (f"Scans: {k['scans']}, compliance rate: {k['compliance']}%, "
             f"violations: {k['violations']}, by type: {counts.to_dict()}")
    rules = "\n".join(d["text"] for d in search("safety rules for " + " ".join(counts.index), top_k=3))

    prompt = (f"Write a short workplace safety report summary (max 120 words, plain text).\n"
              f"Say what happened, which rules were broken, and 2 recommended actions.\n\n"
              f"Statistics: {stats}\n\nSafety rules:\n{rules}")
    return ask_llm(prompt) or f"(Gemini not configured) {stats}"


def build_pdf(k, counts, scans, summary):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, "AI-Powered Visual Data Analytics and Business Intelligence Platform", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Safety Compliance Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, "Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    def heading(text):
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)

    def row(a, b):
        pdf.cell(90, 7, safe(a), border=1)
        pdf.cell(90, 7, safe(b), border=1, new_x="LMARGIN", new_y="NEXT")

    heading("1. Key numbers")
    row("Images / videos checked", k["scans"])
    row("Compliance rate", f"{k['compliance']}%" if k["compliance"] is not None else "-")
    row("Total violations", k["violations"])
    row("User satisfaction", f"{k['satisfaction']}%" if k["satisfaction"] is not None else "-")
    pdf.ln(4)

    heading("2. Summary")
    pdf.multi_cell(0, 6, safe(summary))
    pdf.ln(4)

    heading("3. Violations by type")
    if len(counts):
        for name, n in counts.items():
            row(name, n)
    else:
        pdf.cell(0, 7, "No violations recorded.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    heading("4. Latest scans")
    for s in scans.sort_values("time", ascending=False).head(15).itertuples():
        pdf.cell(0, 6, safe(f"{s.time:%Y-%m-%d %H:%M}  |  {s.source}  |  {s.violations} violation(s) "
                            f"{s.violation_types}"), new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())
