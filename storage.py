"""
MILESTONE 4 - Save history for the dashboard
---------------------------------------------
Two simple CSV files (open them in Excel too):
    logs/scans.csv     -> one row per analysed image / video
    logs/feedback.csv  -> one row per 👍 / 👎 on an AI answer
"""
import os
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
SCANS_FILE = os.path.join(LOG_DIR, "scans.csv")
FEEDBACK_FILE = os.path.join(LOG_DIR, "feedback.csv")


def _append(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _read(path, columns):
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df


# ── scans ──
def log_scan(source, summary, media="image"):
    _append(SCANS_FILE, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "media": media,
        "violations": len(summary["violations"]),
        "violation_types": ";".join(summary["violations"]),
        "compliant": summary["compliant"],
    })


def load_scans():
    df = _read(SCANS_FILE, ["time", "source", "media", "violations", "violation_types", "compliant"])
    df["violation_types"] = df["violation_types"].fillna("")
    return df


def violation_counts(scans):
    """How many times each violation type happened."""
    items = [v for text in scans["violation_types"] for v in str(text).split(";") if v]
    return pd.Series(items, dtype="object").value_counts()


# ── feedback ──
def log_feedback(question, liked):
    _append(FEEDBACK_FILE, {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "question": question, "liked": int(liked)})


def satisfaction():
    """% of 👍 votes (project target: 85% or more). None if no votes yet."""
    df = _read(FEEDBACK_FILE, ["time", "question", "liked"])
    if df.empty:
        return None
    return round(df["liked"].mean() * 100, 1)


# ── KPIs ──
def kpis(scans):
    total = len(scans)
    return {
        "scans": total,
        "compliance": round(scans["compliant"].astype(bool).mean() * 100, 1) if total else None,
        "violations": int(scans["violations"].sum()) if total else 0,
        "satisfaction": satisfaction(),
    }


def clear_all():
    for path in (SCANS_FILE, FEEDBACK_FILE):
        if os.path.exists(path):
            os.remove(path)
