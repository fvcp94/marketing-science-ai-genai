import os
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="GenAI Insights", page_icon="✨", layout="wide")

DATA_PATH = "data/mock_marketing_data.xlsx"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODEL_CHOICES = [
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
]

SYSTEM_PROMPT = """You are a Marketing Science analyst.
Be concise, executive-friendly, and action-oriented.
Use ONLY the aggregated metrics provided (never row-level records).
Always respond with the following sections:

## What happened
## Why it happened (hypotheses)
## What to do next (3 actions)
## Risks / caveats

Include numbers from context whenever possible.
If the question cannot be answered from the aggregates, say what extra data would be required.
"""


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset at: {DATA_PATH}. Make sure it is committed to GitHub.")
    df = pd.read_excel(DATA_PATH, sheet_name=0)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for c in ["impressions", "clicks", "spend", "mqls", "sqls", "pipeline_value", "closed_won", "revenue"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in ["channel", "region", "segment", "campaign"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("Unknown").str.strip()

    return df


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def kpis(df: pd.DataFrame) -> dict:
    spend = float(df["spend"].sum()) if "spend" in df.columns else 0.0
    revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
    impressions = float(df["impressions"].sum()) if "impressions" in df.columns else 0.0
    clicks = float(df["clicks"].sum()) if "clicks" in df.columns else 0.0
    mqls = float(df["mqls"].sum()) if "mqls" in df.columns else 0.0
    sqls = float(df["sqls"].sum()) if "sqls" in df.columns else 0.0
    won = float(df["closed_won"].sum()) if "closed_won" in df.columns else 0.0
    pipeline = float(df["pipeline_value"].sum()) if "pipeline_value" in df.columns else 0.0

    return {
        "spend": spend,
        "revenue": revenue,
        "pipeline_value": pipeline,
        "roas": safe_div(revenue, spend),
        "ctr": safe_div(clicks, impressions),
        "cpc": safe_div(spend, clicks),
        "cpl": safe_div(spend, mqls),
        "mqls": mqls,
        "sqls": sqls,
        "win_rate": safe_div(won, sqls),
    }


def channel_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("channel", as_index=False).agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        mqls=("mqls", "sum"),
        sqls=("sqls", "sum"),
        pipeline_value=("pipeline_value", "sum") if "pipeline_value" in df.columns else ("spend", "sum"),
    )
    g["roas"] = g["revenue"] / g["spend"].replace(0, np.nan)
    g["cpl"] = g["spend"] / g["mqls"].replace(0, np.nan)
    g["cpsql"] = g["spend"] / g["sqls"].replace(0, np.nan)
    return g.sort_values("revenue", ascending=False)


def openrouter_enabled() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def call_openrouter(messages, model: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.2}

    r = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---------- UI ----------
st.title("✨ GenAI Stakeholder Insights (Ask Questions)")
st.caption("Stakeholders ask questions → AI generates an executive-ready story using aggregated metrics only.")

try:
    df = load_data()
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Scope & Model")

    days = st.slider("Analysis window (days)", 7, 180, 28, 7)

    chosen = st.selectbox("Free model (OpenRouter)", FREE_MODEL_CHOICES, index=0)
    custom = st.text_input("Optional: override model ID", value="")
    model = custom.strip() if custom.strip() else chosen

    st.divider()
    st.subheader("GenAI setup")
    if openrouter_enabled():
        st.success("OPENROUTER_API_KEY detected ✅")
    else:
        st.warning("OPENROUTER_API_KEY not set ❗")
        st.caption("Streamlit Cloud → App → Settings → Secrets")
        st.code('OPENROUTER_API_KEY = "your_key"', language="toml")

dmax = df["date"].max() if "date" in df.columns else None
if dmax is None or pd.isna(dmax):
    st.error("No valid `date` column found. GenAI Insights needs a date column for windowing.")
    st.stop()

dmin = dmax - pd.Timedelta(days=days)
scope = df[(df["date"] >= dmin) & (df["date"] <= dmax)].copy()

K = kpis(scope)

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spend", f"${K['spend']:,.0f}")
c2.metric("Revenue", f"${K['revenue']:,.0f}")
c3.metric("ROAS", f"{K['roas']:.2f}x")
c4.metric("MQLs", f"{K['mqls']:,.0f}")
c5.metric("Win rate", f"{K['win_rate']*100:.1f}%")

# Deterministic context
ct = channel_table(scope).head(12).round(3)

# Trend chart
daily = scope.groupby("date", as_index=False).agg(spend=("spend", "sum"), revenue=("revenue", "sum"), mqls=("mqls", "sum"))
fig = px.line(daily, x="date", y=["spend", "revenue", "mqls"], title="Window trend: spend, revenue, MQLs")
fig.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))

left, right = st.columns([1.2, 1])
with left:
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Channel snapshot (top 12)")
    st.dataframe(ct, use_container_width=True)

st.divider()

# Preset questions (buttons) + input UX that works reliably
preset_questions = [
    "What changed in this window vs the previous window, and what should we do next?",
    "Which channels are inefficient (high spend, low revenue/ROAS) and what are 2 hypotheses?",
    "Where is funnel leakage happening (click → MQL → SQL) and what experiments should we run?",
    "Give an executive summary: top drivers, top risks, and 3 actions to improve ROI in the next 2 weeks.",
    "If we reallocated 10–20% budget, which channels would you increase/decrease and why?",
]

if "draft_question" not in st.session_state:
    st.session_state.draft_question = preset_questions[0]

if "chat" not in st.session_state:
    st.session_state.chat = [
        {
            "role": "assistant",
            "content": "Ask me about ROI, channel performance, funnel health, and recommended next actions.",
        }
    ]

tabs = st.tabs(["💬 Stakeholder Q&A", "🧾 What the AI sees (aggregates only)"])

with tabs[0]:
    st.subheader("Model questions (click one)")
    bcols = st.columns(2)
    for i, q in enumerate(preset_questions):
        if bcols[i % 2].button(q, use_container_width=True):
            st.session_state.draft_question = q

    st.markdown("### Ask your question")
    st.session_state.draft_question = st.text_area(
        "Question",
        value=st.session_state.draft_question,
        height=90,
        label_visibility="collapsed",
    )

    run = st.button("Generate answer", type="primary")

    # Render chat history
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Build safe context
    # (Aggregates only — never row-level)
    prev_scope = df[(df["date"] >= (dmin - pd.Timedelta(days=days))) & (df["date"] < dmin)].copy()
    K_prev = kpis(prev_scope) if len(prev_scope) else None

    context = {
        "window_start": str(dmin.date()),
        "window_end": str(dmax.date()),
        "kpis": {k: float(v) for k, v in K.items()},
        "channel_table_top12": ct.to_dict(orient="records"),
        "previous_window_kpis": ({k: float(v) for k, v in K_prev.items()} if K_prev else None),
    }

    if run:
        q = st.session_state.draft_question.strip()
        if not q:
            st.warning("Type a question first.")
        else:
            st.session_state.chat.append({"role": "user", "content": q})

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"QUESTION:\n{q}\n\nAGGREGATED CONTEXT (JSON):\n{json.dumps(context)}"},
            ]

            with st.chat_message("assistant"):
                if not openrouter_enabled():
                    st.warning("GenAI not configured. Add OPENROUTER_API_KEY in Streamlit Cloud → Settings → Secrets.")
                    st.json(context)
                    answer = "Add OPENROUTER_API_KEY to enable answers."
                else:
                    with st.spinner("Generating…"):
                        try:
                            answer = call_openrouter(messages, model=model)
                            st.markdown(answer)
                        except Exception as e:
                            st.error(f"LLM call failed: {e}")
                            st.json(context)
                            answer = "LLM call failed — verify API key and model name."

            st.session_state.chat.append({"role": "assistant", "content": answer})

with tabs[1]:
    st.subheader("Aggregates only (safe + interview-ready)")
    st.caption("This page intentionally sends only aggregated metrics to the LLM. No row-level records are shared.")

    st.write("**Channel snapshot (top 12):**")
    st.dataframe(ct, use_container_width=True)

    # Rebuild the same context used above (so this tab works even before a question)
    prev_scope = df[(df["date"] >= (dmin - pd.Timedelta(days=days))) & (df["date"] < dmin)].copy()
    K_prev = kpis(prev_scope) if len(prev_scope) else None

    context_preview = {
        "window_start": str(dmin.date()),
        "window_end": str(dmax.date()),
        "kpis": {k: float(v) for k, v in K.items()},
        "channel_table_top12": ct.to_dict(orient="records"),
        "previous_window_kpis": ({k: float(v) for k, v in K_prev.items()} if K_prev else None),
    }
    st.json(context_preview)
