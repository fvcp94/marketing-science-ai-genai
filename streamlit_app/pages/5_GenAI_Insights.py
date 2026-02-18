import os
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="GenAI Insights", page_icon="✨", layout="wide")

DATA_PATH = "data/mock_marketing_data.xlsx"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "mistralai/mistral-7b-instruct:free"

SYSTEM_PROMPT = """You are a Marketing Science analyst.
Be executive-friendly and action-oriented.
Use ONLY the aggregated metrics provided (never row-level records).
Always format:
1) What happened
2) Why (hypotheses)
3) What to do next (3 actions)
4) Risks/caveats
Include key numbers when available.
"""


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset at: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH, sheet_name=0)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")

    for c in ["impressions", "clicks", "spend", "mqls", "sqls", "pipeline_value", "closed_won", "revenue"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in ["channel", "region", "segment", "campaign"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("Unknown").str.strip()

    return df


def kpis(df: pd.DataFrame) -> dict:
    spend = float(df["spend"].sum()) if "spend" in df.columns else 0.0
    revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
    clicks = float(df["clicks"].sum()) if "clicks" in df.columns else 0.0
    imps = float(df["impressions"].sum()) if "impressions" in df.columns else 0.0
    mqls = float(df["mqls"].sum()) if "mqls" in df.columns else 0.0
    sqls = float(df["sqls"].sum()) if "sqls" in df.columns else 0.0
    won = float(df["closed_won"].sum()) if "closed_won" in df.columns else 0.0

    return {
        "spend": spend,
        "revenue": revenue,
        "roas": (revenue / spend) if spend else 0.0,
        "mqls": mqls,
        "sqls": sqls,
        "win_rate": (won / sqls) if sqls else 0.0,
        "ctr": (clicks / imps) if imps else 0.0,
    }


def channel_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("channel", as_index=False).agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        mqls=("mqls", "sum"),
        sqls=("sqls", "sum"),
    )
    g["roas"] = g["revenue"] / g["spend"].replace(0, np.nan)
    g["cpl"] = g["spend"] / g["mqls"].replace(0, np.nan)
    return g.sort_values("revenue", ascending=False)


def openrouter_enabled() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def call_openrouter(messages, model: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    r = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


st.title("✨ GenAI Stakeholder Insights (Q&A)")
st.caption("Stakeholders ask questions → AI answers using aggregated metrics only.")

df = load_data()

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    days = st.slider("Window (days)", 7, 180, 28, 7)

dmax = df["date"].max()
dmin = dmax - pd.Timedelta(days=days)
scope = df[(df["date"] >= dmin) & (df["date"] <= dmax)].copy()

K = kpis(scope)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spend", f"${K['spend']:,.0f}")
c2.metric("Revenue", f"${K['revenue']:,.0f}")
c3.metric("ROAS", f"{K['roas']:.2f}x")
c4.metric("MQLs", f"{K['mqls']:,.0f}")

ct = channel_table(scope).head(12).round(3)

context = {
    "window_start": str(dmin.date()),
    "window_end": str(dmax.date()),
    "kpis": {k: float(v) for k, v in K.items()},
    "channel_table_top12": ct.to_dict(orient="records"),
}

tabs = st.tabs(["💬 Chat", "📦 Context sent to LLM"])
with tabs[0]:
    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "assistant", "content": "Ask me about ROI, channel performance, funnel health, or what to do next."}
        ]

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask a stakeholder-style question…")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

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
    st.subheader("Aggregates only (safe)")
    st.dataframe(ct, use_container_width=True)
    st.json(context)
