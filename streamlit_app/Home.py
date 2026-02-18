import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Marketing Science AI + GenAI Platform",
    page_icon="📈",
    layout="wide",
)

DATA_PATH = "data/mock_marketing_data.xlsx"


@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing dataset at: {path}\n"
            "Make sure `data/mock_marketing_data.xlsx` is committed to GitHub."
        )

    df = pd.read_excel(path, sheet_name=0)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Make common numeric cols numeric if they exist
    numeric_cols = [
        "impressions", "clicks", "spend", "mqls", "sqls",
        "pipeline_value", "closed_won", "revenue"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Standardize common categoricals
    cat_cols = ["channel", "campaign", "region", "segment", "utm_source"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("Unknown").str.strip()

    return df


def kpis(df: pd.DataFrame) -> dict:
    spend = float(df["spend"].sum()) if "spend" in df.columns else 0.0
    revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
    impressions = float(df["impressions"].sum()) if "impressions" in df.columns else 0.0
    clicks = float(df["clicks"].sum()) if "clicks" in df.columns else 0.0
    mqls = float(df["mqls"].sum()) if "mqls" in df.columns else 0.0
    sqls = float(df["sqls"].sum()) if "sqls" in df.columns else 0.0
    won = float(df["closed_won"].sum()) if "closed_won" in df.columns else 0.0

    ctr = clicks / impressions if impressions else 0.0
    cpc = spend / clicks if clicks else 0.0
    cpl = spend / mqls if mqls else 0.0
    roas = revenue / spend if spend else 0.0
    win_rate = won / sqls if sqls else 0.0

    return {
        "spend": spend,
        "revenue": revenue,
        "roas": roas,
        "ctr": ctr,
        "cpc": cpc,
        "mqls": mqls,
        "sqls": sqls,
        "win_rate": win_rate,
        "rows": int(len(df)),
    }


st.title("📈 Marketing Science AI + GenAI Platform")
st.caption(
    "End-to-end marketing analytics, predictive modeling, MMM optimization, and GenAI stakeholder Q&A."
)

# Debug-friendly: show that file exists in cloud
with st.sidebar:
    st.header("Dataset")
    st.write("Expected path:")
    st.code(DATA_PATH)
    st.write("Files in /data (cloud):")
    try:
        st.code("\n".join(os.listdir("data")))
    except Exception:
        st.info("No data folder visible at runtime.")

try:
    df = load_data()
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

k = kpis(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spend", f"${k['spend']:,.0f}")
c2.metric("Revenue", f"${k['revenue']:,.0f}")
c3.metric("ROAS", f"{k['roas']:.2f}x")
c4.metric("MQLs", f"{k['mqls']:,.0f}")
c5.metric("Win rate", f"{k['win_rate']*100:.1f}%")

st.divider()

st.subheader("Data Preview")
st.write(f"Rows: **{k['rows']:,}** | Columns: **{len(df.columns)}**")
st.dataframe(df.head(200), use_container_width=True)

st.info(
    "Next: open the pages in the sidebar (Data Health, Performance Dashboard, Predictive Growth, MMM, GenAI Insights)."
)
