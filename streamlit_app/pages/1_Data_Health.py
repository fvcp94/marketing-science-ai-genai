import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Data Health", page_icon="🧪", layout="wide")

DATA_PATH = "data/mock_marketing_data.xlsx"


@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset at: {path}")
    df = pd.read_excel(path, sheet_name=0)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")

    # numeric
    for c in ["impressions", "clicks", "spend", "mqls", "sqls", "pipeline_value", "closed_won", "revenue"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # categoricals
    for c in ["channel", "campaign", "region", "segment", "utm_source"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("Unknown").str.strip()

    return df


st.title("🧪 Data Health & QA")
st.caption("Quick checks to build trust before reporting/modeling.")

df = load_data()

# Missingness
st.subheader("Missingness")
miss = (df.isna().mean() * 100).sort_values(ascending=False).reset_index()
miss.columns = ["column", "pct_missing"]
fig = px.bar(miss, x="column", y="pct_missing", title="Percent missing by column")
fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# KPI sanity checks
st.subheader("KPI sanity checks")
tmp = df.copy()

def safe_div(a, b):
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

if set(["clicks", "impressions"]).issubset(tmp.columns):
    tmp["ctr"] = safe_div(tmp["clicks"], tmp["impressions"])
if set(["spend", "clicks"]).issubset(tmp.columns):
    tmp["cpc"] = safe_div(tmp["spend"], tmp["clicks"])
if set(["spend", "mqls"]).issubset(tmp.columns):
    tmp["cpl"] = safe_div(tmp["spend"], tmp["mqls"])

c1, c2, c3 = st.columns(3)
c1.metric("Rows with CTR > 25%", int((tmp.get("ctr", pd.Series([0]*len(tmp))) > 0.25).sum()))
c2.metric("Rows with CPC > $50", int((tmp.get("cpc", pd.Series([0]*len(tmp))) > 50).sum()))
c3.metric("Rows with CPL > $500", int((tmp.get("cpl", pd.Series([0]*len(tmp))) > 500).sum()))

# Outliers (robust z-score) by channel
st.subheader("Outlier candidates (robust z-score by channel)")
metric = st.selectbox("Metric", ["spend", "clicks", "mqls", "sqls", "revenue", "pipeline_value"], index=0)

x = df[["date"] + [c for c in ["channel", "campaign", "region", "segment"] if c in df.columns] + [metric]].copy()
x[metric] = pd.to_numeric(x[metric], errors="coerce").fillna(0)

def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    return (s - med) / (1.4826 * mad + 1e-9)

if "channel" in x.columns:
    x["rz"] = x.groupby("channel")[metric].transform(robust_z)
else:
    x["rz"] = robust_z(x[metric])

out = x.loc[x["rz"].abs() >= 6].sort_values("rz", ascending=False)
st.caption("Flags rows worth investigating (not automatically 'bad data').")
st.dataframe(out.head(250), use_container_width=True)

with st.expander("Preview raw columns"):
    st.write(list(df.columns))
