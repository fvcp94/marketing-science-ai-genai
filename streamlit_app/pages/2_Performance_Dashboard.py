import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Performance Dashboard", page_icon="📊", layout="wide")

DATA_PATH = "data/mock_marketing_data.xlsx"


@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset at: {path}")
    df = pd.read_excel(path, sheet_name=0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["impressions", "clicks", "spend", "mqls", "sqls", "closed_won", "revenue"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["channel", "region", "segment"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("Unknown").str.strip()
    return df


st.title("📊 Performance Dashboard")
df = load_data()

with st.sidebar:
    st.header("Filters")
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    dr = st.date_input("Date range", value=(dmin, dmax))
    channels = st.multiselect("Channel", sorted(df["channel"].unique()), default=[])

d = df.copy()
if dr and len(dr) == 2:
    d = d[(d["date"].dt.date >= dr[0]) & (d["date"].dt.date <= dr[1])]
if channels:
    d = d[d["channel"].isin(channels)]

daily = d.groupby("date", as_index=False).agg(
    spend=("spend", "sum"),
    revenue=("revenue", "sum"),
    mqls=("mqls", "sum"),
    sqls=("sqls", "sum"),
    closed_won=("closed_won", "sum"),
)

c1, c2 = st.columns(2)
c1.plotly_chart(px.line(daily, x="date", y="spend", title="Daily Spend"), use_container_width=True)
c2.plotly_chart(px.line(daily, x="date", y="revenue", title="Daily Revenue"), use_container_width=True)

by_ch = d.groupby("channel", as_index=False).agg(
    spend=("spend", "sum"),
    revenue=("revenue", "sum"),
    mqls=("mqls", "sum"),
    sqls=("sqls", "sum"),
)
by_ch["roas"] = by_ch["revenue"] / by_ch["spend"].replace(0, np.nan)
by_ch["cpl"] = by_ch["spend"] / by_ch["mqls"].replace(0, np.nan)

st.subheader("Channel Summary")
st.dataframe(by_ch.sort_values("revenue", ascending=False).round(3), use_container_width=True)
