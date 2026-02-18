import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

st.set_page_config(page_title="Predictive Growth", page_icon="🤖", layout="wide")

DATA_PATH = "data/mock_marketing_data.xlsx"
TARGETS = ["mqls", "sqls", "revenue", "pipeline_value"]


@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset at: {path}")
    df = pd.read_excel(path, sheet_name=0)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")

    for c in ["impressions", "clicks", "spend", "mqls", "sqls", "pipeline_value", "closed_won", "revenue"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in ["channel", "region", "segment"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("Unknown").str.strip()

    return df


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["dow"] = d["date"].dt.dayofweek
    d["dom"] = d["date"].dt.day
    d["month_num"] = d["date"].dt.month

    d["ctr"] = (d["clicks"] / d["impressions"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    d["cpc"] = (d["spend"] / d["clicks"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    d["cpm"] = (d["spend"] / d["impressions"].replace(0, np.nan) * 1000).replace([np.inf, -np.inf], np.nan).fillna(0)
    return d


st.title("🤖 Predictive Growth Forecasting")
st.caption("Time-split forecasting to support proactive budget and pipeline planning (no leakage).")

df = load_data().sort_values("date")
df = build_features(df)

# Aggregate to daily x channel (more stable)
group_cols = ["date"] + [c for c in ["channel", "region", "segment"] if c in df.columns]
agg = df.groupby(group_cols, as_index=False).agg(
    impressions=("impressions", "sum"),
    clicks=("clicks", "sum"),
    spend=("spend", "sum"),
    mqls=("mqls", "sum"),
    sqls=("sqls", "sum"),
    pipeline_value=("pipeline_value", "sum"),
    revenue=("revenue", "sum"),
    ctr=("ctr", "mean"),
    cpc=("cpc", "mean"),
    cpm=("cpm", "mean"),
)
agg["dow"] = agg["date"].dt.dayofweek
agg["dom"] = agg["date"].dt.day
agg["month_num"] = agg["date"].dt.month

col1, col2 = st.columns([1, 1])
target = col1.selectbox("Target", [t for t in TARGETS if t in agg.columns], index=0)

months = sorted(agg["date"].dt.to_period("M").astype(str).unique().tolist())
split_month = col2.selectbox("Train/Test split month", months[1:] if len(months) > 1 else months, index=min(2, max(0, len(months[1:]) - 1)))
split_date = pd.to_datetime(f"{split_month}-01")

train = agg[agg["date"] < split_date].copy()
test = agg[agg["date"] >= split_date].copy()

if train.empty or test.empty:
    st.warning("Not enough data on either side of the split. Pick a different split month.")
    st.stop()

num_cols = [c for c in ["impressions", "clicks", "spend", "ctr", "cpc", "cpm", "dow", "dom", "month_num"] if c in agg.columns]
cat_cols = [c for c in ["channel", "region", "segment"] if c in agg.columns]

X_train = train[num_cols + cat_cols]
y_train = train[target].fillna(0)
X_test = test[num_cols + cat_cols]
y_test = test[target].fillna(0)

pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
)

model = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.08,
    max_iter=300,
    random_state=42,
)

pipe = Pipeline([("pre", pre), ("model", model)])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

mae = float(mean_absolute_error(y_test, pred))
baseline = float(mean_absolute_error(y_test, np.full_like(y_test, y_train.mean(), dtype=float)))

st.success(f"Model MAE: {mae:,.2f}  |  Baseline MAE: {baseline:,.2f}")

preds = test[group_cols].copy()
preds["actual"] = y_test.values
preds["pred"] = pred
preds["abs_error"] = (preds["actual"] - preds["pred"]).abs()

st.subheader("Actual vs Predicted (daily total)")
tot = preds.groupby("date", as_index=False).agg(actual=("actual", "sum"), pred=("pred", "sum"))
m = tot.melt(id_vars="date", value_vars=["actual", "pred"], var_name="series", value_name="value")
fig = px.line(m, x="date", y="value", color="series", title=f"{target}: actual vs predicted")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Where the model struggled most")
st.dataframe(preds.sort_values("abs_error", ascending=False).head(40), use_container_width=True)
