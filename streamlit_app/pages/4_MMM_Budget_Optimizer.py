import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import Ridge

st.set_page_config(page_title="MMM Budget Optimizer", page_icon="🧠", layout="wide")

DATA_PATH = "data/mock_marketing_data.xlsx"


@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset at: {path}")
    df = pd.read_excel(path, sheet_name=0)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")

    for c in ["spend", "revenue"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in ["channel"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("Unknown").str.strip()

    return df


def adstock(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + alpha * carry
        out[i] = carry
    return out


def hill(x: np.ndarray, k: float, s: float) -> np.ndarray:
    x = np.clip(x, 0, None)
    return (x**s) / (x**s + k**s + 1e-9)


st.title("🧠 MMM-style Budget Optimizer")
st.caption("Lightweight MMM approximation for demo: adstock + saturation + regression on revenue.")

df = load_data()

if not set(["date", "channel", "spend", "revenue"]).issubset(df.columns):
    st.error("Your dataset must have columns: date, channel, spend, revenue")
    st.stop()

col1, col2, col3 = st.columns(3)
alpha = col1.slider("Adstock alpha", 0.0, 0.9, 0.5, 0.05)
ridge = col2.slider("Ridge strength", 0.1, 50.0, 5.0, 0.5)
pct_move = col3.slider("Budget to reallocate", 0.0, 0.4, 0.15, 0.05)

# Daily spend by channel
daily_spend = df.groupby(["date", "channel"], as_index=False)["spend"].sum()
revenue_by_date = df.groupby("date", as_index=False)["revenue"].sum().set_index("date")["revenue"]

spend_pivot = daily_spend.pivot_table(index="date", columns="channel", values="spend", aggfunc="sum").fillna(0).sort_index()
y = revenue_by_date.reindex(spend_pivot.index).fillna(0).values

channels = list(spend_pivot.columns)
X = []
for ch in channels:
    x = spend_pivot[ch].values.astype(float)
    x_ad = adstock(x, alpha=alpha)
    k = np.percentile(x_ad[x_ad > 0], 60) if np.any(x_ad > 0) else 1.0
    x_sat = hill(x_ad, k=k, s=1.5)
    X.append(x_sat)

X = np.vstack(X).T

model = Ridge(alpha=ridge, fit_intercept=True, random_state=42)
model.fit(X, y)
yhat = model.predict(X)

ss_res = float(np.sum((y - yhat) ** 2))
ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-9
r2 = 1 - ss_res / ss_tot

st.info(f"Model R² (in-sample): **{r2:.3f}** (demo metric)")

coefs = pd.DataFrame({"channel": channels, "coef": model.coef_}).sort_values("coef", ascending=False)

c1, c2 = st.columns(2)
fig1 = px.bar(coefs, x="channel", y="coef", title="Relative channel contribution weights (coef)")
fig1.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
c1.plotly_chart(fig1, use_container_width=True)

trend = pd.DataFrame({"date": spend_pivot.index, "actual_revenue": y, "pred_revenue": yhat})
fig2 = px.line(trend, x="date", y=["actual_revenue", "pred_revenue"], title="Revenue: actual vs predicted (MMM demo)")
fig2.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
c2.plotly_chart(fig2, use_container_width=True)

st.subheader("Budget reallocation recommendation (simple)")
spend_tot = df.groupby("channel")["spend"].sum().sort_values(ascending=False)
score = coefs.set_index("channel")["coef"]

total = float(spend_tot.sum())
move = total * float(pct_move)

# move from lowest coef → highest coef
losers = score.sort_values().index.tolist()
winners = score.sort_values(ascending=False).index.tolist()

plan = spend_tot.astype(float).copy()
remaining = move

for ch in losers:
    if remaining <= 0:
        break
    take = min(plan[ch], remaining)
    plan[ch] -= take
    remaining -= take

give_total = move - remaining
win_scores = score.loc[winners].clip(lower=0)
denom = float(win_scores.sum()) if float(win_scores.sum()) > 0 else len(winners)

for ch in winners:
    w = float(win_scores[ch]) / denom if denom else 1 / len(winners)
    plan[ch] += give_total * w

out = pd.DataFrame(
    {
        "channel": spend_tot.index,
        "current_spend": spend_tot.values,
        "recommended_spend": plan.reindex(spend_tot.index).values,
    }
)
out["delta"] = out["recommended_spend"] - out["current_spend"]
st.dataframe(out.sort_values("delta", ascending=False).round(2), use_container_width=True)
