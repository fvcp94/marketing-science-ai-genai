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
        raise FileNotFoundError(f"Missing dataset at: {path}. Make sure it is committed to GitHub.")
    df = pd.read_excel(path, sheet_name=0)

    # Standardize common columns (safe if missing)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    num_cols = ["impressions", "clicks", "spend", "mqls", "sqls", "pipeline_value", "closed_won", "revenue"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    cat_cols = ["channel", "campaign", "region", "segment", "utm_source"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("Unknown").str.strip()

    return df


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    return (s - med) / (1.4826 * mad + 1e-9)


st.title("🧪 Data Health & Trust Layer")
st.caption("Stakeholder-friendly QA checks to validate reliability before dashboards, forecasting, MMM, or GenAI insights.")

try:
    df = load_data()
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

# ---------- Top Summary Cards ----------
rows = len(df)
cols = len(df.columns)
date_min = df["date"].min() if "date" in df.columns else None
date_max = df["date"].max() if "date" in df.columns else None
date_coverage = (date_max - date_min).days if (date_min is not None and date_max is not None and pd.notna(date_min) and pd.notna(date_max)) else None

missing_cells = int(df.isna().sum().sum())
missing_pct = float(df.isna().mean().mean() * 100)

dupe_count = int(df.duplicated().sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{rows:,}")
c2.metric("Columns", f"{cols:,}")
c3.metric("Missing cells", f"{missing_cells:,}")
c4.metric("Avg missing %", f"{missing_pct:.2f}%")
c5.metric("Duplicate rows", f"{dupe_count:,}")

st.divider()

tab1, tab2, tab3 = st.tabs(["✅ QA Checklist", "📉 Missingness", "🚨 Outliers & Anomalies"])

# ---------- TAB 1: Checklist ----------
with tab1:
    st.subheader("QA Checklist (what a Marketing Science team expects)")

    # Basic checks
    checks = []
    checks.append(("Dataset loads successfully", True))

    # Date checks
    if "date" in df.columns:
        checks.append(("Date column present", True))
        checks.append(("Date has non-null values", df["date"].notna().any()))
        if date_coverage is not None:
            checks.append(("Date coverage ≥ 30 days", date_coverage >= 30))
    else:
        checks.append(("Date column present", False))

    # Required-ish columns (soft validation)
    expected = ["channel", "spend", "revenue"]
    for col in expected:
        checks.append((f"Column present: {col}", col in df.columns))

    # Non-negative metrics checks
    nonneg_cols = [c for c in ["impressions", "clicks", "spend", "mqls", "sqls", "revenue", "pipeline_value"] if c in df.columns]
    for c in nonneg_cols:
        checks.append((f"{c} is non-negative", (df[c] >= 0).all()))

    # KPI sanity checks
    tmp = df.copy()
    if set(["clicks", "impressions"]).issubset(tmp.columns):
        tmp["ctr"] = safe_div(tmp["clicks"], tmp["impressions"])
        ctr_bad = int((tmp["ctr"] > 0.25).sum())
    else:
        ctr_bad = None

    if set(["spend", "clicks"]).issubset(tmp.columns):
        tmp["cpc"] = safe_div(tmp["spend"], tmp["clicks"])
        cpc_bad = int((tmp["cpc"] > 50).sum())
    else:
        cpc_bad = None

    if set(["spend", "mqls"]).issubset(tmp.columns):
        tmp["cpl"] = safe_div(tmp["spend"], tmp["mqls"])
        cpl_bad = int((tmp["cpl"] > 500).sum())
    else:
        cpl_bad = None

    # Show checklist
    ok = sum(1 for _, v in checks if v is True)
    total = len(checks)
    st.progress(ok / total if total else 1.0)

    table = pd.DataFrame(checks, columns=["Check", "Pass"])
    table["Pass"] = table["Pass"].map(lambda x: "✅" if x else "⚠️")
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("KPI sanity flags (not errors — just investigate)")
    a, b, c = st.columns(3)
    a.metric("CTR > 25% rows", "N/A" if ctr_bad is None else f"{ctr_bad:,}")
    b.metric("CPC > $50 rows", "N/A" if cpc_bad is None else f"{cpc_bad:,}")
    c.metric("CPL > $500 rows", "N/A" if cpl_bad is None else f"{cpl_bad:,}")

    st.info(
        "Why this matters: stakeholders trust forecasts/MMM more when you show a repeatable QA layer. "
        "These checks are fast, interpretable, and easy to operationalize."
    )

# ---------- TAB 2: Missingness ----------
with tab2:
    st.subheader("Missingness by column")

    miss = (df.isna().mean().sort_values(ascending=False) * 100).reset_index()
    miss.columns = ["column", "pct_missing"]

    colA, colB = st.columns([1.2, 1])
    with colA:
        fig = px.bar(miss, x="column", y="pct_missing", title="Percent missing by column")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("### Data coverage")
        if "date" in df.columns and pd.notna(date_min) and pd.notna(date_max):
            st.write(f"**From:** {date_min.date()}  \n**To:** {date_max.date()}  \n**Days:** {date_coverage:,}")
        else:
            st.write("Date coverage: **N/A** (no date column or invalid dates)")

        st.markdown("### Quick actions")
        st.write("- Prioritize fixes for columns with high missingness that are used in KPIs/models.")
        st.write("- For categorical columns: consider mapping null → `Unknown`.")
        st.write("- For numeric columns: decide between impute vs drop vs treat as 0 based on semantics.")

    st.subheader("Missingness heatmap (top 25 columns)")
    # Heatmap using a sample to avoid rendering huge data
    show_cols = miss["column"].head(25).tolist()
    sample = df[show_cols].head(300).isna().astype(int)
    heat = px.imshow(sample.T, aspect="auto", title="Missingness heatmap (1 = missing)")
    heat.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(heat, use_container_width=True)

# ---------- TAB 3: Outliers ----------
with tab3:
    st.subheader("Outlier candidates (robust z-score)")
    st.caption("Flags records worth investigating — not automatically incorrect.")

    metric_options = [c for c in ["spend", "clicks", "mqls", "sqls", "revenue", "pipeline_value"] if c in df.columns]
    if not metric_options:
        st.warning("No numeric metrics found for outlier detection.")
        st.stop()

    left, right = st.columns([1, 1])
    metric = left.selectbox("Metric", metric_options, index=0)
    threshold = right.slider("Robust z-score threshold", 3.0, 10.0, 6.0, 0.5)

    x_cols = [c for c in ["date", "channel", "campaign", "region", "segment"] if c in df.columns] + [metric]
    x = df[x_cols].copy()
    x[metric] = pd.to_numeric(x[metric], errors="coerce").fillna(0)

    if "channel" in x.columns:
        x["rz"] = x.groupby("channel")[metric].transform(robust_z)
    else:
        x["rz"] = robust_z(x[metric])

    out = x.loc[x["rz"].abs() >= threshold].copy()
    out["abs_rz"] = out["rz"].abs()
    out = out.sort_values(["abs_rz", metric], ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Outlier rows", f"{len(out):,}")
    k2.metric("Max robust z", f"{out['abs_rz'].max():.2f}" if len(out) else "0.00")
    k3.metric("Metric total", f"{x[metric].sum():,.0f}")

    st.dataframe(out.head(250), use_container_width=True)

    st.subheader("Distribution view (helps explain to stakeholders)")
    if "channel" in df.columns:
        top_channels = df.groupby("channel")[metric].sum().sort_values(ascending=False).head(8).index.tolist()
        dist_df = df[df["channel"].isin(top_channels)][["channel", metric]].copy()
        fig2 = px.box(dist_df, x="channel", y=metric, title=f"{metric} distribution for top channels")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        fig2 = px.histogram(df, x=metric, nbins=50, title=f"{metric} distribution")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, use_container_width=True)

with st.expander("Preview data"):
    st.dataframe(df.head(100), use_container_width=True)
