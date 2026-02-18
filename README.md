# 📈 Marketing Science AI + GenAI Platform

An end-to-end **Marketing Data Science + Generative AI decision intelligence platform** that transforms campaign data into predictive insights, budget optimization strategies, and executive-ready narratives.

Built to simulate how modern **Marketing Science / Growth Analytics teams** operate — combining deterministic modeling with GenAI storytelling for stakeholder decision support.

---

## 🚀 Platform Overview

This platform answers high-impact marketing leadership questions:

- Which channels drive pipeline and revenue?
- Where is funnel leakage happening?
- How should we reallocate budget for higher ROI?
- What revenue can we forecast?
- What actions should we take next?
- What changed vs last period — and why?

---

## 🧩 Core Modules

### 1️⃣ Data Health & Trust Layer

Stakeholder-friendly QA before analytics or modeling.

**Capabilities**

- Missingness detection  
- KPI sanity validation (CTR, CPC, CPL)  
- Duplicate checks  
- Date coverage validation  
- Outlier detection (robust z-score)  
- Channel distribution diagnostics  

**Why it matters**

Ensures forecasting, MMM, and GenAI insights are built on reliable data.

---

### 2️⃣ Performance Intelligence Dashboard

Executive view of marketing performance.

**Analytics included**

- Spend vs Revenue trends  
- Funnel conversion tracking  
- ROAS / CPL / CPSQL metrics  
- Channel performance benchmarking  
- Region & segment breakdowns  
- Time-window filtering  

---

### 3️⃣ Predictive Growth Forecasting

Machine-learning pipeline to anticipate demand and pipeline generation.

**Modeling approach**

- Time-split training  
- Gradient boosting regression  
- Seasonality features  
- Funnel lag signals  
- Baseline vs model benchmarking  

**Outputs**

- Revenue forecasts  
- MQL / SQL predictions  
- Forecast accuracy metrics  
- Error diagnostics  

---

### 4️⃣ Marketing Mix Modeling (MMM)

Channel contribution and diminishing-returns modeling.

**Techniques**

- Adstock transformations  
- Saturation curves  
- Ridge regression attribution  
- Channel ROI estimation  

**Business outcomes**

- Contribution weighting  
- Budget efficiency diagnostics  
- Scenario planning  

---

### 5️⃣ Budget Optimization Engine

Simulates strategic reallocation scenarios.

**Capabilities**

- Identifies over-invested channels  
- Detects underfunded high-ROI channels  
- Recommends reallocation %  
- Quantifies expected impact  

---

### 6️⃣ GenAI Stakeholder Insights ✨

Natural-language marketing intelligence powered by free LLMs.

Stakeholders can ask:

- “What changed vs last period?”
- “Where should we reallocate budget?”
- “Which channels are inefficient?”
- “What actions improve ROI fastest?”

**Features**

- Preset executive questions  
- Custom Q&A chat  
- Executive summary format  
- Action recommendations  
- Risk caveats  
- Funnel diagnostics  

**Safety guardrails**

- Aggregated metrics only  
- No row-level exposure  
- Privacy-safe storytelling  

---

## 🧠 Example GenAI Output Structure

Every answer is formatted for leadership:

```
What happened  
Why it happened  
What to do next (3 actions)  
Risks / caveats  
```

Designed for VP / Director consumption — not analysts.

---

## 🛠️ Tech Stack

| Layer | Tools |
|------|------|
| App Framework | Streamlit |
| Data | Pandas, NumPy |
| Visualization | Plotly |
| Modeling | Scikit-learn |
| MMM | Ridge + Adstock |
| GenAI | OpenRouter (Free LLMs) |
| Deployment | Streamlit Cloud |

---

## 📊 Dataset Scope

Mock marketing dataset simulating:

- Multi-channel campaigns  
- Funnel progression  
- Pipeline value  
- Revenue attribution  
- Regional segmentation  
- Time-series spend signals  

---

## ☁️ Live Deployment

Deployed via Streamlit Cloud.

**Main entry point**

```
streamlit_app/Home.py
```

---

## 🔑 GenAI Setup

Add secret in Streamlit Cloud:

```toml
OPENROUTER_API_KEY = "your_key"
```

Supports free models like:

- Mistral 7B  
- Llama 3  
- Gemma  

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app/Home.py
```

---

## 📁 Project Structure

```
marketing-science-ai-genai/
│
├── streamlit_app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Data_Health.py
│       ├── 2_Performance_Dashboard.py
│       ├── 3_Predictive_Growth.py
│       ├── 4_MMM_Budget_Optimizer.py
│       └── 5_GenAI_Insights.py
│
├── data/
│   └── mock_marketing_data.xlsx
│
├── requirements.txt
└── README.md
```

---

## 💼 Business Impact

This platform demonstrates how marketing science teams:

- Move beyond dashboards → decision systems  
- Combine ML + MMM + GenAI  
- Translate data → strategy  
- Support executive planning  

---

## 🔮 Future Enhancements

Planned roadmap:

- Bayesian MMM  
- Customer LTV modeling  
- CAC payback forecasting  
- Incrementality experiments  
- Multi-touch attribution  
- Scenario simulation engine  
- Agentic marketing copilots  

---

## 👤 Author

**Febin Varghese**  
Marketing Data Scientist | AI/GenAI Analytics | Growth Modeling

---

## ⭐ If you find this useful

Star the repo ⭐ — helps others discover Marketing Science AI applications.
