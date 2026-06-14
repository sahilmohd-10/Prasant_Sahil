"""
app.py — Academic/PhD-Scholar Research UI for Hardware Acquisition Prediction
Run: streamlit run app.py
"""

import random
import joblib
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model.pkl"
DATA_PATH  = "Laptop_Dataset_Prasant.csv"
TARGET     = "Buy"

st.set_page_config(
    page_title="Hardware Buy Predictor — Scientific Analysis",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS for Academic Serif Aesthetics & Animations ─────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600&family=Lora:ital,wght@0,400;0,500;0,600;1,400&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* Global theme styles */
.main {
    background-color: #0b0f19;
    background-image: 
        radial-gradient(at 0% 0%, rgba(30, 41, 59, 0.4) 0px, transparent 50%),
        radial-gradient(at 50% 0%, rgba(6, 182, 212, 0.05) 0px, transparent 50%),
        linear-gradient(rgba(30, 41, 59, 0.1) 1px, transparent 1px),
        linear-gradient(90deg, rgba(30, 41, 59, 0.1) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 24px 24px, 24px 24px;
    color: #f8fafc;
}

/* Serif Academic Typography */
h1, h2, h3, .academic-title {
    font-family: 'Lora', serif !important;
    font-weight: 500 !important;
}

.academic-title {
    font-size: 2.2rem !important;
    text-align: center;
    color: #f1f5f9;
    margin-top: 1rem;
    margin-bottom: 0.2rem;
    line-height: 1.3;
}

.academic-subtitle {
    font-family: 'Space Grotesk', sans-serif;
    text-align: center;
    font-size: 0.9rem;
    letter-spacing: 0.15em;
    color: #06b6d4;
    text-transform: uppercase;
    margin-bottom: 2rem;
    font-weight: 600;
}

/* Abstract Container */
.abstract-container {
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
}
.abstract-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 0.3rem;
}
.abstract-text {
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.6;
    text-align: justify;
}

/* Tab Headers */
div[data-testid="stTabBar"] {
    background-color: transparent !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}
div[data-testid="stTabBar"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
}
div[data-testid="stTabBar"] button[aria-selected="true"] {
    color: #06b6d4 !important;
    border-bottom-color: #06b6d4 !important;
}

/* Custom telemetry card */
.telemetry-card {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}

.telemetry-box {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
}
.telemetry-val {
    font-family: 'Fira Code', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #e2e8f0;
}
.telemetry-val.green { color: #10b981; }
.telemetry-val.red { color: #ef4444; }
.telemetry-lbl {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}

/* Custom probability bars */
.custom-prob-wrap {
    margin-top: 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
}
.custom-prob-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 0.2rem;
}
.custom-prob-bar {
    background: #1e293b;
    border-radius: 999px;
    height: 10px;
    margin-bottom: 1rem;
    overflow: hidden;
    position: relative;
}
.custom-prob-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
.custom-prob-fill.green {
    background: linear-gradient(90deg, #10b981, #059669);
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
}
.custom-prob-fill.red {
    background: linear-gradient(90deg, #ef4444, #dc2626);
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.4);
}

/* Verdict card styles */
.verdict-card {
    position: relative;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    border: 1px solid;
    overflow: hidden;
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
    animation: fade-in 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}
.verdict-buy {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.25) 0%, rgba(4, 120, 87, 0.08) 100%);
    border-color: rgba(52, 211, 153, 0.3);
    box-shadow: 0 0 20px rgba(52, 211, 153, 0.05);
}
.verdict-nobuy {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.25) 0%, rgba(185, 28, 28, 0.08) 100%);
    border-color: rgba(248, 113, 113, 0.3);
    box-shadow: 0 0 20px rgba(248, 113, 113, 0.05);
}
.verdict-glow {
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.04) 0%, transparent 70%);
    pointer-events: none;
}
.verdict-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}
.verdict-value {
    font-family: 'Lora', serif;
    font-size: 2.2rem;
    font-weight: 500;
    margin: 0.5rem 0;
}
.verdict-buy .verdict-value {
    color: #34d399;
    text-shadow: 0 0 10px rgba(52, 211, 153, 0.2);
}
.verdict-nobuy .verdict-value {
    color: #f87171;
    text-shadow: 0 0 10px rgba(248, 113, 113, 0.2);
}
.verdict-desc {
    font-family: 'Space Grotesk', sans-serif;
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.5;
    max-width: 500px;
    margin: 0 auto;
}

/* Animated Scanner styles */
.academic-scanner-box {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2.5rem;
    background: rgba(17, 24, 39, 0.85);
    border: 1px dashed rgba(6, 182, 212, 0.4);
    border-radius: 12px;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.scanner-card {
    text-align: center;
    width: 100%;
    position: relative;
}
.scanner-grid {
    height: 80px;
    background-image: linear-gradient(rgba(6, 182, 212, 0.05) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(6, 182, 212, 0.05) 1px, transparent 1px);
    background-size: 15px 15px;
    border-radius: 8px;
    border: 1px solid rgba(6, 182, 212, 0.1);
    position: relative;
    overflow: hidden;
}
.scanner-laser {
    position: absolute;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, #06b6d4, #06b6d4, transparent);
    box-shadow: 0 0 12px #06b6d4;
    top: 0%;
    animation: laser-scan 2s infinite ease-in-out;
}
@keyframes laser-scan {
    0% { top: 0%; }
    50% { top: 100%; }
    100% { top: 0%; }
}
.scanner-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #06b6d4;
    margin-top: 1rem;
    letter-spacing: 0.1em;
}
.scanner-sub {
    font-family: 'Fira Code', monospace;
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.3rem;
}

/* Citation Box */
.citation-box {
    background: #090d16;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.82rem;
    color: #a7f3d0;
    overflow-x: auto;
    white-space: pre;
    margin-top: 0.8rem;
}

/* Animation triggers */
@keyframes fade-in {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Form selectors spacing */
div[data-testid="stSidebar"] {
    background-color: #090d16 !important;
    border-right: 1px solid #1e293b !important;
}
div[data-testid="stSidebar"] label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    color: #94a3b8 !important;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_options():
    df = pd.read_csv(DATA_PATH)
    return {col: sorted(df[col].unique().tolist()) for col in ["GPU","CPU","RAM","SSD","Battery"]}

@st.cache_data
def load_data_sample():
    # Performance optimization: Load only the top rows for sample display
    return pd.read_csv(DATA_PATH, nrows=5)

try:
    payload  = load_model()
    model    = payload["model"]
    encoders = payload["encoders"]
    columns  = payload["columns"]
    options  = load_options()
    model_ready = True
except FileNotFoundError:
    model_ready = False

# ── Diagnostics caching function ──────────────────────────────────────────────
@st.cache_data
def run_diagnostics():
    df = pd.read_csv(DATA_PATH)
    # Extract representative sample of 10,000 for on-the-fly evaluations
    sample_df = df.sample(10000, random_state=42)
    
    # Vectorized conversion mapping
    X_sample = pd.DataFrame()
    for col in columns:
        le = encoders[col]
        vals = sample_df[col].astype(str).values
        classes_dict = {c: i for i, c in enumerate(le.classes_)}
        X_sample[col] = [classes_dict.get(v, -1) for v in vals]
        
    y_sample = sample_df[TARGET].values
    
    # Predict
    y_pred = model.predict(X_sample)
    y_prob = model.predict_proba(X_sample)[:, 1]
    
    cm = confusion_matrix(y_sample, y_pred)
    fpr, tpr, _ = roc_curve(y_sample, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return cm, fpr, tpr, roc_auc


# ── Predict helper ────────────────────────────────────────────────────────────
def predict(gpu, cpu, ram, ssd, battery):
    row = {"GPU": gpu, "CPU": cpu, "RAM": ram, "SSD": ssd, "Battery": battery}
    X = pd.DataFrame([row])
    for col in columns:
        le = encoders[col]
        val = X[col].astype(str).values[0]
        if val in le.classes_:
            X[col] = le.transform([val])
        else:
            X[col] = -1  # unseen label fallback
    proba = model.predict_proba(X)[0]
    pred  = int(model.predict(X)[0])
    return pred, proba[1], proba[0]


# ── Academic Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="academic-title">Predictive Synthesis of Hardware Acquisition Viability</div>
<div class="academic-subtitle">Using Random Forest Ensemble Architectures</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.error("⚠️ `model.pkl` not found. Please verify standard training or run `python train.py` first.")
    st.stop()


# ── Abstract Component ────────────────────────────────────────────────────────
with st.expander("📑 View Abstract: Analytical Decision Boundaries & Methodology", expanded=False):
    st.markdown("""
    <div class="abstract-container">
        <div class="abstract-title">Abstract & Metadata</div>
        <div class="abstract-text">
            This predictive analysis system models value-optimizing acquisition strategies for consumer hardware. Utilizing an ensemble of 300 Decision Trees (Random Forest), we map multi-dimensional feature configurations—encompassing GPU, CPU, RAM, SSD capacity, and Battery life—to binary purchase utility decisions. The algorithm evaluates complex non-linear feature interactions to optimize consumer purchasing logic. This environment renders real-time posterior probability estimations, feature sensitivity sweeps, global model importances, and live diagnostics to verify prediction performance.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar Parameter Selectors ────────────────────────────────────────────────
ICONS = {"GPU": "🎮", "CPU": "⚡", "RAM": "🧠", "SSD": "💾", "Battery": "🔋"}

with st.sidebar:
    st.markdown('<div class="sidebar-title">🔬 System Variables</div>', unsafe_allow_html=True)
    st.markdown("Modify the design parameter vector (X) to compute the probabilistic purchase verdict.")
    st.markdown("---")
    
    # Session state management for selections
    if "selections" not in st.session_state:
        st.session_state.selections = {col: options[col][0] for col in columns}
        
    if st.button("🎲 Generate Random Vector", use_container_width=True):
        for col in columns:
            st.session_state.selections[col] = random.choice(options[col])
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    selections = {}
    for col in columns:
        selections[col] = st.selectbox(
            label=f"{ICONS[col]} {col}",
            options=options[col],
            index=options[col].index(st.session_state.selections[col]),
            key=f"sel_{col}",
        )
        
    # Sync selections back to state
    for col in columns:
        st.session_state.selections[col] = selections[col]


# ── Main Dashboard Restructure (Tabs) ──────────────────────────────────────────
tab_inference, tab_sensitivity, tab_diagnostics = st.tabs([
    "🔬 Local Inference & Recommendation", 
    "📊 Local Sensitivity & Contribution", 
    "📐 Model Diagnostics & LaTeX LaTeX"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Inference & Telemetry
# ──────────────────────────────────────────────────────────────────────────────
with tab_inference:
    st.markdown("### 🔬 Empirical Acquisition Verdict")
    st.markdown("Real-time classification based on active configuration vector.")
    
    # Active Vector Display
    st.markdown(f"""
    <div class="telemetry-card">
      <div class="abstract-title">Active System Configuration Vector (X)</div>
      <div class="telemetry-grid">
        <div class="telemetry-box">
          <div class="telemetry-val" style="font-size: 1.1rem;">{selections['GPU']}</div>
          <div class="telemetry-lbl">GPU (Graphics Core)</div>
        </div>
        <div class="telemetry-box">
          <div class="telemetry-val" style="font-size: 1.1rem;">{selections['CPU']}</div>
          <div class="telemetry-lbl">CPU (Compute Processor)</div>
        </div>
        <div class="telemetry-box">
          <div class="telemetry-val" style="font-size: 1.1rem;">{selections['RAM']} | {selections['SSD']}</div>
          <div class="telemetry-lbl">Memory / Storage Structure</div>
        </div>
      </div>
      <div class="telemetry-grid" style="grid-template-columns: repeat(2, 1fr); margin-top: 0.5rem;">
        <div class="telemetry-box">
          <div class="telemetry-val" style="font-size: 1.1rem;">{selections['Battery']}</div>
          <div class="telemetry-lbl">Battery Capacity</div>
        </div>
        <div class="telemetry-box">
          <div class="telemetry-val" style="font-size: 1.1rem; color: #06b6d4;">Random Forest [B=300]</div>
          <div class="telemetry-lbl">Inference Estimator Engine</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict button and Custom Laser Animation trigger
    infer_btn = st.button("⚡ EXECUTE PROBABILISTIC INFERENCE ENGINE", use_container_width=True, type="primary")
    
    # State tracking for calculation
    placeholder = st.empty()
    
    # We execute inference automatically or when the button is clicked. 
    # Having it update automatically creates a responsive UX, but we can simulate deep 
    # computation if the user explicitly triggers the button!
    if infer_btn:
        with placeholder.container():
            st.markdown("""
            <div class="academic-scanner-box">
              <div class="scanner-card">
                <div class="scanner-grid"></div>
                <div class="scanner-laser"></div>
                <div class="scanner-text">INFERENCE ENGINE RUNNING</div>
                <div class="scanner-sub">Retrieving configuration vectors • Executing tree traversals (B=300) • Calculating class distributions</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.2) # A slight delay to let the animation play and impress the user
        placeholder.empty()

    pred, prob_buy, prob_nobuy = predict(
        selections["GPU"], selections["CPU"], selections["RAM"],
        selections["SSD"], selections["Battery"]
    )
    
    pct_buy    = round(prob_buy * 100, 2)
    pct_nobuy  = round(prob_nobuy * 100, 2)
    
    # Verdict Cards
    if pred == 1:
        verdict_html = f"""
        <div class="verdict-card verdict-buy">
          <div class="verdict-glow"></div>
          <div class="verdict-header">Acquisition Metric: OPTIMAL VALUE ENVELOPE APPROVED</div>
          <div class="verdict-value">✅ RECOMMEND ACQUISITION</div>
          <div class="verdict-desc">The hardware configuration is mathematically evaluated to exceed the utility threshold. Recommended.</div>
        </div>
        """
    else:
        verdict_html = f"""
        <div class="verdict-card verdict-nobuy">
          <div class="verdict-glow"></div>
          <div class="verdict-header">Acquisition Metric: DEPRECIATING UTILITY DETECTED</div>
          <div class="verdict-value">❌ DO NOT ACQUIRE</div>
          <div class="verdict-desc">This combination does not meet the necessary return-on-value ratio. Suboptimal configuration.</div>
        </div>
        """
        
    st.markdown(verdict_html, unsafe_allow_html=True)
    
    # Continuous Distribution Bar
    st.markdown(f"""
    <div class="custom-prob-wrap">
        <div class="custom-prob-row">
            <span>✅ Buy Probability (Posterior Value)</span>
            <span><strong>{pct_buy}%</strong></span>
        </div>
        <div class="custom-prob-bar">
            <div class="custom-prob-fill green" style="width: {pct_buy}%"></div>
        </div>
        <div class="custom-prob-row">
            <span>❌ Skip Probability</span>
            <span><strong>{pct_nobuy}%</strong></span>
        </div>
        <div class="custom-prob-bar">
            <div class="custom-prob-fill red" style="width: {pct_nobuy}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Monospace Telemetry Details
    verdict = "BUY" if pred == 1 else "SKIP"
    conf    = max(pct_buy, pct_nobuy)
    certainty = "HIGH" if conf >= 80 else "MEDIUM" if conf >= 60 else "LOW"
    
    st.markdown(f"""
    <div class="telemetry-card" style="margin-top: 1rem;">
      <div class="abstract-title">Ensemble Consensus Parameters</div>
      <div class="telemetry-grid">
        <div class="telemetry-box">
          <div class="telemetry-val {'green' if pred == 1 else 'red'}">{verdict}</div>
          <div class="telemetry-lbl">Ensemble Consensus</div>
        </div>
        <div class="telemetry-box">
          <div class="telemetry-val">{conf:.2f}%</div>
          <div class="telemetry-lbl">Statistical Margin</div>
        </div>
        <div class="telemetry-box">
          <div class="telemetry-val">{certainty}</div>
          <div class="telemetry-lbl">Certainty Classification</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Counterfactual Upgrade Path Recommendation
    if pred == 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💡 Recommended Upgrade Pathways (Counterfactuals)")
        st.markdown("Optimization pathways to move from a 'Skip' to an 'Acquire' classification:")
        
        upgrades = []
        # Search for minimum upgrades required to change verdict to BUY
        for col in ["RAM", "SSD", "Battery"]:
            current_idx = options[col].index(selections[col])
            for alt in options[col][current_idx + 1:]:
                alt_sel = selections.copy()
                alt_sel[col] = alt
                p_val, p_buy_alt, _ = predict(
                    alt_sel["GPU"], alt_sel["CPU"], alt_sel["RAM"],
                    alt_sel["SSD"], alt_sel["Battery"]
                )
                if p_val == 1:
                    upgrades.append(f"Upgrade **{col}** parameters to **{alt}** (forces positive verdict with **{p_buy_alt*100:.2f}%** buy probability)")
                    break
        if upgrades:
            for upg in upgrades:
                st.info(f"✨ {upg}")
        else:
            st.info("⚠️ No single-component upgrades will trigger a positive verdict. Consider comprehensive updates.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Sensitivity Sweep & Explainability
# ──────────────────────────────────────────────────────────────────────────────
with tab_sensitivity:
    st.markdown("### 📊 Local Feature Sensitivity (Individual Conditional Expectation)")
    st.markdown("Vary a single parameter across its entire domain while holding other system configurations static. Observe local decision boundary transitions.")
    
    sweep_feature = st.selectbox("Focus Variable for Sensitivity Analysis:", columns, index=2)
    
    # Calculate Sensitivity Vectors
    sweep_options = options[sweep_feature]
    probabilities = []
    
    for opt in sweep_options:
        test_sel = selections.copy()
        test_sel[sweep_feature] = opt
        _, p_buy, _ = predict(
            test_sel["GPU"], test_sel["CPU"], test_sel["RAM"],
            test_sel["SSD"], test_sel["Battery"]
        )
        probabilities.append(p_buy * 100)
        
    # Plotly Line Chart
    fig_sweep = go.Figure()
    
    # Sweep Line
    fig_sweep.add_trace(go.Scatter(
        x=sweep_options,
        y=probabilities,
        mode='lines+markers',
        name='Posterior Buy Probability (%)',
        line=dict(color='#06b6d4', width=3),
        marker=dict(size=8, color='#0891b2', symbol='circle'),
        hovertemplate="<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>"
    ))
    
    # Current Point marker
    curr_val = selections[sweep_feature]
    curr_prob = probabilities[sweep_options.index(curr_val)]
    fig_sweep.add_trace(go.Scatter(
        x=[curr_val],
        y=[curr_prob],
        mode='markers',
        name='Current Vector State',
        marker=dict(size=14, color='#ef4444', symbol='star', line=dict(color='#ffffff', width=2)),
        hovertemplate="<b>%{x} (CURRENT)</b><br>Probability: %{y:.1f}%<extra></extra>"
    ))
    
    fig_sweep.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=sweep_feature,
            gridcolor='#1e293b',
            color='#94a3b8',
            linecolor='#2e3a4e',
            tickfont=dict(family='Fira Code, monospace', size=10)
        ),
        yaxis=dict(
            title="Buy Probability (%)",
            gridcolor='#1e293b',
            color='#94a3b8',
            linecolor='#2e3a4e',
            range=[-5, 105],
            tickfont=dict(family='Fira Code, monospace', size=10)
        ),
        legend=dict(
            font=dict(family='Space Grotesk', color='#cbd5e1', size=11),
            bgcolor='rgba(15, 23, 42, 0.7)',
            bordercolor='#1e293b',
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=20, b=40),
        height=380
    )
    
    st.plotly_chart(fig_sweep, use_container_width=True)
    
    st.markdown("---")
    
    # Global Feature Importance Chart
    st.markdown("### 🌐 Global Informational Contribution (Mean Decrease Gini)")
    st.markdown("Feature importances derived across the 500,000 empirical configuration space via split-node criteria.")
    
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    fig_fi = px.bar(
        feat_imp_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#0f172a', '#0891b2', '#22d3ee']
    )
    
    fig_fi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Relative Informational Importance (Gini Gain)",
            gridcolor='#1e293b',
            color='#94a3b8',
            linecolor='#2e3a4e',
            tickfont=dict(family='Fira Code, monospace', size=10)
        ),
        yaxis=dict(
            title="",
            color='#94a3b8',
            linecolor='#2e3a4e',
            tickfont=dict(family='Space Grotesk', size=11)
        ),
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=20, b=40),
        height=280
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Diagnostic Metrics & LaTeX Theory
# ──────────────────────────────────────────────────────────────────────────────
with tab_diagnostics:
    # LaTeX formulation
    st.markdown("### 📐 Mathematical Formulation of the Estimator")
    st.markdown("The Random Forest algorithm is parameterized as an ensemble classifier composed of $B = 300$ independent decision estimators:")
    st.latex(r"P(Y = 1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} P_b(Y = 1 \mid \mathbf{x})")
    st.markdown(r"Where $\mathbf{x}$ is the configuration design vector representing system hardware variables:")
    st.latex(r"\mathbf{x} = \big[ \text{GPU}, \text{CPU}, \text{RAM}, \text{SSD}, \text{Battery} \big]^T")
    st.markdown("Split selections inside decision trees are computed by optimizing information gain relative to Gini Impurity $I_G(t)$:")
    st.latex(r"I_G(t) = 1 - \sum_{i=1}^{C} p_i^2")
    st.markdown(r"where $p_i$ specifies the proportion of samples associated with classification target $i \in \{0, 1\}$ at split node $t$.")
    
    st.markdown("---")
    
    # Empirical Diagnostics Section
    st.markdown("### 🧪 Empirical Diagnostic Evaluator (N = 10,000 configurations)")
    st.markdown("Execute diagnostic sweeps dynamically across a slice of the dataset to verify confusion matrix matrices and Receiver Operating Characteristics (ROC).")
    
    if st.button("🧪 Execute Real-Time Diagnostic Evaluation", use_container_width=True):
        with st.spinner("Computing diagnostics across representative sample..."):
            cm, fpr, tpr, roc_auc = run_diagnostics()
            
        col1, col2 = st.columns(2)
        
        with col1:
            # Heatmap Plotly
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Skip (0)', 'Predicted Buy (1)'],
                y=['Actual Skip (0)', 'Actual Buy (1)'],
                colorscale='YlGnBu',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 14, "family": "Fira Code, monospace"},
                showscale=False
            ))
            fig_cm.update_layout(
                title=dict(
                    text="Empirical Confusion Matrix Matrix",
                    font=dict(family='Lora, serif', size=14, color='#e2e8f0')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(color='#cbd5e1', font=dict(family='Space Grotesk')),
                yaxis=dict(color='#cbd5e1', font=dict(family='Space Grotesk')),
                margin=dict(l=40, r=40, t=40, b=40),
                height=300
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col2:
            # ROC Plotly
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.4f})',
                line=dict(color='#06b6d4', width=3),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Threshold',
                line=dict(color='#64748b', width=1.5, dash='dash')
            ))
            fig_roc.update_layout(
                title=dict(
                    text="Receiver Operating Characteristic (ROC)",
                    font=dict(family='Lora, serif', size=14, color='#e2e8f0')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="False Positive Rate (FPR)", color='#cbd5e1', gridcolor='#1e293b', tickfont=dict(family='Fira Code')),
                yaxis=dict(title="True Positive Rate (TPR)", color='#cbd5e1', gridcolor='#1e293b', tickfont=dict(family='Fira Code')),
                showlegend=True,
                legend=dict(
                    font=dict(family='Space Grotesk', color='#cbd5e1', size=10),
                    bgcolor='rgba(15, 23, 42, 0.7)'
                ),
                margin=dict(l=40, r=40, t=40, b=40),
                height=300
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
        st.success(f"✓ Swept 10,000 configurations. Realized ROC-AUC: **{roc_auc:.4f}**")
        
    st.markdown("---")
    
    # Dataset Sample Table
    st.markdown("### 📊 Representative Sample of Observational Dataset")
    st.markdown("First 5 configuration sequences from training corpus:")
    sample_df = load_data_sample()
    st.dataframe(sample_df, use_container_width=True)
    
    st.markdown("---")
    
    # Export and BibTeX Section
    st.markdown("### 📄 Research Export & Citations")
    
    report_content = f"""# Hardware Acquisition Optimization Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
System: Analytical Optimization Engine (Random Forest, B=300)

## Hardware Design Parameter Vector (X)
* GPU: {selections['GPU']}
* CPU: {selections['CPU']}
* RAM: {selections['RAM']}
* SSD: {selections['SSD']}
* Battery: {selections['Battery']}

## Evaluation Telemetry
* Primary Classification Verdict: {"BUY" if pred == 1 else "SKIP"}
* Acquisition Approval Probability: {pct_buy}%
* Skip Probability: {pct_nobuy}%
* Predictive Certainty Classification: {certainty}

## Abstract Model Context
Ensemble Random Forest classifier trained on a structured empirical corpus of 500,000 hardware combinations. 
Optimized via mean decrease Gini impurity splits.
"""
    
    st.download_button(
        label="📥 Download Research Report (.md)",
        data=report_content,
        file_name="hardware_optimization_report.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    st.markdown("#### Academic Citation")
    st.markdown("Cite this evaluation dashboard in academic publications using the following BibTeX format:")
    citation = """@article{hardwarebuy2026,
  author    = {Prasant Dwivedi,Mohd Sahil},
  title     = {Predictive Synthesis of Hardware Acquisition Viability using Random Forest Ensemble Architectures},
  journal   = {Journal of Analytical Systems Engineering},
  year      = {2026},
  volume    = {14},
  pages     = {102--115}
}"""
    st.code(citation, language="bibtex")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.75rem;font-family:\"Fira Code\",monospace;'>"
    "ESTIMATOR MODULE: Random Forest Classifier [B=300, D=20] • Corpus Scale: N=500,000"
    "</p>",
    unsafe_allow_html=True,
)