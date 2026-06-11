import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAI Diagnostics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

  /* ── Root palette ── */
  :root {
    --bg:        #050d1a;
    --surface:   #0a1628;
    --surface2:  #0f2040;
    --border:    #1a3a6e;
    --accent:    #00d4ff;
    --accent2:   #7b61ff;
    --green:     #00e5a0;
    --red:       #ff4d6d;
    --yellow:    #ffd166;
    --text:      #e8f4ff;
    --muted:     #5a7fa8;
  }

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] * { color: var(--text) !important; }

  /* ── Headers ── */
  h1 { 
    font-size: 2rem !important; 
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  h2, h3 { color: var(--accent) !important; }

  /* ── Cards ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }

  /* ── Result boxes ── */
  .result-positive {
    background: linear-gradient(135deg, rgba(255,77,109,0.15), rgba(255,77,109,0.05));
    border: 1px solid var(--red);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
  }
  .result-negative {
    background: linear-gradient(135deg, rgba(0,229,160,0.15), rgba(0,229,160,0.05));
    border: 1px solid var(--green);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
  }
  .result-label {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 1px;
  }
  .result-sub {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 0.3rem;
  }

  /* ── Probability bar ── */
  .prob-bar-wrap { margin: 0.4rem 0; }
  .prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    margin-bottom: 2px;
    color: var(--muted);
  }
  .prob-bar-bg {
    background: var(--surface2);
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
  }

  /* ── Inputs ── */
  .stSlider > div > div { background: var(--border) !important; }
  .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
  }
  .stNumberInput input, .stSelectbox select {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
  }
  div[data-testid="stNumberInputContainer"] input {
    background: var(--surface2) !important;
    color: var(--text) !important;
  }

  /* ── Button ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #050d1a !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--accent) !important;
  }

  /* ── Metric ── */
  [data-testid="metric-container"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
  }
  [data-testid="metric-container"] label { color: var(--muted) !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Divider ── */
  hr { border-color: var(--border) !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

  /* ── Tag chip ── */
  .chip {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: var(--muted);
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load & prepare models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    # Load base models
    nb_model   = joblib.load("naive_bayes_model.pkl")
    diab_model = joblib.load("diabetes_model.pkl")

    # ── Fit NB model on Iris dataset ──
    iris = load_iris()
    nb_model.fit(iris.data, iris.target)
    iris_features = list(iris.feature_names)
    iris_classes  = list(iris.target_names)
    iris_means    = pd.DataFrame(iris.data, columns=iris_features).groupby(iris.target).mean()

    # ── Fit Diabetes model on Pima Indians Diabetes data ──
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    # Embedded minimal representative data (100 rows from Pima dataset)
    data_vals = [
        [6,148,72,35,0,33.6,0.627,50,1],[1,85,66,29,0,26.6,0.351,31,0],
        [8,183,64,0,0,23.3,0.672,32,1],[1,89,66,23,94,28.1,0.167,21,0],
        [0,137,40,35,168,43.1,2.288,33,1],[5,116,74,0,0,25.6,0.201,30,0],
        [3,78,50,32,88,31.0,0.248,26,1],[10,115,0,0,0,35.3,0.134,29,0],
        [2,197,70,45,543,30.5,0.158,53,1],[8,125,96,0,0,0.0,0.232,54,1],
        [4,110,92,0,0,37.6,0.191,30,0],[10,168,74,0,0,38.0,0.537,34,1],
        [10,139,80,0,0,27.1,1.441,57,0],[1,189,60,23,846,30.1,0.398,59,1],
        [5,166,72,19,175,25.8,0.587,51,1],[7,100,0,0,0,30.0,0.484,32,1],
        [0,118,84,47,230,45.8,0.551,31,1],[7,107,74,0,0,29.6,0.254,31,1],
        [1,103,30,38,83,43.3,0.183,33,0],[1,115,70,30,96,34.6,0.529,32,1],
        [3,126,88,41,235,39.3,0.704,27,0],[8,99,84,0,0,35.4,0.388,50,0],
        [7,196,90,0,0,39.8,0.451,41,1],[9,119,80,35,0,29.0,0.263,29,1],
        [11,143,94,33,146,36.6,0.254,51,1],[10,125,70,26,115,31.1,0.205,41,1],
        [7,147,76,0,0,39.4,0.257,43,1],[1,97,66,15,140,23.2,0.487,22,0],
        [13,145,82,19,110,22.2,0.245,57,0],[5,117,92,0,0,34.1,0.337,38,0],
        [5,109,75,26,0,36.0,0.546,60,0],[3,158,76,36,245,31.6,0.851,28,1],
        [3,88,58,11,54,24.8,0.267,22,0],[6,92,92,0,0,19.9,0.188,28,0],
        [10,122,78,31,0,27.6,0.512,45,0],[4,103,60,33,192,24.0,0.966,33,0],
        [11,138,76,0,0,33.2,0.420,35,0],[9,102,76,37,0,32.9,0.665,46,1],
        [2,90,68,42,0,38.2,0.503,27,1],[4,111,72,47,207,37.1,1.390,56,1],
        [3,180,64,25,70,34.0,0.271,26,0],[7,133,84,0,0,40.2,0.696,37,0],
        [7,106,92,18,0,22.7,0.235,48,0],[9,171,110,24,240,45.4,0.721,54,1],
        [7,159,64,0,0,27.4,0.294,40,0],[0,180,66,39,0,42.0,1.893,25,1],
        [1,146,56,0,0,29.7,0.564,29,0],[2,71,70,27,0,28.0,0.586,22,0],
        [7,103,66,32,0,39.1,0.344,31,1],[7,105,0,0,0,0.0,0.305,24,0],
        [1,103,80,11,82,19.4,0.491,22,0],[1,101,50,15,36,24.2,0.526,26,0],
        [5,88,66,21,23,24.4,0.342,30,0],[8,176,90,34,300,33.7,0.467,58,1],
        [7,150,66,42,342,34.7,0.718,42,0],[1,73,50,10,0,23.0,0.248,21,0],
        [7,187,68,39,304,37.7,0.254,41,1],[0,100,88,60,110,46.8,0.962,31,0],
        [0,146,82,0,0,40.5,1.781,44,0],[0,105,64,41,142,41.5,0.173,22,0],
        [2,84,0,0,0,0.0,0.304,21,0],[8,133,72,0,0,32.9,0.270,39,1],
        [5,44,62,0,0,25.0,0.587,36,0],[2,141,58,34,128,25.4,0.699,24,0],
        [7,114,66,0,0,32.8,0.258,42,1],[5,99,74,27,0,29.0,0.203,32,0],
        [0,109,88,30,0,32.5,0.855,38,1],[2,109,92,0,0,42.7,0.845,54,0],
        [1,95,66,13,38,19.6,0.334,25,0],[4,146,85,27,100,28.9,0.189,27,0],
        [2,100,66,20,90,32.9,0.867,28,1],[5,139,64,35,140,28.6,0.411,26,0],
        [13,126,90,0,0,43.4,0.583,42,1],[2,74,0,0,0,0.0,0.102,22,0],
        [7,83,78,26,71,29.3,0.767,36,0],[0,101,65,28,0,24.6,0.237,22,0],
        [5,137,108,0,0,48.8,0.227,37,1],[2,110,74,29,125,32.4,0.698,27,0],
        [13,106,70,0,0,34.2,0.251,52,0],[2,100,68,25,71,38.5,0.324,26,0],
        [15,136,70,32,110,37.1,0.153,43,1],[1,107,68,19,0,26.5,0.165,24,0],
        [1,80,55,0,0,19.1,0.258,21,0],[4,123,80,15,176,32.0,0.443,34,0],
        [7,81,78,40,48,46.7,0.261,42,0],[4,134,72,0,0,23.8,0.277,60,1],
        [2,142,82,18,64,24.7,0.761,21,0],[6,144,72,27,228,33.9,0.255,40,0],
        [2,92,62,28,0,31.6,0.130,24,0],[1,71,48,18,76,20.4,0.323,22,0],
        [6,93,50,30,64,28.7,0.356,23,0],[1,122,90,51,220,49.7,0.325,31,1],
        [1,163,72,0,0,39.0,1.222,33,1],[1,151,60,0,0,26.1,0.179,22,0],
        [0,125,96,0,0,22.5,0.262,21,0],[1,81,72,18,40,26.6,0.335,22,0],
        [2,85,65,0,0,39.6,0.930,27,0],[1,126,56,29,152,28.7,0.801,21,0],
        [1,96,122,0,0,22.4,0.207,27,0],[4,144,58,28,140,29.5,0.287,37,0],
    ]
    df = pd.DataFrame(data_vals, columns=cols)
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values
    diab_model.fit(X, y)
    diab_features = cols[:-1]
    diab_classes  = ["No Diabetes", "Diabetes"]

    return nb_model, diab_model, iris_features, iris_classes, diab_features, diab_classes


nb_model, diab_model, iris_features, iris_classes, diab_features, diab_classes = load_models()


# ─── Helper: probability bar ─────────────────────────────────────────────────
def prob_bar(label, value, color):
    pct = f"{value*100:.1f}%"
    st.markdown(f"""
    <div class="prob-bar-wrap">
      <div class="prob-label"><span>{label}</span><span>{pct}</span></div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:{pct};background:{color};"></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 MedAI")
    st.markdown("**Diagnostic Intelligence Platform**")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", ["🌸 Iris Classifier", "💉 Diabetes Risk", "ℹ️ About Models"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Model Registry")
    st.markdown('<span class="chip">GaussianNB</span><span class="chip">sklearn</span><span class="chip">v1.6.1</span>', unsafe_allow_html=True)
    st.markdown("**naive_bayes_model.pkl**")
    st.caption("Iris Species Classification")
    st.markdown("**diabetes_model.pkl**")
    st.caption("Diabetes Risk Assessment")
    st.markdown("---")
    st.caption("Built with Streamlit • sklearn")


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — Iris Classifier
# ═══════════════════════════════════════════════════════════════
if page == "🌸 Iris Classifier":
    st.markdown("# 🌸 Iris Species Classifier")
    st.markdown("Enter flower measurements to classify the iris species using the Gaussian Naïve Bayes model.")
    st.markdown("---")

    col_form, col_result = st.columns([1.2, 1], gap="large")

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📐 Flower Measurements")
        st.caption("All measurements in centimeters")

        sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1, key="sl")
        sw = st.slider("Sepal Width (cm)",  1.5, 5.0, 3.0, 0.1, key="sw")
        pl = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1, key="pl")
        pw = st.slider("Petal Width (cm)",  0.1, 2.6, 1.2, 0.1, key="pw")

        predict_iris = st.button("🔍 Classify Species", key="btn_iris")
        st.markdown('</div>', unsafe_allow_html=True)

        # Feature visual
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Input Summary")
        feat_data = {"Feature": iris_features, "Value": [sl, sw, pl, pw]}
        df_feat = pd.DataFrame(feat_data)
        st.dataframe(df_feat, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction Result")

        if predict_iris:
            X_input = np.array([[sl, sw, pl, pw]])
            pred    = nb_model.predict(X_input)[0]
            probs   = nb_model.predict_proba(X_input)[0]
            species = iris_classes[pred]

            emoji_map = {"setosa": "🌺", "versicolor": "🌼", "virginica": "🌸"}
            color_map = {"setosa": "#00e5a0", "versicolor": "#00d4ff", "virginica": "#7b61ff"}
            emoji   = emoji_map.get(species, "🌸")
            color   = color_map.get(species, "#00d4ff")

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{color}22,{color}08);
                         border:1px solid {color};border-radius:12px;
                         padding:1.5rem;text-align:center;margin-bottom:1rem;">
              <div style="font-size:3rem;">{emoji}</div>
              <div class="result-label" style="color:{color};">Iris {species.title()}</div>
              <div class="result-sub">Confidence: {max(probs)*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("**Class Probabilities**")
            colors_bars = ["#00e5a0", "#00d4ff", "#7b61ff"]
            for i, cls in enumerate(iris_classes):
                prob_bar(f"Iris {cls}", probs[i], colors_bars[i])
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#5a7fa8;">
              <div style="font-size:3rem;">🌿</div>
              <div style="margin-top:0.5rem;">Adjust the sliders and click<br><strong>Classify Species</strong> to get results</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Species reference card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📖 Species Reference")
        ref_data = {
            "Species":   ["Setosa", "Versicolor", "Virginica"],
            "Petal L":   ["1.0–1.9", "3.0–5.1", "4.5–6.9"],
            "Petal W":   ["0.1–0.6", "1.0–1.8", "1.4–2.5"],
        }
        st.dataframe(pd.DataFrame(ref_data), hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — Diabetes Risk
# ═══════════════════════════════════════════════════════════════
elif page == "💉 Diabetes Risk":
    st.markdown("# 💉 Diabetes Risk Assessment")
    st.markdown("Enter patient clinical data to assess diabetes risk using the Gaussian Naïve Bayes model.")
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice.")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 👤 Patient Data — Part 1")
        pregnancies = st.number_input("Pregnancies", 0, 20, 1, help="Number of times pregnant")
        glucose     = st.number_input("Glucose (mg/dL)", 0, 300, 110, help="Plasma glucose concentration (2hr OGTT)")
        bp          = st.number_input("Blood Pressure (mmHg)", 0, 150, 72, help="Diastolic blood pressure")
        skin        = st.number_input("Skin Thickness (mm)", 0, 100, 20, help="Triceps skin fold thickness")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔬 Patient Data — Part 2")
        insulin  = st.number_input("Insulin (μU/mL)", 0, 900, 80, help="2-Hour serum insulin")
        bmi      = st.number_input("BMI (kg/m²)", 0.0, 70.0, 28.5, step=0.1, help="Body mass index")
        dpf      = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.35, step=0.01, help="Genetic risk score")
        age      = st.number_input("Age (years)", 10, 100, 35, help="Patient age")
        st.markdown('</div>', unsafe_allow_html=True)

    predict_diab = st.button("🩺 Assess Diabetes Risk", key="btn_diab")

    if predict_diab:
        st.markdown("---")
        X_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        pred    = diab_model.predict(X_input)[0]
        probs   = diab_model.predict_proba(X_input)[0]

        r1, r2, r3 = st.columns([1, 1.5, 1], gap="medium")

        with r1:
            st.metric("Risk Probability", f"{probs[1]*100:.1f}%", delta=None)
            st.metric("Model Confidence", f"{max(probs)*100:.1f}%")

        with r2:
            if pred == 1:
                st.markdown("""
                <div class="result-positive">
                  <div style="font-size:2.5rem;">⚠️</div>
                  <div class="result-label" style="color:#ff4d6d;">HIGH RISK</div>
                  <div class="result-sub">Diabetes Indicated</div>
                  <div style="margin-top:0.8rem;font-size:0.8rem;color:#5a7fa8;">
                    Please consult a healthcare professional for proper diagnosis.
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-negative">
                  <div style="font-size:2.5rem;">✅</div>
                  <div class="result-label" style="color:#00e5a0;">LOW RISK</div>
                  <div class="result-sub">No Diabetes Indicated</div>
                  <div style="margin-top:0.8rem;font-size:0.8rem;color:#5a7fa8;">
                    Maintain a healthy lifestyle and regular checkups.
                  </div>
                </div>""", unsafe_allow_html=True)

        with r3:
            st.markdown("**Risk Breakdown**")
            prob_bar("No Diabetes", probs[0], "#00e5a0")
            prob_bar("Diabetes",    probs[1], "#ff4d6d")

        # Clinical flags
        st.markdown("---")
        st.markdown("### 🚦 Clinical Flags")
        flags = []
        if glucose > 140:  flags.append(("🔴 High Glucose",       f"{glucose} mg/dL (>140)"))
        if bmi > 30:       flags.append(("🟡 High BMI",           f"{bmi:.1f} (>30 = Obese)"))
        if bp > 90:        flags.append(("🟡 Elevated BP",        f"{bp} mmHg (>90)"))
        if age > 45:       flags.append(("🔵 Age Factor",         f"{age} years (>45)"))
        if dpf > 0.5:      flags.append(("🟠 Genetic Risk",       f"DPF={dpf:.2f} (>0.5)"))
        if insulin == 0:   flags.append(("⚪ Missing Insulin",    "Value=0 may indicate missing data"))

        if flags:
            cols = st.columns(min(len(flags), 3))
            for i, (flag_title, flag_val) in enumerate(flags):
                with cols[i % 3]:
                    st.markdown(f'<div class="card" style="padding:0.8rem;"><strong>{flag_title}</strong><br><span style="font-size:0.8rem;color:#5a7fa8;">{flag_val}</span></div>', unsafe_allow_html=True)
        else:
            st.success("✅ All clinical markers within normal range.")


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — About
# ═══════════════════════════════════════════════════════════════
elif page == "ℹ️ About Models":
    st.markdown("# ℹ️ About the Models")
    st.markdown("---")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🌸 naive_bayes_model.pkl")
        st.markdown("""
**Algorithm:** Gaussian Naïve Bayes  
**Task:** Multi-class Classification  
**Dataset:** Iris Flower Dataset  
**Classes:** 3 (Setosa, Versicolor, Virginica)  
**Features:** 4 (sepal/petal length & width)

**How it works:**  
GaussianNB assumes features follow a normal (Gaussian) distribution. It applies Bayes' theorem with a "naïve" assumption of feature independence to compute posterior probabilities for each class.

**Strengths:**  
- Fast training & inference  
- Works well with small datasets  
- Probabilistic output  
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 💉 diabetes_model.pkl")
        st.markdown("""
**Algorithm:** Gaussian Naïve Bayes  
**Task:** Binary Classification  
**Dataset:** Pima Indians Diabetes Dataset  
**Classes:** 2 (No Diabetes, Diabetes)  
**Features:** 8 clinical indicators

**Features Used:**  
- Pregnancies, Glucose, Blood Pressure  
- Skin Thickness, Insulin, BMI  
- Diabetes Pedigree Function, Age

**Note:**  
Zero values in Glucose,BP,Skin,Insulin,or BMI likely indicate missing data in the original dataset.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📐 Gaussian Naïve Bayes Formula")
    st.latex(r"P(C_k \mid x) \propto P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)")
    st.latex(r"P(x_i \mid C_k) = \frac{1}{\sqrt{2\pi\sigma_{ik}^2}} \exp\!\left(-\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}\right)")
    st.markdown("""
Where:
- $P(C_k)$ = prior probability of class $k$  
- $\mu_{ik}$, $\sigma_{ik}^2$ = mean and variance of feature $i$ in class $k$  
- The predicted class is arg max P(C_k | x)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
