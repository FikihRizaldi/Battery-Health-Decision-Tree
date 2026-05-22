from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "decision_tree_battery.pkl"
DATA_PATH = BASE_DIR / "data" / "battery_features.csv"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "outputs" / "04_evaluation" / "feature_importance.csv"
FOLD_EVALUATION_PATH = BASE_DIR / "outputs" / "03_modeling" / "fold_evaluation.csv"

FEATURE_LABELS = {
    "ambient_temperature": "Suhu lingkungan",
    "voltage_mean": "Rata-rata tegangan",
    "voltage_min": "Tegangan minimum",
    "voltage_max": "Tegangan maksimum",
    "voltage_std": "Variasi tegangan",
    "current_mean": "Rata-rata arus",
    "current_min": "Arus minimum",
    "current_max": "Arus maksimum",
    "current_std": "Variasi arus",
    "temperature_mean": "Rata-rata suhu baterai",
    "temperature_min": "Suhu minimum baterai",
    "temperature_max": "Suhu maksimum baterai",
    "temperature_std": "Variasi suhu baterai",
}

CLASS_COLORS = {
    "Healthy": "#14b8a6",
    "Warning": "#f59e0b",
    "End_of_Life": "#ef4444",
}

CLASS_DESCRIPTIONS = {
    "Healthy": "Pola sensor masih menunjukkan kondisi baterai yang stabil.",
    "Warning": "Pola sensor berada pada area transisi dan perlu dipantau.",
    "End_of_Life": "Pola sensor mengarah pada kondisi baterai yang sudah menurun.",
}


st.set_page_config(
    page_title="Battery Health Decision Tree",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
        #MainMenu, header, footer {
            visibility: hidden;
        }

        :root {
            --bg: #0b0f14;
            --panel: #111827;
            --panel-soft: #16202f;
            --text: #f8fafc;
            --muted: #94a3b8;
            --line: rgba(148, 163, 184, 0.22);
            --violet: #8b5cf6;
            --indigo: #4f46e5;
            --teal: #14b8a6;
            --amber: #f59e0b;
            --red: #ef4444;
        }

        .stApp {
            background:
                linear-gradient(180deg, #0b0f14 0%, #101827 48%, #0b0f14 100%);
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .block-container {
            max-width: 1320px;
            padding: 28px 34px 44px;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 28px;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: var(--text);
        }

        section[data-testid="stSidebar"] label {
            font-size: 13px;
            font-weight: 700;
        }

        div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid var(--line);
            border-radius: 8px;
        }

        div[data-testid="stExpander"] details summary {
            font-weight: 800;
            color: #f8fafc;
        }

        div[data-testid="stNumberInput"] input {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            color: #f8fafc;
            font-weight: 650;
        }

        div[data-testid="stNumberInput"] input:focus {
            border-color: var(--violet);
            box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.35);
        }

        .stButton > button,
        .stFormSubmitButton > button {
            width: 100%;
            min-height: 46px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: linear-gradient(135deg, var(--indigo), var(--violet));
            color: white;
            font-weight: 850;
            box-shadow: 0 14px 34px rgba(79, 70, 229, 0.28);
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            border-color: rgba(255, 255, 255, 0.22);
            color: white;
            transform: translateY(-1px);
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 18px;
            margin-bottom: 18px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 8px;
            background: linear-gradient(135deg, var(--violet), var(--teal));
            box-shadow: 0 12px 30px rgba(20, 184, 166, 0.18);
        }

        .brand-title {
            color: #f8fafc;
            font-size: 16px;
            font-weight: 850;
            margin: 0;
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: 13px;
            margin: 2px 0 0;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 9px 14px;
            color: #dbeafe;
            background: rgba(37, 99, 235, 0.12);
            border: 1px solid rgba(147, 197, 253, 0.22);
            font-size: 13px;
            font-weight: 750;
            white-space: nowrap;
        }

        .hero {
            border-radius: 8px;
            padding: 30px;
            border: 1px solid var(--line);
            background:
                linear-gradient(135deg, rgba(17, 24, 39, 0.98), rgba(30, 41, 59, 0.96));
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.30);
            margin-bottom: 22px;
        }

        .hero-kicker {
            color: #c4b5fd;
            font-size: 13px;
            font-weight: 850;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .hero-title {
            color: #ffffff;
            font-size: 42px;
            line-height: 1.08;
            font-weight: 900;
            margin: 0;
        }

        .hero-copy {
            color: #cbd5e1;
            font-size: 15px;
            line-height: 1.7;
            max-width: 820px;
            margin: 14px 0 0;
        }

        .metric-card,
        .result-panel {
            border-radius: 8px;
            border: 1px solid var(--line);
            background: rgba(15, 23, 42, 0.92);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.24);
        }

        .metric-card {
            min-height: 116px;
            padding: 18px;
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--teal), var(--violet));
        }

        .metric-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
        }

        .metric-value {
            color: #ffffff;
            display: block;
            font-size: 28px;
            line-height: 1;
            font-weight: 900;
            margin-top: 14px;
        }

        .metric-note {
            color: #94a3b8;
            font-size: 12px;
            margin-top: 9px;
        }

        .result-panel {
            min-height: 356px;
            padding: 24px;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
            margin-bottom: 26px;
        }

        .section-title {
            color: #f8fafc;
            font-size: 18px;
            font-weight: 900;
            margin: 0;
        }

        .section-caption {
            color: var(--muted);
            font-size: 13px;
            margin: 4px 0 0;
        }

        .class-chip {
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 12px;
            font-weight: 850;
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }

        .prediction-name {
            color: #ffffff;
            font-size: 48px;
            line-height: 1;
            font-weight: 950;
            margin: 0 0 12px;
        }

        .prediction-desc {
            color: #cbd5e1;
            font-size: 15px;
            line-height: 1.75;
            margin: 0;
        }

        .confidence-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        .confidence-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 9px;
        }

        .confidence-value {
            color: #ffffff;
            font-size: 30px;
            font-weight: 900;
        }

        .chart-header {
            padding: 18px;
            margin-bottom: 10px;
            border-radius: 8px;
            border: 1px solid var(--line);
            background: rgba(15, 23, 42, 0.92);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.16);
        }

        .panel-heading {
            color: #f8fafc;
            font-size: 17px;
            font-weight: 900;
            margin: 0 0 2px;
        }

        .panel-note {
            color: var(--muted);
            font-size: 13px;
            margin: 0 0 14px;
        }

        .stPlotlyChart {
            border-radius: 8px;
            overflow: hidden;
            padding: 10px;
            border: 1px solid var(--line);
            background: rgba(15, 23, 42, 0.88);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.22);
        }

        .dataframe {
            border-radius: 8px;
        }

        @media (max-width: 820px) {
            .block-container {
                padding: 22px 18px 34px;
            }

            .topbar {
                align-items: flex-start;
                flex-direction: column;
            }

            .hero {
                padding: 24px;
            }

            .hero-title {
                font-size: 32px;
            }

            .prediction-name {
                font-size: 38px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    fold_eval = pd.read_csv(FOLD_EVALUATION_PATH)
    return data, importance, fold_eval


def display_class(name):
    return "End of Life" if name == "End_of_Life" else name


def feature_input(label, key, data, step=0.01):
    min_value = float(data[key].min())
    max_value = float(data[key].max())
    median_value = float(data[key].median())
    if min_value == max_value:
        max_value = min_value + 1.0

    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=median_value,
        step=step,
        format="%.4f",
    )


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <span class="metric-value">{value}</span>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(height=360):
    return dict(
        height=height,
        margin=dict(l=10, r=10, t=16, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.65)",
        font=dict(color="#e5e7eb", family="Inter, Arial, sans-serif"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.14)", zeroline=False),
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.14)", zeroline=False),
    )


def probability_chart(class_names, probabilities):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[display_class(name) for name in class_names],
            y=probabilities,
            marker=dict(
                color=[CLASS_COLORS.get(name, "#8b5cf6") for name in class_names],
                line=dict(color="rgba(255,255,255,0.18)", width=1),
            ),
            text=[f"{value:.2f}" for value in probabilities],
            textposition="outside",
            hovertemplate="%{x}<br>Probabilitas: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(**plotly_layout(height=318), showlegend=False)
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_xaxes(title=None)
    return fig


def feature_importance_chart(importance):
    top = importance.head(10).sort_values("importance", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["importance"],
            y=top["label"],
            orientation="h",
            marker=dict(
                color=top["importance"],
                colorscale=[[0, "#334155"], [0.5, "#14b8a6"], [1, "#8b5cf6"]],
                line=dict(color="rgba(255,255,255,0.16)", width=1),
            ),
            hovertemplate="%{y}<br>Nilai: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(**plotly_layout(height=376), showlegend=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    return fig


def fold_chart(fold_eval):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fold_eval["fold"],
            y=fold_eval["test_accuracy"],
            mode="lines+markers",
            name="Akurasi",
            line=dict(color="#14b8a6", width=3),
            marker=dict(size=9, color="#14b8a6"),
            hovertemplate="Fold %{x}<br>Akurasi: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fold_eval["fold"],
            y=fold_eval["test_f1_weighted"],
            mode="lines+markers",
            name="F1 berbobot",
            line=dict(color="#8b5cf6", width=3),
            marker=dict(size=9, color="#8b5cf6"),
            hovertemplate="Fold %{x}<br>F1: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(**plotly_layout(height=376))
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_xaxes(dtick=1, title="Fold")
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=1.16,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
        )
    )
    return fig


def render_result(prediction, confidence, submitted):
    if submitted:
        label = display_class(prediction)
        desc = CLASS_DESCRIPTIONS[prediction]
        color = CLASS_COLORS[prediction]
        confidence_text = f"{confidence:.1%}"
    else:
        label = "Belum Diprediksi"
        desc = "Nilai input sudah siap. Jalankan prediksi untuk melihat kelas kesehatan baterai."
        color = "#64748b"
        confidence_text = "-"

    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-header">
                <div>
                    <p class="section-title">Hasil Prediksi</p>
                    <p class="section-caption">Klasifikasi kondisi berdasarkan pola sensor discharge</p>
                </div>
                <span class="class-chip" style="background:{color};">{label}</span>
            </div>
            <h2 class="prediction-name">{label}</h2>
            <p class="prediction-desc">{desc}</p>
            <div class="confidence-box">
                <div class="confidence-label">Confidence tertinggi</div>
                <div class="confidence-value">{confidence_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_css()
bundle = load_model()
data, feature_importance, fold_evaluation = load_data()

model = bundle["model"]
feature_columns = bundle["feature_columns"]
class_names = bundle["class_names"]

with st.sidebar:
    st.markdown("### Control Panel")
    st.caption("Input sensor discharge baterai")

    with st.form("prediction_form"):
        user_input = {}

        with st.expander("Lingkungan", expanded=True):
            user_input["ambient_temperature"] = feature_input(
                FEATURE_LABELS["ambient_temperature"], "ambient_temperature", data
            )

        with st.expander("Tegangan", expanded=True):
            for key in ["voltage_mean", "voltage_min", "voltage_max", "voltage_std"]:
                user_input[key] = feature_input(FEATURE_LABELS[key], key, data)

        with st.expander("Arus", expanded=False):
            for key in ["current_mean", "current_min", "current_max", "current_std"]:
                user_input[key] = feature_input(FEATURE_LABELS[key], key, data)

        with st.expander("Suhu baterai", expanded=False):
            for key in [
                "temperature_mean",
                "temperature_min",
                "temperature_max",
                "temperature_std",
            ]:
                user_input[key] = feature_input(FEATURE_LABELS[key], key, data)

        submitted = st.form_submit_button("Jalankan Prediksi")

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark"></div>
            <div>
                <p class="brand-title">Battery Health Decision Tree</p>
                <p class="brand-subtitle">UTS Proyek Data Mining</p>
            </div>
        </div>
        <div class="status-pill">Model aktif: Decision Tree</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Battery Health Monitoring</div>
        <h1 class="hero-title">Dashboard Prediksi Kesehatan Baterai Lithium-Ion</h1>
        <p class="hero-copy">
            Sistem ini menggunakan model Decision Tree untuk mengklasifikasikan kondisi baterai
            berdasarkan pola tegangan, arus, suhu, dan suhu lingkungan pada proses discharge.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
with metric_1:
    metric_card("Model", "Decision Tree", "Model final")
with metric_2:
    metric_card("Akurasi", f"{fold_evaluation['test_accuracy'].mean():.4f}", "Rata-rata validasi")
with metric_3:
    metric_card("F1 berbobot", f"{fold_evaluation['test_f1_weighted'].mean():.4f}", "Rata-rata validasi")
with metric_4:
    metric_card("Fitur", str(len(feature_columns)), "Sensor terpilih")

st.write("")

input_df = pd.DataFrame([user_input], columns=feature_columns)
prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]
confidence = float(max(probabilities))
display_probabilities = probabilities if submitted else [0, 0, 0]

left_col, right_col = st.columns([0.95, 1.05])

with left_col:
    render_result(prediction, confidence, submitted)

with right_col:
    st.markdown(
        """
        <div class="chart-header">
            <p class="panel-heading">Probabilitas Kelas</p>
            <p class="panel-note">Perbandingan skor prediksi untuk setiap kelas target</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        probability_chart(class_names, display_probabilities),
        width="stretch",
        config={"displayModeBar": False},
    )

st.write("")

chart_col_1, chart_col_2 = st.columns(2)

with chart_col_1:
    st.markdown(
        """
        <div class="chart-header">
            <p class="panel-heading">Kepentingan Fitur</p>
            <p class="panel-note">Fitur paling berpengaruh pada model Decision Tree</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        feature_importance_chart(feature_importance),
        width="stretch",
        config={"displayModeBar": False},
    )

with chart_col_2:
    st.markdown(
        """
        <div class="chart-header">
            <p class="panel-heading">Performa per Fold</p>
            <p class="panel-note">Stabilitas akurasi dan F1-score selama validasi</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        fold_chart(fold_evaluation),
        width="stretch",
        config={"displayModeBar": False},
    )

with st.expander("Data input model"):
    st.dataframe(input_df.rename(columns=FEATURE_LABELS), width="stretch")
