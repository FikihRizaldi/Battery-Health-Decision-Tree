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

CLASS_COLORS = {
    "Healthy": "#22c55e",
    "Warning": "#f59e0b",
    "End_of_Life": "#ef4444",
}

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
    "temperature_mean": "Rata-rata suhu",
    "temperature_min": "Suhu minimum",
    "temperature_max": "Suhu maksimum",
    "temperature_std": "Variasi suhu",
}


st.set_page_config(
    page_title="Battery Health Decision Tree",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
        #MainMenu, footer, header {visibility: hidden;}

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.28), transparent 34%),
                linear-gradient(135deg, #12091f 0%, #17112c 48%, #20153c 100%);
            color: #f8fafc;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2.6rem;
            max-width: 1280px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #160a2b 0%, #21113f 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] label {
            color: #d8d4ef !important;
            font-size: 0.86rem;
            font-weight: 650;
        }

        div[data-testid="stNumberInput"] input {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: #ffffff;
            border-radius: 10px;
        }

        .hero-card {
            padding: 28px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(91, 33, 182, 0.92), rgba(49, 46, 129, 0.92));
            border: 1px solid rgba(255, 255, 255, 0.14);
            box-shadow: 0 22px 70px rgba(12, 8, 28, 0.36);
            margin-bottom: 22px;
        }

        .hero-card .eyebrow {
            color: #ddd6fe;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .hero-card h1 {
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3.45rem);
            line-height: 1.05;
            margin: 0;
        }

        .hero-card p {
            color: #e9d5ff;
            max-width: 780px;
            margin-top: 12px;
            line-height: 1.65;
            font-size: 1rem;
        }

        .metric-card, .result-card, .chart-card {
            padding: 20px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.94);
            color: #171126;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 18px 44px rgba(7, 4, 20, 0.24);
        }

        .metric-card span, .result-card span {
            color: #6b5f86;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .metric-card strong {
            display: block;
            margin-top: 8px;
            color: #1f1140;
            font-size: 1.65rem;
        }

        .result-card {
            min-height: 245px;
        }

        .prediction-label {
            display: inline-flex;
            padding: 8px 12px;
            border-radius: 999px;
            background: #ede9fe;
            color: #4c1d95;
            font-size: 0.82rem;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .prediction-main {
            color: #1f1140;
            font-size: clamp(2.1rem, 5vw, 4rem);
            font-weight: 850;
            margin: 0 0 8px;
        }

        .prediction-copy {
            color: #5b516e;
            line-height: 1.55;
            margin: 0;
        }

        .stButton > button {
            width: 100%;
            height: 46px;
            border-radius: 13px;
            border: 0;
            color: #ffffff;
            background: linear-gradient(135deg, #7c3aed, #4f46e5);
            box-shadow: 0 14px 28px rgba(79, 70, 229, 0.28);
            font-weight: 800;
        }

        .stButton > button:hover {
            border: 0;
            background: linear-gradient(135deg, #8b5cf6, #6366f1);
            color: #ffffff;
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.055);
        }

        .chart-title {
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 0 0 12px;
        }

        .small-note {
            color: #c4b5fd;
            font-size: 0.86rem;
            line-height: 1.55;
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


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def class_display(name):
    return "End of Life" if name == "End_of_Life" else name


def prediction_text(prediction):
    if prediction == "Healthy":
        return "Baterai berada pada kondisi baik berdasarkan pola sensor yang dimasukkan."
    if prediction == "Warning":
        return "Baterai berada pada area transisi dan perlu dipantau lebih lanjut."
    return "Baterai terindikasi mendekati akhir masa pakai berdasarkan pola sensor."


def probability_chart(class_names, probabilities):
    colors = [CLASS_COLORS.get(name, "#7c3aed") for name in class_names]
    fig = go.Figure(
        data=[
            go.Bar(
                x=[class_display(name) for name in class_names],
                y=probabilities,
                marker_color=colors,
                text=[f"{value:.2f}" for value in probabilities],
                textposition="outside",
                hovertemplate="%{x}<br>Probabilitas: %{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
        font=dict(color="#1f1140"),
        yaxis=dict(range=[0, 1], gridcolor="rgba(76, 29, 149, 0.12)"),
        xaxis=dict(title=None),
        showlegend=False,
    )
    return fig


def feature_importance_chart(importance):
    top = importance.head(10).sort_values("importance", ascending=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=top["importance"],
                y=top["label"],
                orientation="h",
                marker_color="#7c3aed",
                hovertemplate="%{y}<br>Nilai: %{x:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.94)",
        font=dict(color="#1f1140"),
        xaxis=dict(gridcolor="rgba(76, 29, 149, 0.12)"),
        yaxis=dict(title=None),
    )
    return fig


def fold_chart(fold_eval):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fold_eval["fold"],
            y=fold_eval["test_accuracy"],
            mode="lines+markers",
            name="Akurasi",
            line=dict(color="#7c3aed", width=3),
            marker=dict(size=9),
            hovertemplate="Fold %{x}<br>Akurasi: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fold_eval["fold"],
            y=fold_eval["test_f1_weighted"],
            mode="lines+markers",
            name="F1 berbobot",
            line=dict(color="#22c55e", width=3),
            marker=dict(size=9),
            hovertemplate="Fold %{x}<br>F1: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.94)",
        font=dict(color="#1f1140"),
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(76, 29, 149, 0.12)"),
        xaxis=dict(dtick=1, title="Fold"),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


inject_css()
bundle = load_model()
data, feature_importance, fold_evaluation = load_data()

model = bundle["model"]
feature_columns = bundle["feature_columns"]
class_names = bundle["class_names"]

with st.sidebar:
    st.markdown("## Input Prediksi")
    st.markdown(
        '<p class="small-note">Masukkan nilai sensor discharge. Nilai awal memakai median dataset.</p>',
        unsafe_allow_html=True,
    )

    user_input = {}
    with st.expander("Lingkungan", expanded=True):
        user_input["ambient_temperature"] = feature_input(
            "Suhu lingkungan", "ambient_temperature", data
        )

    with st.expander("Tegangan", expanded=True):
        for key in ["voltage_mean", "voltage_min", "voltage_max", "voltage_std"]:
            user_input[key] = feature_input(FEATURE_LABELS[key], key, data)

    with st.expander("Arus", expanded=True):
        for key in ["current_mean", "current_min", "current_max", "current_std"]:
            user_input[key] = feature_input(FEATURE_LABELS[key], key, data)

    with st.expander("Suhu Baterai", expanded=True):
        for key in ["temperature_mean", "temperature_min", "temperature_max", "temperature_std"]:
            user_input[key] = feature_input(FEATURE_LABELS[key], key, data)

    predict_button = st.button("Prediksi Kondisi Baterai", type="primary")

st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Deployment Model Data Mining</div>
        <h1>Klasifikasi Kesehatan Baterai Lithium-Ion</h1>
        <p>Aplikasi ini menggunakan model Decision Tree untuk memprediksi kondisi baterai berdasarkan ringkasan sensor discharge.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
with metric_1:
    metric_card("Model", "Decision Tree")
with metric_2:
    metric_card("Akurasi", f"{fold_evaluation['test_accuracy'].mean():.4f}")
with metric_3:
    metric_card("F1 berbobot", f"{fold_evaluation['test_f1_weighted'].mean():.4f}")
with metric_4:
    metric_card("Jumlah fitur", str(len(feature_columns)))

st.write("")

input_df = pd.DataFrame([user_input], columns=feature_columns)
prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]

left, right = st.columns([1.05, 1])

with left:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="prediction-label">Hasil Prediksi</div>
            <div class="prediction-main">{class_display(prediction) if predict_button else "Siap Diprediksi"}</div>
            <p class="prediction-copy">{prediction_text(prediction) if predict_button else "Atur parameter pada sidebar, lalu tekan tombol prediksi untuk melihat hasil klasifikasi."}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<p class="chart-title">Probabilitas Kelas</p>', unsafe_allow_html=True)
    if predict_button:
        st.plotly_chart(probability_chart(class_names, probabilities), use_container_width=True)
    else:
        st.plotly_chart(
            probability_chart(class_names, [0, 0, 0]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

st.write("")

chart_col_1, chart_col_2 = st.columns(2)
with chart_col_1:
    st.markdown('<p class="chart-title">Kepentingan Fitur</p>', unsafe_allow_html=True)
    st.plotly_chart(feature_importance_chart(feature_importance), use_container_width=True)

with chart_col_2:
    st.markdown('<p class="chart-title">Performa per Fold</p>', unsafe_allow_html=True)
    st.plotly_chart(fold_chart(fold_evaluation), use_container_width=True)

with st.expander("Data input yang dikirim ke model"):
    st.dataframe(input_df.rename(columns=FEATURE_LABELS), use_container_width=True)
