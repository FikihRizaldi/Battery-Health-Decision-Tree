from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from model_components import BatteryFeatureEngineer, IQRClipper

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "battery_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "decision_tree_battery.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "01_eda"
PREPROCESSING_DIR = OUTPUT_DIR / "02_preprocessing"
MODELING_DIR = OUTPUT_DIR / "03_modeling"
EVALUATION_DIR = OUTPUT_DIR / "04_evaluation"

METRICS_PATH = EVALUATION_DIR / "metrics.txt"
FOLD_RESULTS_PATH = MODELING_DIR / "fold_evaluation.csv"
FEATURE_IMPORTANCE_PATH = EVALUATION_DIR / "feature_importance.png"
FEATURE_IMPORTANCE_CSV_PATH = EVALUATION_DIR / "feature_importance.csv"
CONFUSION_MATRIX_PATH = EVALUATION_DIR / "confusion_matrix.png"
TREE_PATH = MODELING_DIR / "decision_tree.png"
RULES_PATH = MODELING_DIR / "decision_tree_rules.txt"
EDA_CLASS_PATH = EDA_DIR / "eda_class_distribution.png"
EDA_SOH_PATH = EDA_DIR / "eda_soh_trend.png"
EDA_CORR_PATH = EDA_DIR / "eda_correlation_heatmap.png"
EDA_NOTE_PATH = EDA_DIR / "catatan_eda.txt"
DATASET_SUMMARY_PATH = PREPROCESSING_DIR / "dataset_summary.txt"
MISSING_VALUES_PATH = PREPROCESSING_DIR / "missing_values.csv"
SELECTED_FEATURES_PATH = PREPROCESSING_DIR / "selected_features.csv"
EXCLUDED_FEATURES_PATH = PREPROCESSING_DIR / "excluded_features.csv"

TARGET = "health_status"
GROUP_COLUMN = "battery_id"
EXCLUDED_COLUMNS = [
    "capacity_ahr",
    "soh",
    "elapsed_time_sec",
    "cycle_index",
    "discharge_index",
]

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
    "voltage_range": "Rentang tegangan",
    "current_range": "Rentang arus",
    "temperature_range": "Rentang suhu",
    "voltage_stability_ratio": "Rasio stabilitas tegangan",
    "temperature_stability_ratio": "Rasio stabilitas suhu",
}


# Tahap 1: membaca dataset dan memilih fitur
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET, GROUP_COLUMN])

    drop_columns = [GROUP_COLUMN, TARGET] + EXCLUDED_COLUMNS
    feature_columns = [col for col in df.columns if col not in drop_columns]

    X = df[feature_columns]
    y = df[TARGET]
    groups = df[GROUP_COLUMN]
    return df, X, y, groups, feature_columns


# Tahap 2: eksplorasi data
def make_eda_plots(df):
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x=TARGET, order=df[TARGET].value_counts().index, ax=ax)
    ax.set_title("Distribusi Kelas Kondisi Baterai")
    ax.set_xlabel("Status Kesehatan")
    ax.set_ylabel("Jumlah Data")
    fig.tight_layout()
    fig.savefig(EDA_CLASS_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sample_batteries = sorted(df[GROUP_COLUMN].unique())[:8]
    sns.lineplot(
        data=df[df[GROUP_COLUMN].isin(sample_batteries)],
        x="discharge_index",
        y="soh",
        hue=GROUP_COLUMN,
        ax=ax,
    )
    ax.axhline(0.8, color="orange", linestyle="--", label="Batas Warning")
    ax.axhline(0.7, color="red", linestyle="--", label="Batas End of Life")
    ax.set_title("Tren Penurunan State of Health")
    ax.set_xlabel("Siklus Discharge")
    ax.set_ylabel("State of Health")
    fig.tight_layout()
    fig.savefig(EDA_SOH_PATH, dpi=180)
    plt.close(fig)

    corr_columns = [
        "cycle_index",
        "discharge_index",
        "elapsed_time_sec",
        "voltage_mean",
        "voltage_min",
        "current_mean",
        "temperature_mean",
        "temperature_max",
        "capacity_ahr",
        "soh",
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[corr_columns].corr(numeric_only=True), annot=True, fmt=".2f", cmap="vlag", ax=ax)
    ax.set_title("Korelasi Fitur Numerik")
    fig.tight_layout()
    fig.savefig(EDA_CORR_PATH, dpi=180)
    plt.close(fig)

    with EDA_NOTE_PATH.open("w", encoding="utf-8") as file:
        file.write("Catatan EDA\n\n")
        file.write("Grafik SOH dan korelasi kapasitas dipakai hanya untuk memahami pola data.\n")
        file.write("Kolom capacity_ahr dan soh tidak digunakan sebagai fitur saat melatih model.\n")


# Tahap 3: ringkasan preprocessing dan dokumentasi fitur
def save_preprocessing_outputs(df, feature_columns):
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)

    missing_values = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "kolom", 0: "jumlah_missing"})
    )
    missing_values["persentase_missing"] = missing_values["jumlah_missing"] / len(df) * 100
    missing_values.to_csv(MISSING_VALUES_PATH, index=False)

    pd.DataFrame({"fitur_digunakan": feature_columns}).to_csv(
        SELECTED_FEATURES_PATH,
        index=False,
    )

    excluded_reasons = [
        ("battery_id", "ID grup baterai, dipakai untuk validasi, bukan fitur model"),
        ("health_status", "Target prediksi"),
        ("capacity_ahr", "Sumber pembentukan label, berisiko data leakage"),
        ("soh", "Sumber pembentukan label, berisiko data leakage"),
        ("elapsed_time_sec", "Dihapus agar model tidak memakai shortcut durasi discharge"),
        ("cycle_index", "Dihapus agar model tidak terlalu bergantung pada urutan siklus"),
        ("discharge_index", "Dihapus agar model tidak terlalu bergantung pada urutan discharge"),
    ]
    pd.DataFrame(excluded_reasons, columns=["kolom_dikeluarkan", "alasan"]).to_csv(
        EXCLUDED_FEATURES_PATH,
        index=False,
    )

    with DATASET_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        file.write("Ringkasan data dan preprocessing\n\n")
        file.write(f"Jumlah data: {len(df)}\n")
        file.write(f"Jumlah baterai unik: {df[GROUP_COLUMN].nunique()}\n")
        file.write(f"Jumlah fitur terpilih: {len(feature_columns)}\n")
        file.write(f"Kelas target: {', '.join(sorted(df[TARGET].unique()))}\n\n")
        file.write("Distribusi kelas:\n")
        file.write(df[TARGET].value_counts().to_string())
        file.write("\n\nKolom yang tidak digunakan sebagai fitur:\n")
        for column, reason in excluded_reasons:
            file.write(f"- {column}: {reason}\n")


# Tahap 4: pipeline model Decision Tree
def build_pipeline():
    return Pipeline(
        steps=[
            ("feature_engineering", BatteryFeatureEngineer()),
            ("outlier_capping", IQRClipper(factor=1.5)),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            ),
        ]
    )


# Tahap 5: validasi dengan nested StratifiedGroupKFold
def run_nested_group_cv(X, y, groups, labels, param_grid):
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

    out_of_fold_pred = pd.Series(index=y.index, dtype=object)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        grid = GridSearchCV(
            estimator=build_pipeline(),
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train, groups=groups_train)

        y_train_pred = grid.best_estimator_.predict(X_train)
        y_test_pred = grid.best_estimator_.predict(X_test)
        out_of_fold_pred.iloc[test_idx] = y_test_pred

        train_f1 = f1_score(y_train, y_train_pred, average="weighted", zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

        fold_rows.append(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "test_batteries": groups.iloc[test_idx].nunique(),
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "train_f1_weighted": train_f1,
                "test_f1_weighted": test_f1,
                "test_precision_weighted": precision_score(
                    y_test, y_test_pred, average="weighted", zero_division=0
                ),
                "test_recall_weighted": recall_score(
                    y_test, y_test_pred, average="weighted", zero_division=0
                ),
                "overfit_gap_f1": train_f1 - test_f1,
                "best_params": grid.best_params_,
            }
        )

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(FOLD_RESULTS_PATH, index=False)
    return out_of_fold_pred, fold_results


# Tahap 6: melatih model akhir
def train_final_model(X, y, groups, param_grid):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y, groups=groups)
    return grid


# Tahap 7: evaluasi, visualisasi, dan penyimpanan hasil
def save_model_outputs(final_model, y, oof_pred, labels, final_grid, fold_results, df, feature_columns):
    report = classification_report(y, oof_pred, labels=labels, zero_division=0)

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "class_names": labels,
            "best_params": final_grid.best_params_,
            "validation_strategy": "Nested StratifiedGroupKFold out-of-fold evaluation",
        },
        MODEL_PATH,
    )

    mean_metrics = fold_results[
        [
            "test_accuracy",
            "test_f1_weighted",
            "test_precision_weighted",
            "test_recall_weighted",
            "overfit_gap_f1",
        ]
    ].mean()
    std_metrics = fold_results[
        [
            "test_accuracy",
            "test_f1_weighted",
            "test_precision_weighted",
            "test_recall_weighted",
            "overfit_gap_f1",
        ]
    ].std()

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        file.write("Evaluasi model Decision Tree\n\n")
        file.write("Data\n")
        file.write(f"- Jumlah data: {len(df)}\n")
        file.write(f"- Jumlah baterai unik: {df[GROUP_COLUMN].nunique()}\n")
        file.write(f"- Jumlah fitur awal: {len(feature_columns)}\n")
        file.write(f"- Kelas target: {', '.join(labels)}\n\n")
        file.write("Strategi validasi\n")
        file.write("- Model hanya menggunakan algoritma Decision Tree.\n")
        file.write("- Evaluasi memakai nested StratifiedGroupKFold berdasarkan battery_id.\n")
        file.write("- Setiap baterai pernah menjadi data uji pada salah satu fold.\n")
        file.write("- Model akhir dilatih ulang pada seluruh data setelah evaluasi selesai.\n")
        file.write("- Fitur capacity_ahr, soh, elapsed_time_sec, cycle_index, dan discharge_index tidak digunakan.\n")
        file.write("- Tujuannya mengurangi shortcut prediksi dan membuat model lebih fokus pada pola sensor.\n\n")
        file.write("Parameter terbaik model akhir\n")
        file.write(f"{final_grid.best_params_}\n\n")
        file.write("Rata-rata metrik per fold\n")
        file.write(
            f"- Akurasi: {mean_metrics['test_accuracy']:.4f} +/- {std_metrics['test_accuracy']:.4f}\n"
        )
        file.write(
            f"- F1-score berbobot: {mean_metrics['test_f1_weighted']:.4f} +/- {std_metrics['test_f1_weighted']:.4f}\n"
        )
        file.write(
            f"- Precision berbobot: {mean_metrics['test_precision_weighted']:.4f} +/- {std_metrics['test_precision_weighted']:.4f}\n"
        )
        file.write(
            f"- Recall berbobot: {mean_metrics['test_recall_weighted']:.4f} +/- {std_metrics['test_recall_weighted']:.4f}\n"
        )
        file.write(
            f"- Selisih F1 train-test: {mean_metrics['overfit_gap_f1']:.4f} +/- {std_metrics['overfit_gap_f1']:.4f}\n\n"
        )
        file.write("Laporan klasifikasi out-of-fold\n")
        file.write(report)
        file.write("\n\nDetail per fold tersimpan pada outputs/03_modeling/fold_evaluation.csv\n")

    cm = confusion_matrix(y, oof_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title("Matriks Konfusi Decision Tree")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=180)
    plt.close(fig)

    engineered_features = final_model.named_steps["feature_engineering"].get_feature_names_out()
    tree = final_model.named_steps["model"]
    importance_df = pd.DataFrame(
        {
            "feature": engineered_features,
            "label": [FEATURE_LABELS.get(feature, feature) for feature in engineered_features],
            "importance": tree.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_CSV_PATH, index=False)
    top_importance = importance_df.head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=top_importance, x="importance", y="label", ax=ax, color="#2563eb")
    ax.set_title("Kepentingan Fitur Decision Tree")
    ax.set_xlabel("Nilai kepentingan")
    ax.set_ylabel("Fitur")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=180)
    plt.close(fig)

    readable_rules = export_text(
        tree,
        feature_names=[FEATURE_LABELS.get(feature, feature) for feature in engineered_features],
        max_depth=4,
    )
    readable_rules = readable_rules.replace("class:", "kelas:")
    readable_rules = readable_rules.replace(
        "truncated branch of depth",
        "cabang dipotong pada kedalaman",
    )
    with RULES_PATH.open("w", encoding="utf-8") as file:
        file.write("Aturan model Decision Tree\n\n")
        file.write("Aturan dibatasi sampai kedalaman 4 agar mudah dibaca.\n\n")
        file.write(readable_rules)

    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        tree,
        feature_names=[FEATURE_LABELS.get(feature, feature) for feature in engineered_features],
        class_names=labels,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Visualisasi Decision Tree Sederhana")
    fig.tight_layout()
    fig.savefig(TREE_PATH, dpi=180)
    plt.close(fig)

    print(f"Model tersimpan: {MODEL_PATH}")
    print(f"Metrik tersimpan: {METRICS_PATH}")
    print(f"Matriks konfusi tersimpan: {CONFUSION_MATRIX_PATH}")
    print(f"Kepentingan fitur tersimpan: {FEATURE_IMPORTANCE_PATH}")
    print(f"Aturan Decision Tree tersimpan: {RULES_PATH}")
    print(f"Visualisasi Decision Tree tersimpan: {TREE_PATH}")
    print(report)


def main():
    print("\nTahap 0 - Menyiapkan folder output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    for output_dir in [EDA_DIR, PREPROCESSING_DIR, MODELING_DIR, EVALUATION_DIR]:
        output_dir.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    print("Tahap 1 - Membaca dataset dan seleksi fitur")
    df, X, y, groups, feature_columns = load_dataset()
    labels = sorted(y.unique())

    print("Tahap 2 - Membuat hasil EDA")
    make_eda_plots(df)

    print("Tahap 3 - Menyimpan output preprocessing dan seleksi fitur")
    save_preprocessing_outputs(df, feature_columns)

    print("Tahap 4 - Menyiapkan hyperparameter Decision Tree")
    param_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [3, 4, 5, 6, 7, 8],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 3, 5, 10],
    }

    print("Tahap 5 - Evaluasi nested StratifiedGroupKFold")
    oof_pred, fold_results = run_nested_group_cv(X, y, groups, labels, param_grid)

    print("Tahap 6 - Melatih model akhir Decision Tree")
    final_grid = train_final_model(X, y, groups, param_grid)

    print("Tahap 7 - Menyimpan model, metrik, dan visualisasi")
    save_model_outputs(
        final_model=final_grid.best_estimator_,
        y=y,
        oof_pred=oof_pred,
        labels=labels,
        final_grid=final_grid,
        fold_results=fold_results,
        df=df,
        feature_columns=feature_columns,
    )


if __name__ == "__main__":
    main()
