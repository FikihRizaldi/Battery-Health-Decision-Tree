# Klasifikasi Kesehatan Baterai dengan Decision Tree

Proyek ini memakai satu model utama, yaitu Decision Tree, untuk memprediksi kondisi baterai lithium-ion.

## Isi Folder

- `src/build_dataset.py`: membuat `data/battery_features.csv` dari file `.mat`.
- `model_components.py`: komponen pipeline yang dipakai model.
- `train_decision_tree.py`: menjalankan EDA, preprocessing, training, dan evaluasi.
- `app.py`: aplikasi web Streamlit untuk prediksi dan visualisasi model.
- `notebook_analisis.ipynb`: versi notebook untuk menampilkan proses per tahap.
- `web/`: tampilan web statis dengan HTML dan CSS.
- `draft_laporan.md`: bahan laporan.
- `models/decision_tree_battery.pkl`: model akhir.
- `outputs/01_eda/`: hasil EDA.
- `outputs/02_preprocessing/`: ringkasan preprocessing dan fitur.
- `outputs/03_modeling/`: hasil pemodelan.
- `outputs/04_evaluation/`: hasil evaluasi.

## Cara Menjalankan

```bash
python src/build_dataset.py
python train_decision_tree.py
streamlit run app.py
```

Tampilan web statis dapat dibuka dari:

```text
web/index.html
```

## Target

Target prediksi adalah `health_status`.

- `Healthy`: SOH >= 80%
- `Warning`: 70% <= SOH < 80%
- `End_of_Life`: SOH < 70%

Kolom `capacity_ahr` dan `soh` tidak dipakai sebagai fitur model karena dipakai untuk membentuk target.
