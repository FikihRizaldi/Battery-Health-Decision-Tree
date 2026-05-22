from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "5.+Battery+Data+Set" / "5. Battery Data Set"
OUTPUT_PATH = PROJECT_ROOT / "data" / "battery_features.csv"
NOMINAL_CAPACITY_AHR = 2.0


def to_1d_array(value):
    return np.asarray(value, dtype=float).reshape(-1)


def safe_float(value):
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0]) if arr.size else np.nan


def stats(prefix, values):
    arr = to_1d_array(values)
    if arr.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.nanmean(arr)),
        f"{prefix}_min": float(np.nanmin(arr)),
        f"{prefix}_max": float(np.nanmax(arr)),
        f"{prefix}_std": float(np.nanstd(arr)),
    }


def health_label(soh):
    if soh >= 0.80:
        return "Healthy"
    if soh >= 0.70:
        return "Warning"
    return "End_of_Life"


def extract_battery(mat_path):
    mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    battery_id = mat_path.stem
    battery = mat_data[battery_id]
    rows = []
    discharge_index = 0

    for cycle_index, cycle in enumerate(np.asarray(battery.cycle).reshape(-1), start=1):
        if str(cycle.type).lower() != "discharge":
            continue

        discharge_index += 1
        data = cycle.data
        capacity = safe_float(getattr(data, "Capacity", np.nan))
        soh = capacity / NOMINAL_CAPACITY_AHR
        time_values = to_1d_array(getattr(data, "Time", []))

        row = {
            "battery_id": battery_id,
            "cycle_index": cycle_index,
            "discharge_index": discharge_index,
            "ambient_temperature": safe_float(getattr(cycle, "ambient_temperature", np.nan)),
            "elapsed_time_sec": float(np.nanmax(time_values)) if time_values.size else np.nan,
            "capacity_ahr": capacity,
            "soh": soh,
            "health_status": health_label(soh),
        }

        row.update(stats("voltage", getattr(data, "Voltage_measured", [])))
        row.update(stats("current", getattr(data, "Current_measured", [])))
        row.update(stats("temperature", getattr(data, "Temperature_measured", [])))
        rows.append(row)

    return rows


def main():
    mat_files = sorted(DATASET_ROOT.rglob("B*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"Tidak ada file .mat di {DATASET_ROOT}")

    rows = []
    seen = set()
    for mat_path in mat_files:
        # Beberapa baterai muncul di lebih dari satu folder dataset.
        # Untuk laporan sederhana, gunakan kemunculan pertama agar tidak duplikatif.
        if mat_path.stem in seen:
            continue
        seen.add(mat_path.stem)
        rows.extend(extract_battery(mat_path))

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset fitur tersimpan: {OUTPUT_PATH}")
    print(f"Jumlah baris: {len(df)}")
    print(f"Jumlah baterai: {df['battery_id'].nunique()}")
    print("Distribusi kelas:")
    print(df["health_status"].value_counts())


if __name__ == "__main__":
    main()
