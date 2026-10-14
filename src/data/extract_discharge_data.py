from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Nasa-Battery-Dataset"
    / "1. BatteryAgingARC-FY08Q4"
    / "B0005.mat"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "B0005_discharge_data.csv"


# --------------------------------------------------
# 2. Load the MATLAB dataset
# --------------------------------------------------

mat_data = loadmat(DATASET_FILE)

battery = mat_data["B0005"][0, 0]

cycles = battery["cycle"][0]


# --------------------------------------------------
# 3. Extract discharge records
# --------------------------------------------------

records = []

for index, cycle in enumerate(cycles):

    cycle_type = str(cycle["type"][0]).strip().lower()

    if cycle_type != "discharge":
        continue

    measurement_data = cycle["data"][0, 0]

    # Extract measurements as one-dimensional arrays.
    voltage = np.asarray(
        measurement_data["Voltage_measured"][0, 0],
        dtype=float,
    ).ravel()

    current = np.asarray(
        measurement_data["Current_measured"][0, 0],
        dtype=float,
    ).ravel()

    temperature = np.asarray(
        measurement_data["Temperature_measured"][0, 0],
        dtype=float,
    ).ravel()

    time = np.asarray(
        measurement_data["Time"][0, 0],
        dtype=float,
    ).ravel()

    # Capacity is stored as a single value for the discharge.
    capacity_ah = float(
        np.asarray(
            measurement_data["Capacity"][0, 0],
            dtype=float,
        ).ravel()[0]
    )

    # Compute basic descriptive features.
    duration_s = float(time[-1] - time[0])

    record = {
        "battery_id": "B0005",
        "cycle_index": index,
        "capacity_ah": capacity_ah,
        "mean_voltage": float(np.mean(voltage)),
        "mean_current": float(np.mean(current)),
        "mean_temperature": float(np.mean(temperature)),
        "discharge_duration_s": duration_s,
        "measurement_count": len(voltage),
    }

    records.append(record)


# --------------------------------------------------
# 4. Create a DataFrame
# --------------------------------------------------

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No discharge records were extracted.")

# Keep only rows with valid, positive capacity.
df = df[
    df["capacity_ah"].notna()
    & (df["capacity_ah"] > 0)
].copy()


# --------------------------------------------------
# 5. Calculate State of Health
# --------------------------------------------------

# Verify this rated capacity against the dataset documentation
# before treating these values as final SoH labels.
RATED_CAPACITY_AH = 2.0

df["soh_percent"] = (
    df["capacity_ah"] / RATED_CAPACITY_AH
) * 100


# --------------------------------------------------
# 6. Save the processed dataset
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# 7. Display a summary
# --------------------------------------------------

print("\nDischarge data extraction completed!")

print(f"Number of discharge records: {len(df)}")

print(f"\nSaved CSV to:\n{OUTPUT_FILE}")

print("\nFirst five records:")

print(df.head().to_string(index=False))

print("\nDataset information:")

print(df.info())

print("\nCapacity summary:")

print(df["capacity_ah"].describe())

print("\nSoH summary:")

print(df["soh_percent"].describe())