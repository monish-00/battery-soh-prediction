from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# 1. Locate the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "B0005_discharge_data.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# 2. Check and load the dataset
if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_FILE}"
    )

df = pd.read_csv(DATASET_FILE)

df = df.sort_values("cycle_index").reset_index(drop=True)


# 3. Display data quality information
print("\nDataset loaded successfully!")

print("Number of discharge records:", len(df))

print("\nMissing values:")
print(df.isnull().sum())

print("\nCapacity statistics:")
print(df["capacity_ah"].describe())

print("\nSoH statistics:")
print(df["soh_percent"].describe())


# 4. Plot capacity degradation
plt.figure(figsize=(10, 5))

plt.plot(
    df["cycle_index"],
    df["capacity_ah"],
    marker="o",
    markersize=3,
    linewidth=1.5,
)

plt.xlabel("Experimental Record Index")
plt.ylabel("Discharge Capacity (Ah)")
plt.title("NASA B0005 - Capacity Degradation")
plt.grid(True)
plt.tight_layout()

capacity_path = (
    OUTPUT_DIR / "B0005_capacity_degradation.png"
)

plt.savefig(capacity_path, dpi=150)

print(f"\nCapacity plot saved: {capacity_path}")

plt.show()


# 5. Plot State of Health
plt.figure(figsize=(10, 5))

plt.plot(
    df["cycle_index"],
    df["soh_percent"],
    marker="o",
    markersize=3,
    linewidth=1.5,
)

plt.xlabel("Experimental Record Index")
plt.ylabel("State of Health (%)")
plt.title("NASA B0005 - State of Health Degradation")
plt.grid(True)
plt.tight_layout()

soh_path = OUTPUT_DIR / "B0005_soh_degradation.png"

plt.savefig(soh_path, dpi=150)

print(f"SoH plot saved: {soh_path}")

plt.show()