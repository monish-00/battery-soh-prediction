from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Locate the processed dataset
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "B0005_discharge_data.csv"
)

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATASET_FILE}"
    )


# --------------------------------------------------
# 2. Load the data
# --------------------------------------------------

df = pd.read_csv(DATASET_FILE)

df = df.sort_values("cycle_index").reset_index(drop=True)

print("\nDataset loaded successfully.")

print(f"Number of records: {len(df)}")

print("\nMissing values:")

print(df.isnull().sum())

print("\nCapacity range:")

print(
    f"Minimum: {df['capacity_ah'].min():.4f} Ah"
)

print(
    f"Maximum: {df['capacity_ah'].max():.4f} Ah"
)

print("\nSoH range:")

print(
    f"Minimum: {df['soh_percent'].min():.2f}%"
)

print(
    f"Maximum: {df['soh_percent'].max():.2f}%"
)


# --------------------------------------------------
# 3. Plot capacity degradation
# --------------------------------------------------

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

capacity_plot = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "B0005_capacity_degradation.png"
)

plt.savefig(capacity_plot, dpi=150)

plt.show()

print(f"\nCapacity plot saved to: {capacity_plot}")


# --------------------------------------------------
# 4. Plot SoH degradation
# --------------------------------------------------

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

soh_plot = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "B0005_soh_degradation.png"
)

plt.savefig(soh_plot, dpi=150)

plt.show()

print(f"SoH plot saved to: {soh_plot}")