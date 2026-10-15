from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "B0005_discharge_data.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    print("=" * 60)
    print("BATTERY DEGRADATION VISUALIZATION")
    print("=" * 60)

    # Check whether the dataset exists.
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # Load the corrected dataset.
    df = pd.read_csv(DATASET_PATH)

    # Use the discharge order for plotting.
    df = df.sort_values("discharge_index") if (
        "discharge_index" in df.columns
    ) else df.sort_values("cycle_index")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Plot 1: Capacity degradation
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["discharge_index"],
        df["capacity_ah"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    plt.title("NASA Battery B0005 - Capacity Degradation")
    plt.xlabel("Discharge Cycle Index")
    plt.ylabel("Discharge Capacity (Ah)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    capacity_path = (
        OUTPUT_DIR / "B0005_capacity_degradation.png"
    )

    plt.savefig(capacity_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nCapacity plot saved: {capacity_path}")

    # --------------------------------------------------
    # Plot 2: State of Health degradation
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["discharge_index"],
        df["soh_percent"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    plt.title("NASA Battery B0005 - State of Health")
    plt.xlabel("Discharge Cycle Index")
    plt.ylabel("State of Health (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    soh_path = OUTPUT_DIR / "B0005_soh_degradation.png"

    plt.savefig(soh_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SoH plot saved: {soh_path}")

    print("\nBoth plots have been generated successfully!")


if __name__ == "__main__":
    main()