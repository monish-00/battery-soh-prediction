from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Nasa-Battery-Dataset"
    / "1. BatteryAgingARC-FY08Q4"
    / "B0005.mat"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "B0005_discharge_data.csv"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Temporary assumption. Verify against the NASA
# dataset documentation before finalizing SoH labels.
RATED_CAPACITY_AH = 2.0


def main():
    if not MAT_FILE.exists():
        raise FileNotFoundError(
            f"NASA MATLAB file not found: {MAT_FILE}"
        )

    print("Loading NASA battery dataset...")

    dataset = loadmat(
        MAT_FILE,
        squeeze_me=True,
        struct_as_record=False,
    )

    battery = dataset["B0005"]
    cycles = np.atleast_1d(battery.cycle)

    records = []

    for position, cycle in enumerate(cycles):
        cycle_type = str(cycle.type).strip().lower()

        # Only process discharge records.
        if cycle_type != "discharge":
            continue

        data = cycle.data

        # Extract complete measurement arrays.
        voltage = np.asarray(
            data.Voltage_measured,
            dtype=float,
        ).reshape(-1)

        current = np.asarray(
            data.Current_measured,
            dtype=float,
        ).reshape(-1)

        temperature = np.asarray(
            data.Temperature_measured,
            dtype=float,
        ).reshape(-1)

        time = np.asarray(
            data.Time,
            dtype=float,
        ).reshape(-1)

        current_load = np.asarray(
            data.Current_load,
            dtype=float,
        ).reshape(-1)

        voltage_load = np.asarray(
            data.Voltage_load,
            dtype=float,
        ).reshape(-1)

        capacity_ah = float(data.Capacity)

        # All time-series arrays should have compatible
        # lengths for reliable feature extraction.
        lengths = {
            len(voltage),
            len(current),
            len(temperature),
            len(time),
            len(current_load),
            len(voltage_load),
        }

        if len(lengths) != 1:
            print(
                f"Skipping cycle at position {position}: "
                "measurement arrays have different lengths."
            )
            continue

        if len(time) < 2:
            print(
                f"Skipping cycle at position {position}: "
                "insufficient timestamps."
            )
            continue

        if not np.all(np.isfinite(time)):
            print(
                f"Skipping cycle at position {position}: "
                "invalid timestamps."
            )
            continue

        # Sort by time if the timestamps are not ordered.
        order = np.argsort(time)

        time = time[order]
        voltage = voltage[order]
        current = current[order]
        temperature = temperature[order]
        current_load = current_load[order]
        voltage_load = voltage_load[order]

        # Remove duplicate timestamps for duration and
        # time-based calculations.
        unique_mask = np.concatenate(
            ([True], np.diff(time) > 0)
        )

        time = time[unique_mask]
        voltage = voltage[unique_mask]
        current = current[unique_mask]
        temperature = temperature[unique_mask]
        current_load = current_load[unique_mask]
        voltage_load = voltage_load[unique_mask]

        if len(time) < 2:
            continue

        # Check measurements for invalid values.
        arrays = [
            voltage,
            current,
            temperature,
            current_load,
            voltage_load,
        ]

        if not all(
            np.all(np.isfinite(array))
            for array in arrays
        ):
            print(
                f"Skipping cycle at position {position}: "
                "non-finite sensor measurements."
            )
            continue

        if not np.isfinite(capacity_ah) or capacity_ah <= 0:
            print(
                f"Skipping cycle at position {position}: "
                "invalid capacity."
            )
            continue

        # Duration is the difference between the final
        # and initial timestamps.
        duration_seconds = float(time[-1] - time[0])

        # A discharge record should have meaningful
        # elapsed time.
        if duration_seconds <= 0:
            print(
                f"Skipping cycle at position {position}: "
                "non-positive discharge duration."
            )
            continue

        # NASA SoH label: provisional until the reference
        # capacity is verified.
        soh_percent = (
            capacity_ah / RATED_CAPACITY_AH
        ) * 100.0

        discharge_index = len(records) + 1

        records.append(
            {
                "battery_id": "B0005",
                "source_cycle_position": position,
                "discharge_index": discharge_index,
                "capacity_ah": capacity_ah,
                "mean_voltage": float(np.mean(voltage)),
                "min_voltage": float(np.min(voltage)),
                "max_voltage": float(np.max(voltage)),
                "mean_current": float(np.mean(current)),
                "mean_abs_current": float(
                    np.mean(np.abs(current))
                ),
                "mean_temperature": float(
                    np.mean(temperature)
                ),
                "max_temperature": float(
                    np.max(temperature)
                ),
                "mean_current_load": float(
                    np.mean(current_load)
                ),
                "mean_voltage_load": float(
                    np.mean(voltage_load)
                ),
                "discharge_duration_s": duration_seconds,
                "measurement_count": len(time),
                "soh_percent": soh_percent,
            }
        )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            "No valid discharge records were extracted."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nExtraction complete!")
    print(f"Discharge records: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nFirst five extracted records:")
    print(df.head().to_string(index=False))

    print("\nMeasurement count statistics:")
    print(df["measurement_count"].describe())

    print("\nDischarge duration statistics:")
    print(df["discharge_duration_s"].describe())

    print("\nCapacity statistics:")
    print(df["capacity_ah"].describe())


if __name__ == "__main__":
    main()