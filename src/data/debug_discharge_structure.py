from pathlib import Path

from scipy.io import loadmat
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Nasa-Battery-Dataset"
    / "1. BatteryAgingARC-FY08Q4"
    / "B0005.mat"
)


def describe_value(name, value):
    """Print the type, shape, and sample of a MATLAB field."""

    print(f"\nField: {name}")
    print(f"Python type: {type(value)}")

    if isinstance(value, np.ndarray):
        print(f"Shape: {value.shape}")
        print(f"Data type: {value.dtype}")

        if value.size > 0:
            print("First few values:")
            print(value.flatten()[:5])

    elif isinstance(value, (str, int, float, np.number)):
        print(f"Value: {value}")

    else:
        print(f"Value: {value}")


def main():
    if not MAT_FILE.exists():
        print(f"MATLAB file not found:\n{MAT_FILE}")
        return

    print("Loading original NASA dataset...")

    dataset = loadmat(
        MAT_FILE,
        squeeze_me=True,
        struct_as_record=False,
    )

    battery = dataset["B0005"]
    cycles = np.atleast_1d(battery.cycle)

    print(f"\nTotal cycle records: {len(cycles)}")

    discharge_count = 0

    for index, cycle in enumerate(cycles):
        cycle_type = str(cycle.type).strip().lower()

        if cycle_type != "discharge":
            continue

        discharge_count += 1

        print("\n" + "=" * 60)
        print(f"DISCHARGE RECORD {discharge_count}")
        print(f"Position in cycle array: {index}")
        print(f"Cycle type: {cycle.type}")

        data = cycle.data

        fields = getattr(data, "_fieldnames", None)

        if fields:
            print("\nAvailable data fields:")

            for field in fields:
                value = getattr(data, field)
                describe_value(field, value)

        else:
            print("\nCould not identify MATLAB structure fields.")
            print("Data type:", type(data))
            print("Data:", data)

        # Inspect only the first discharge record.
        if discharge_count == 1:
            break

    print("\n" + "=" * 60)
    print("Inspection complete.")


if __name__ == "__main__":
    main()