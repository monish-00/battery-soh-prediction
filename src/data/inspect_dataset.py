from pathlib import Path
from collections import Counter

from scipy.io import loadmat


# 1. Locate the dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Nasa-Battery-Dataset"
    / "1. BatteryAgingARC-FY08Q4"
    / "B0005.mat"
)


# 2. Load the battery data.
mat_data = loadmat(DATASET_FILE)
battery = mat_data["B0005"][0, 0]
cycles = battery["cycle"][0]


# 3. Count the different experiment types.
cycle_types = []

for cycle in cycles:
    cycle_type = str(cycle["type"][0])
    cycle_types.append(cycle_type)

print("Total cycles:", len(cycles))

print("\nExperiment type counts:")

for name, count in Counter(cycle_types).items():
    print(f"{name}: {count}")


# 4. Inspect the first discharge cycle.
print("\nSearching for a discharge cycle...")

for index, cycle in enumerate(cycles):
    cycle_type = str(cycle["type"][0]).lower()

    if cycle_type == "discharge":
        print(f"\nFirst discharge cycle found at index: {index}")

        measurement_data = cycle["data"][0, 0]

        print("\nFields inside the discharge measurements:")

        for field_name in measurement_data.dtype.names:
            value = measurement_data[field_name]

            print(
                f"{field_name}: "
                f"shape={value.shape}, "
                f"dtype={value.dtype}"
            )

        break
else:
    print("No discharge cycle was found.")