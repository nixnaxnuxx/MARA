from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"

IMPORTANT_COLUMN_NAMES = {
    "id",
    "participant_id",
    "subject_id",
    "date",
    "day",
    "day_in_study",
    "sleep_start_day_in_study",
    "phase",
    "cycle_phase",
    "menstrual_phase",
    "label",
}


def inspect_csv(path: Path) -> None:
    """Display schema information without displaying participant values."""

    relative_path = path.relative_to(PROJECT_DIR)
    size_mb = path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 80)
    print(f"FILE: {relative_path}")
    print(f"SIZE: {size_mb:.2f} MB")

    try:
        sample = pd.read_csv(path, nrows=5)
    except Exception as error:
        print(f"ERROR READING FILE: {error}")
        return

    columns = list(sample.columns)
    important_columns = [
        column
        for column in columns
        if column.lower() in IMPORTANT_COLUMN_NAMES
    ]

    print(f"COLUMN COUNT: {len(columns)}")
    print("COLUMNS:")
    for column in columns:
        print(f"  - {column}")

    if important_columns:
        print("POSSIBLE JOIN, TIME OR LABEL COLUMNS:")
        for column in important_columns:
            print(f"  - {column}")


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {RAW_DIR}"
        )

    csv_files = sorted(RAW_DIR.rglob("*.csv"))

    print("=" * 80)
    print("MOSAIC-PHASE DATASET INVENTORY")
    print("=" * 80)
    print(f"Raw data folder: {RAW_DIR}")
    print(f"CSV files found: {len(csv_files)}")

    if not csv_files:
        print(
            "\nNo CSV files were found.\n"
            "Extract the mcPHASES files into data/raw and run this script again."
        )
        return

    for csv_file in csv_files:
        inspect_csv(csv_file)

    print("\n" + "=" * 80)
    print("INVENTORY COMPLETE")
    print("=" * 80)
    print("No participant-level values were printed.")


if __name__ == "__main__":
    main()