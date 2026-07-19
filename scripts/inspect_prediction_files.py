from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

SEARCH_DIRECTORIES = [
    PROJECT_DIR / "reports",
    PROJECT_DIR / "artifacts",
    PROJECT_DIR / "outputs",
]

KEYWORDS = (
    "oof",
    "out_of_fold",
    "validation",
    "valid",
    "val",
    "prediction",
    "probability",
    "calibration",
    "router",
)


def is_relevant_file(path: Path) -> bool:
    filename = path.name.lower()

    return any(
        keyword in filename
        for keyword in KEYWORDS
    )


def inspect_csv(path: Path) -> None:
    try:
        dataframe = pd.read_csv(
            path,
            nrows=5,
        )

        print("=" * 80)
        print(f"FILE: {path.relative_to(PROJECT_DIR)}")
        print(f"FORMAT: CSV")
        print(f"COLUMNS ({len(dataframe.columns)}):")

        for column in dataframe.columns:
            print(f"  - {column}")

    except Exception as error:
        print("=" * 80)
        print(f"FILE: {path.relative_to(PROJECT_DIR)}")
        print(f"ERROR: {error}")


def inspect_parquet(path: Path) -> None:
    try:
        dataframe = pd.read_parquet(
            path
        ).head(5)

        print("=" * 80)
        print(f"FILE: {path.relative_to(PROJECT_DIR)}")
        print(f"FORMAT: Parquet")
        print(f"COLUMNS ({len(dataframe.columns)}):")

        for column in dataframe.columns:
            print(f"  - {column}")

    except Exception as error:
        print("=" * 80)
        print(f"FILE: {path.relative_to(PROJECT_DIR)}")
        print(f"ERROR: {error}")


def main() -> None:
    candidates: list[Path] = []

    for directory in SEARCH_DIRECTORIES:
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {
                ".csv",
                ".parquet",
            }:
                continue

            if is_relevant_file(path):
                candidates.append(path)

    candidates = sorted(
        set(candidates)
    )

    if not candidates:
        print(
            "No likely validation or OOF prediction "
            "files were found."
        )

        print(
            "\nCSV and Parquet files currently present:"
        )

        for directory in SEARCH_DIRECTORIES:
            if not directory.exists():
                continue

            for path in sorted(
                directory.rglob("*")
            ):
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in {".csv", ".parquet"}
                ):
                    print(
                        "  - "
                        + str(
                            path.relative_to(
                                PROJECT_DIR
                            )
                        )
                    )

        return

    print(
        f"Found {len(candidates)} possible "
        "prediction file(s).\n"
    )

    for path in candidates:
        if path.suffix.lower() == ".csv":
            inspect_csv(path)

        elif path.suffix.lower() == ".parquet":
            inspect_parquet(path)


if __name__ == "__main__":
    main()