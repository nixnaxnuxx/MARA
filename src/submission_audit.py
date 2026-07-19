from __future__ import annotations

import json
import py_compile
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

REQUIRED_SOURCE_FILES = [
    "src/app.py",
    "src/ui_theme.py",
    "src/interactive_components.py",
    "src/synthetic_journey.py",
    "src/risk_coverage_explorer.py",
    "src/robustness_dashboard.py",
    "src/judge_mode.py",
    "src/ai_mara_chat.py",
]

REQUIRED_PUBLIC_ASSETS = [
    "app_assets/validation_risk_coverage.csv",
    "app_assets/validation_risk_coverage_metadata.json",
]

REQUIRED_PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "altair",
    "openai",
]

FORBIDDEN_TRACKED_FILES = [
    ".streamlit/secrets.toml",
    "reports/checkpoint7a_validation_prediction_sets.csv",
    "reports/checkpoint8_final_test_predictions.csv",
]

failures: list[str] = []
warnings: list[str] = []
passes: list[str] = []


def passed(message: str) -> None:
    passes.append(message)
    print(f"[PASS] {message}")


def warned(message: str) -> None:
    warnings.append(message)
    print(f"[WARN] {message}")


def failed(message: str) -> None:
    failures.append(message)
    print(f"[FAIL] {message}")


def check_required_files() -> None:
    print("\n--- Required files ---")

    for relative_path in (
        REQUIRED_SOURCE_FILES
        + REQUIRED_PUBLIC_ASSETS
    ):
        path = PROJECT_DIR / relative_path

        if path.exists():
            passed(relative_path)
        else:
            failed(
                f"Missing required file: "
                f"{relative_path}"
            )


def compile_python_files() -> None:
    print("\n--- Python compilation ---")

    for relative_path in REQUIRED_SOURCE_FILES:
        path = PROJECT_DIR / relative_path

        if not path.exists():
            continue

        try:
            py_compile.compile(
                str(path),
                doraise=True,
            )

            passed(
                f"Compiled {relative_path}"
            )

        except py_compile.PyCompileError as error:
            failed(
                f"Compilation error in "
                f"{relative_path}: {error}"
            )


def check_requirements() -> None:
    print("\n--- Requirements ---")

    requirements_path = (
        PROJECT_DIR
        / "requirements.txt"
    )

    if not requirements_path.exists():
        failed("requirements.txt is missing")
        return

    content = (
        requirements_path
        .read_text(
            encoding="utf-8"
        )
        .lower()
    )

    for package in REQUIRED_PACKAGES:
        if package in content:
            passed(
                f"{package} is listed"
            )
        else:
            failed(
                f"{package} is not listed "
                "in requirements.txt"
            )


def check_validation_assets() -> None:
    print("\n--- Public validation assets ---")

    csv_path = (
        PROJECT_DIR
        / "app_assets"
        / "validation_risk_coverage.csv"
    )

    metadata_path = (
        PROJECT_DIR
        / "app_assets"
        / "validation_risk_coverage_metadata.json"
    )

    if csv_path.exists():
        try:
            dataframe = pd.read_csv(
                csv_path
            )

            required_columns = {
                "Threshold",
                "Decision coverage",
                "Pair-set coverage",
                "No-call rate",
                "Frozen policy",
            }

            missing_columns = (
                required_columns
                - set(dataframe.columns)
            )

            if missing_columns:
                failed(
                    "Validation asset is missing "
                    f"columns: {sorted(missing_columns)}"
                )
            else:
                passed(
                    "Validation aggregate CSV "
                    f"contains {len(dataframe)} "
                    "threshold rows"
                )

            prohibited_columns = {
                "id",
                "study_interval",
                "day_in_study",
                "phase",
            }

            exposed_columns = (
                prohibited_columns
                & set(dataframe.columns)
            )

            if exposed_columns:
                failed(
                    "Public aggregate CSV contains "
                    "row-level columns: "
                    f"{sorted(exposed_columns)}"
                )
            else:
                passed(
                    "Validation aggregate contains "
                    "no participant-level columns"
                )

        except Exception as error:
            failed(
                "Could not validate aggregate CSV: "
                f"{error}"
            )

    if metadata_path.exists():
        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                metadata.get(
                    "contains_identifiers"
                )
                is False
            ):
                passed(
                    "Metadata confirms no identifiers"
                )
            else:
                warned(
                    "Metadata does not explicitly "
                    "confirm identifier removal"
                )

            if (
                metadata.get(
                    "frozen_threshold"
                )
                == 0.62
            ):
                passed(
                    "Frozen threshold is 0.62"
                )
            else:
                failed(
                    "Unexpected frozen threshold "
                    "in metadata"
                )

        except Exception as error:
            failed(
                "Could not validate metadata JSON: "
                f"{error}"
            )


def get_tracked_files() -> set[str] | None:
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
            ],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    return {
        line.strip().replace(
            "\\",
            "/",
        )
        for line in result.stdout.splitlines()
        if line.strip()
    }


def check_git_safety() -> None:
    print("\n--- Git and data safety ---")

    tracked_files = get_tracked_files()

    if tracked_files is None:
        warned(
            "Git repository was not detected; "
            "check secrets and restricted files "
            "manually"
        )
        return

    for relative_path in FORBIDDEN_TRACKED_FILES:
        normalized = relative_path.replace(
            "\\",
            "/",
        )

        if normalized in tracked_files:
            failed(
                f"Sensitive file is tracked: "
                f"{normalized}"
            )
        else:
            passed(
                f"Not tracked: {normalized}"
            )

    suspicious_files: list[str] = []

    for relative_path in tracked_files:
        path = PROJECT_DIR / relative_path

        if (
            not path.exists()
            or not path.is_file()
        ):
            continue

        if path.suffix.lower() not in {
            ".py",
            ".toml",
            ".txt",
            ".md",
            ".json",
            ".yaml",
            ".yml",
        }:
            continue

        try:
            content = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        except OSError:
            continue

        if "sk-" in content:
            suspicious_files.append(
                relative_path
            )

    if suspicious_files:
        failed(
            "Possible API key text found in "
            "tracked files: "
            + ", ".join(
                suspicious_files
            )
        )
    else:
        passed(
            "No obvious API key prefixes "
            "found in tracked text files"
        )


def check_gitignore() -> None:
    print("\n--- Gitignore ---")

    gitignore_path = (
        PROJECT_DIR
        / ".gitignore"
    )

    if not gitignore_path.exists():
        failed(".gitignore is missing")
        return

    content = (
        gitignore_path
        .read_text(
            encoding="utf-8"
        )
        .replace(
            "\\",
            "/",
        )
    )

    if (
        ".streamlit/secrets.toml"
        in content
    ):
        passed(
            "Streamlit secrets are ignored"
        )
    else:
        failed(
            ".streamlit/secrets.toml "
            "is not listed in .gitignore"
        )


def check_dependency_health() -> None:
    print("\n--- Dependency health ---")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "check",
        ],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )

    output = (
        result.stdout.strip()
        or result.stderr.strip()
    )

    if result.returncode == 0:
        passed(
            output
            or "No broken dependencies"
        )
    else:
        failed(
            f"Dependency problem: {output}"
        )


def print_summary() -> None:
    print("\n" + "=" * 60)
    print("MARA SUBMISSION AUDIT")
    print("=" * 60)

    print(
        f"Passed:   {len(passes)}"
    )

    print(
        f"Warnings: {len(warnings)}"
    )

    print(
        f"Failures: {len(failures)}"
    )

    if warnings:
        print("\nWarnings:")

        for message in warnings:
            print(f"  - {message}")

    if failures:
        print("\nFailures:")

        for message in failures:
            print(f"  - {message}")

        print(
            "\nResolve all failures before "
            "submission."
        )

    else:
        print(
            "\nNo blocking submission "
            "problems were detected."
        )


def main() -> None:
    check_required_files()
    compile_python_files()
    check_requirements()
    check_validation_assets()
    check_gitignore()
    check_git_safety()
    check_dependency_health()
    print_summary()

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()