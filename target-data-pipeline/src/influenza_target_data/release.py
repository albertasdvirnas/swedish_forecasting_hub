"""Prepare a GitHub Release notification for a newly merged target week."""

from __future__ import annotations

import argparse
from pathlib import Path

from .storage import load_rows
from .weeks import week_sunday


def latest_week(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    rows = load_rows(path)
    weeks = {row["source_year_week"] for row in rows}
    if not weeks:
        return None
    return max(weeks, key=week_sunday)


def is_new_week(previous: str | None, current: str) -> bool:
    return previous is None or week_sunday(current) > week_sunday(previous)


def render_notes(repository: str, week: str) -> str:
    root = f"https://github.com/{repository}"
    return f"""## New influenza target data

Frozen influenza target data through ISO week **{week}** have been merged into
the hub's `main` branch and are ready for the next forecast round.

- [View the target data]({root}/blob/main/target-data/time-series.csv)
- [Download the CSV]({root}/raw/refs/heads/main/target-data/time-series.csv)
- [Read the forecast submission instructions]({root}/blob/main/docs/model-submissions.md)

To receive future weekly notifications, select **Watch → Custom → Releases**
on the repository's main page.
"""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--current", type=Path, required=True)
    result.add_argument("--previous", type=Path)
    result.add_argument("--repository", required=True)
    result.add_argument("--github-output", type=Path, required=True)
    result.add_argument("--notes", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    current = latest_week(args.current)
    if current is None:
        raise SystemExit("Current target-data file contains no rows")
    previous = latest_week(args.previous)
    should_release = is_new_week(previous, current)

    args.github_output.parent.mkdir(parents=True, exist_ok=True)
    with args.github_output.open("a", encoding="utf-8") as handle:
        handle.write(f"should_release={'true' if should_release else 'false'}\n")
        handle.write(f"week={current}\n")

    if should_release:
        args.notes.parent.mkdir(parents=True, exist_ok=True)
        args.notes.write_text(
            render_notes(args.repository, current), encoding="utf-8"
        )
        print(f"Prepared release notification for {current}")
    else:
        print(f"Latest target week remains {current}; no release is needed")


if __name__ == "__main__":
    main()
