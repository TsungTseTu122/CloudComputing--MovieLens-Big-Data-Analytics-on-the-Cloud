"""Utility to summarize the local Git branch and remotes.

Run this script from anywhere inside the repository to quickly discover
whether your local branch is connected to GitHub and whether it has been
pushed.  It is intentionally lightweight so it works on Windows PowerShell
without extra dependencies.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Set


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BranchStatus:
    """Simple representation of the current branch status."""

    name: str
    ahead: bool = False
    behind: bool = False


def run_git_command(*args: str) -> str:
    """Execute a git command relative to the repository root."""

    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed with: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def parse_remotes(lines: Iterable[str]) -> Dict[str, Set[str]]:
    """Collect remote URLs grouped by remote name."""

    remotes: Dict[str, Set[str]] = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        name, url = parts[0], parts[1]
        remotes.setdefault(name, set()).add(url)
    return remotes


def parse_branch_status(line: str) -> BranchStatus:
    """Interpret the first line of `git status -sb` output."""

    if not line.startswith("## "):
        return BranchStatus(name="(detached)")

    summary = line[3:]
    name = summary.split("...", 1)[0]
    ahead = "ahead" in summary
    behind = "behind" in summary
    return BranchStatus(name=name, ahead=ahead, behind=behind)


def describe_branch() -> BranchStatus:
    status_output = run_git_command("status", "-sb")
    first_line = status_output.splitlines()[0] if status_output else ""
    return parse_branch_status(first_line)


def describe_remotes() -> Dict[str, Set[str]]:
    remotes_output = run_git_command("remote", "-v")
    if not remotes_output:
        return {}
    return parse_remotes(remotes_output.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show whether the current branch has been pushed to a remote.",
    )
    parser.parse_args()  # no custom options today; keeps interface extensible

    try:
        branch = describe_branch()
        remotes = describe_remotes()
    except RuntimeError as exc:  # pragma: no cover - surfaced in CLI usage
        print(exc)
        return

    print(f"Repository root: {REPO_ROOT}")
    print(f"Current branch: {branch.name}")

    if not remotes:
        print("No Git remotes detected. Add one with `git remote add origin <url>`.")
        return

    for name, urls in remotes.items():
        for url in sorted(urls):
            print(f"Remote {name}: {url}")

    if branch.ahead and not branch.behind:
        print("This branch has local commits that have not been pushed yet.")
        print("Run `git push` to publish them to the remote.")
    elif branch.behind and not branch.ahead:
        print("The remote branch has newer commits. Run `git pull` to update.")
    elif branch.behind and branch.ahead:
        print("Local and remote branches diverged. Consider pulling with rebase or merging.")
    else:
        print("Local and remote branches are in sync.")


if __name__ == "__main__":
    main()
