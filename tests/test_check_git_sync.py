from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.check_git_sync import BranchStatus, parse_branch_status, parse_remotes


def test_parse_remotes_groups_urls():
    raw = [
        "origin\thttps://example.com/repo.git (fetch)",
        "origin\thttps://example.com/repo.git (push)",
        "upstream\tgit@github.com:org/repo.git (fetch)",
    ]
    remotes = parse_remotes(raw)
    assert remotes == {
        "origin": {"https://example.com/repo.git"},
        "upstream": {"git@github.com:org/repo.git"},
    }


def test_parse_branch_status_determines_branch_and_flags():
    status = parse_branch_status("## work...origin/work [ahead 2]")
    assert status == BranchStatus(name="work", ahead=True, behind=False)


def test_parse_branch_status_handles_detached_head():
    status = parse_branch_status("?? newfile.txt")
    assert status.name == "(detached)"
