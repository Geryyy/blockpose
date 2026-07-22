"""Non-mutating contract checks for the quarantine cleanup helper."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "quarantine_cleanup.sh"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def test_cleanup_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_cleanup_script_help_documents_safe_modes() -> None:
    result = run_script("--help")
    assert result.returncode == 0, result.stderr
    assert "--apply" in result.stdout
    assert "--include-ros-provenance" in result.stdout
    assert "Preview reviewed cleanup candidates by default" in result.stdout


def test_default_preview_does_not_change_git_status() -> None:
    before = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    result = run_script()
    after = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout

    assert result.returncode == 0, result.stderr
    assert "Mode: preview" in result.stdout
    assert "Preview only." in result.stdout
    assert before == after


def test_ros_group_requires_explicit_opt_in() -> None:
    result = run_script("--group", "ros-provenance")
    assert result.returncode != 0
    assert "requires --include-ros-provenance" in result.stderr


def test_script_contains_no_permanent_removal_command() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    forbidden = ("rm ", "rmdir", "unlink", "git rm", "find -delete")
    assert not any(token in script for token in forbidden)


def test_active_inputs_are_not_quarantine_candidates() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    protected = (
        "docs/.obsidian/workspace.json",
        ".openclaude/worktrees",
        "config/calibration",
        "config/transforms_pzs100_initial_pose_1.yaml",
        "assets/pzs100",
    )
    allowlist_prefix = script.split("validate_candidate()", 1)[0]
    assert not any(path in allowlist_prefix for path in protected)
