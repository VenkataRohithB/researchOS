from __future__ import annotations

from pathlib import Path

import pytest

from researchos.cli import EXIT_INCOMPLETE, EXIT_OK, EXIT_USAGE_ERROR, main
from researchos.project import Project


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_PROVIDER", "SEARCH_PROVIDER", "MAX_STEPS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))
    return tmp_path / "ws"


def only_project(workspace: Path) -> str:
    (project_id,) = Project.list_ids(workspace)
    return project_id


def test_new_status_steer_resume_list(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-q", "new", "Apple chips", "--level", "engineer"]) == EXIT_OK
    project_id = only_project(workspace)

    assert main(["steer", project_id, "compare with x86"]) == EXIT_OK
    assert main(["-q", "resume", project_id]) == EXIT_OK
    assert main(["status", project_id]) == EXIT_OK
    assert main(["list"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "Status:   completed" in out
    assert "Progress: 8 steps over 2 run(s)" in out
    assert "- compare with x86" in out
    assert f"{project_id}  [completed]" in out


def test_budget_exhaustion_exits_incomplete_and_suggests_resume(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MAX_STEPS", "2")

    assert main(["-q", "new", "topic"]) == EXIT_INCOMPLETE

    assert f"researchos resume {only_project(workspace)}" in capsys.readouterr().out


def test_unknown_project_is_a_usage_error(workspace: Path) -> None:
    assert main(["status", "nope"]) == EXIT_USAGE_ERROR


def test_missing_credentials_is_a_usage_error(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "")

    assert main(["list"]) == EXIT_USAGE_ERROR


def test_runs_build_a_website_and_build_regenerates_it(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["-q", "new", "Nanosheets"]) == EXIT_OK
    project_id = only_project(workspace)
    site = workspace / project_id / "website" / "index.html"
    assert site.is_file()

    site.unlink()
    assert main(["build", project_id]) == EXIT_OK

    assert site.is_file()
    assert f"Website: {site}" in capsys.readouterr().out
