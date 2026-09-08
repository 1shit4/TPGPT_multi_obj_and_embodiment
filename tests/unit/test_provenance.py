"""Recording what code produced a result.

These tests exist because of a specific, expensive failure: eight campaigns and
140 runs were deleted because nothing recorded the code behind them
(``ROBOTICS_NOTES`` section 7.26). The properties asserted here are the ones
that would have caught it -- above all, that an **untracked** source file is
treated as fatal, since that was the actual condition and a plain "dirty" check
misses it entirely.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from tpgpt.reporting.provenance import describe, provenance, warn_if_unreproducible


def _git(*args, cwd):
    subprocess.run(("git",) + args, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A tiny git repository with one committed source file."""
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.email", "t@t.t", cwd=tmp_path)
    _git("config", "user.name", "t", cwd=tmp_path)
    (tmp_path / "tpgpt").mkdir()
    (tmp_path / "tpgpt" / "mod.py").write_text("x = 1\n")
    _git("add", "-A", cwd=tmp_path)
    _git("commit", "-qm", "initial", cwd=tmp_path)
    return tmp_path


class TestProvenance:
    def test_a_clean_tree_is_reproducible(self, repo):
        info = provenance(cwd=repo)
        assert info["reproducible"]
        assert info["commit"] and len(info["commit"]) == 40
        assert info["dirty"] == [] and info["untracked_code"] == []

    def test_an_untracked_source_file_makes_a_run_unreproducible(self, repo):
        """The exact condition that invalidated the deleted campaigns.

        ``pipeline.py``, ``filters.py``, ``diagnose.py`` and
        ``run_experiments.py`` had never been committed. They therefore appear
        in no diff and have no history, so a dirty-flag check alone reports the
        tree as clean and the results as trustworthy. They are not.
        """
        (repo / "tpgpt" / "brand_new.py").write_text("y = 2\n")
        info = provenance(cwd=repo)
        assert not info["reproducible"]
        assert info["untracked_code"] == ["tpgpt/brand_new.py"]
        assert info["dirty"] == []          # invisible to a diff, which is the point
        assert "untracked" in describe(info)

    def test_a_modified_tracked_file_makes_a_run_unreproducible(self, repo):
        (repo / "tpgpt" / "mod.py").write_text("x = 999\n")
        info = provenance(cwd=repo)
        assert not info["reproducible"]
        assert info["dirty"] == ["tpgpt/mod.py"]
        assert info["diff_stat"] and "mod.py" in info["diff_stat"]

    def test_untracked_output_does_not_invalidate_a_run(self, repo):
        """Only *code* counts. Every run writes outputs; that is not a defect."""
        (repo / "outputs").mkdir()
        (repo / "outputs" / "result.json").write_text("{}")
        assert provenance(cwd=repo)["reproducible"]

    def test_outside_a_repository_it_reports_unknown_not_clean(self, tmp_path):
        """An absent answer must never look like a passing one.

        Same failure shape as the zeroed ``contact_offset`` of section 7.13: a
        missing measurement that returns a plausible value is worse than one
        that refuses.
        """
        info = provenance(cwd=tmp_path)
        assert info["reproducible"] is False
        assert info["commit"] is None

    def test_the_shape_is_stable_whether_or_not_git_answers(self, repo, tmp_path):
        """A consumer must never have to guess why a field is missing."""
        assert set(provenance(cwd=repo)) <= set(provenance(cwd=tmp_path)) | {"note"}
        for info in (provenance(cwd=repo), provenance(cwd=tmp_path)):
            for key in ("commit", "dirty", "untracked_code", "reproducible",
                        "python", "packages"):
                assert key in info

    def test_it_is_json_serialisable(self, repo):
        """It is written straight into a manifest, so this cannot be assumed."""
        json.dumps(provenance(cwd=repo))

    def test_it_never_raises_when_git_is_unusable(self, tmp_path, monkeypatch):
        """A broken git must not stop an experiment, only downgrade its claim."""
        monkeypatch.setenv("PATH", str(tmp_path))
        info = provenance(cwd=tmp_path)
        assert info["reproducible"] is False


class TestDescribe:
    def test_it_names_the_reason_not_only_the_verdict(self, repo):
        (repo / "tpgpt" / "brand_new.py").write_text("y = 2\n")
        text = describe(provenance(cwd=repo))
        assert "NOT REPRODUCIBLE" in text and "brand_new.py" in text

    def test_a_clean_tree_reads_as_reproducible(self, repo):
        assert describe(provenance(cwd=repo)).startswith("reproducible at")


class TestWarnIfUnreproducible:
    def test_it_warns_but_does_not_stop_a_run_by_default(self, capsys):
        info = warn_if_unreproducible({"reproducible": False, "dirty": ["a.py"]})
        assert info["reproducible"] is False
        assert "NOT REPRODUCIBLE" in capsys.readouterr().out

    def test_strict_refuses_to_run(self):
        with pytest.raises(RuntimeError, match="NOT REPRODUCIBLE"):
            warn_if_unreproducible({"reproducible": False, "dirty": ["a.py"]},
                                   strict=True)

    def test_strict_allows_a_reproducible_run(self):
        info = {"reproducible": True, "commit_short": "abc1234", "branch": "main"}
        assert warn_if_unreproducible(info, strict=True) is info
