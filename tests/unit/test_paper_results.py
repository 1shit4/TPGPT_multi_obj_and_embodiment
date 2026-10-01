import json

import pytest

from scripts.paper_results import compare, read_run
from tpgpt.experiments.audit_artifacts import outcome_summary


@pytest.fixture
def campaign(tmp_path):
    cells = [dict(gripper="panda", obj=o) for o in ("meat", "missing_asset")]
    rows = [dict(cell=cells[0], outcome="success", executed=True),
            dict(cell=cells[1], outcome="setup_error", executed=False)]
    config = dict(cells=cells, source_sha256="fixed_source")
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "summary.json").write_text(json.dumps(outcome_summary(rows, 2)))
    for i, row in enumerate(rows):
        directory = tmp_path / f"cell_{i:03d}"
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps(row))
    return tmp_path


def test_report_counts_scene_build_error_in_scheduled_denominator(campaign):
    _, _, rows, summary = read_run(campaign)
    assert len(rows) == 2
    assert summary["overall_success"] == .5
    assert summary["setup_errors"] == 1
    (campaign / "cell_001" / "result.json").unlink()
    with pytest.raises(ValueError, match="incomplete campaign"):
        read_run(campaign)


def test_report_rejects_stale_summary(campaign):
    p = campaign / "summary.json"
    data = json.loads(p.read_text())
    data["successes"] = 2
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disagrees"):
        read_run(campaign)


def test_comparison_distinguishes_new_grasp_draws_from_executor_pairs():
    def run(cache, labels):
        return (None, {"source_sha256": "same"}, [dict(cell={"obj": "meat"}, outcome="success",
                metrics={"candidate_cache": {"candidate_sha256": cache},
                         "transported_labels_sha256": labels})])
    changed = compare(run("a", "x"), run("b", "y"))
    assert changed["transitions_same_source_and_candidates"] == {}
    assert not changed["pairs"][0]["same_labels"]
    paired = compare(run("a", "x"), run("a", "x"))
    assert paired["pairs"][0]["same_labels"]
    assert paired["transitions_same_source_and_candidates"] == {"success -> success": 1}
