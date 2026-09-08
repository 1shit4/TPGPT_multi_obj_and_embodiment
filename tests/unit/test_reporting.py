"""Reports and the index over ``outputs/``.

The point of these is that a report never silently omits what went wrong. A
page that renders beautifully while leaving out the failure reason, or an index
that quietly skips an experiment, is worse than no report at all -- it makes a
campaign look cleaner than it was.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tpgpt.experiments.pipeline import OUTCOMES, RunResult
from tpgpt.grasp.filters import FilterFunnel
from tpgpt.reporting.html import MANIFEST, write_index, write_manifest, write_report


def make_result(outcome="success", **kwargs) -> RunResult:
    result = RunResult(
        prompt="put the cereal box on the top shelf middle",
        gripper="panda",
        shelf_variant="cubby",
        seed=0,
        outcome=outcome,
        object_name="cereal",
        slot="top_middle",
        metrics={"placement_error_xy": 0.008, "steps": 310, "n_keypoints": 18,
                 "keypoint_residual": 1e-9, "fraction_positive": 1.0, "min_det": 0.67},
    )
    for key, value in kwargs.items():
        setattr(result, key, value)
    return result


class TestRunReport:
    def test_a_successful_run_reads_as_successful(self, tmp_path):
        page = write_report(make_result(), tmp_path).read_text()
        assert "Task completed" in page
        assert "cereal" in page and "top middle" in page

    @pytest.mark.parametrize("outcome", [o for o in OUTCOMES if o != "success"])
    def test_every_failure_reason_is_explained_in_words(self, tmp_path, outcome):
        """A reader should never see a bare identifier like
        ``map_not_a_diffeomorphism`` with nothing to explain it."""
        result = make_result(outcome, detail="something specific went wrong")
        page = write_report(result, tmp_path / outcome).read_text()
        assert "Task failed" in page
        assert "something specific went wrong" in page
        assert outcome not in page or "The " in page

    def test_assumptions_are_stated_not_buried(self, tmp_path):
        result = make_result(assumptions=["the middle slot was used"])
        page = write_report(result, tmp_path).read_text()
        assert "Assumed" in page and "middle slot" in page

    def test_the_filter_funnel_appears_with_its_fallbacks(self, tmp_path):
        funnel = FilterFunnel()
        funnel.add("visibility", "saw it", np.arange(100), np.arange(80))
        funnel.add("collision", "clear of the shelf", np.arange(80), np.arange(80),
                   fallback=True)
        page = write_report(make_result(funnel=funnel), tmp_path).read_text()
        assert "visibility" in page and "100 &rarr; 80" in page
        assert "fell back" in page

    def test_a_report_is_self_contained(self, tmp_path):
        """Images are embedded, so a report can be moved or emailed."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_path = tmp_path / "grasp.png"
        plt.figure(); plt.plot([0, 1], [0, 1]); plt.savefig(figure_path); plt.close()
        page = write_report(make_result(), tmp_path, {"grasp": figure_path}).read_text()
        assert "data:image/png;base64," in page
        assert str(figure_path) not in page

    def test_the_stage_breakdown_names_the_first_failure(self, tmp_path):
        """A reader who stops at the first cross must already know what broke."""
        from tpgpt.experiments.diagnose import STAGES, Diagnosis, Stage

        diagnosis = Diagnosis(stages=[
            Stage("approach", True, "arrived"),
            Stage("reach", True, "arrived"),
            Stage("grasp", False, "it closed on nothing"),
            *[Stage(n, False, "not reached") for n in STAGES[3:]],
        ])
        page = write_report(
            make_result("placed_in_the_wrong_place", diagnosis=diagnosis), tmp_path
        ).read_text()
        assert "it closed on nothing" in page
        assert "First thing that went wrong" in page
        # The stages after the failure are shown as not reached, never omitted:
        # an absent row in a table reads as "passed".
        assert page.count("not reached") == len(STAGES) - 3

    def test_a_run_with_no_trace_says_so_rather_than_looking_clean(self, tmp_path):
        page = write_report(make_result(), tmp_path).read_text()
        assert "No step-by-step trace" in page

    def test_a_missing_figure_is_omitted_not_broken(self, tmp_path):
        page = write_report(make_result(), tmp_path, {"grasp": tmp_path / "absent.png"})
        assert "absent.png" not in page.read_text()


class TestIndex:
    def test_every_manifest_becomes_an_entry(self, tmp_path):
        for name in ("alpha", "beta"):
            write_manifest(tmp_path / name, title=f"{name} campaign",
                           description="what it was", runs=[])
        page = write_index(tmp_path).read_text()
        assert "alpha campaign" in page and "beta campaign" in page
        assert "from 2 manifest files" in page

    def test_run_outcomes_are_summarised(self, tmp_path):
        write_manifest(
            tmp_path / "camp", title="c", description="d",
            runs=[{"label": "box", "outcome": "success", "placement_error_xy": 0.01},
                  {"label": "contacts", "outcome": "policy_stalled"}],
        )
        page = write_index(tmp_path).read_text()
        assert "1 of 2 runs succeeded" in page
        assert "policy_stalled" in page

    def test_the_index_says_how_far_the_runs_got(self, tmp_path):
        """A campaign's headline is where it broke, not just how many passed."""
        passed_to_lift = {"approach": True, "reach": True, "grasp": True,
                          "lift": False, "carry": False, "place": False,
                          "settle": False}
        all_passed = dict.fromkeys(passed_to_lift, True)
        write_manifest(
            tmp_path / "camp", title="c", description="d",
            runs=[{"label": f"r{i}",
                   "outcome": "success" if i == 0 else "policy_stalled",
                   "blame": "success" if i == 0 else "lift",
                   "stages": all_passed if i == 0 else passed_to_lift}
                  for i in range(4)],
        )
        page = write_index(tmp_path).read_text()
        assert "How far the runs got" in page
        assert "4/4" in page   # every run reached 'grasp'
        assert "1/4" in page   # only the success got past 'lift'
        assert "broke at" in page and "lift" in page

    def test_the_funnel_is_derived_from_the_rows_not_a_stored_summary(self, tmp_path):
        """A stored tally can disagree with the rows beside it -- and did, for
        campaigns recorded before a successful run counted as passing every
        stage. The index recomputes rather than trusting it."""
        write_manifest(
            tmp_path / "c", title="c",
            settings={"stages": {"approach": 9, "reach": 9, "grasp": 9,
                                 "lift": 9, "carry": 9, "place": 9, "settle": 9}},
            runs=[{"label": "a", "outcome": "success",
                   "stages": {"approach": True, "reach": False, "grasp": True,
                              "lift": True, "carry": True, "place": True,
                              "settle": True}},
                  {"label": "b", "outcome": "policy_stalled",
                   "stages": {"approach": True, "reach": False}}],
        )
        page = write_index(tmp_path).read_text()
        # One success plus one run that reached 'approach' only.
        assert "2/2" in page   # approach
        assert "1/2" in page   # everything from reach on
        assert "9/2" not in page

    def test_a_long_campaign_does_not_silently_drop_runs(self, tmp_path):
        write_manifest(
            tmp_path / "big", title="big",
            runs=[{"label": f"r{i}", "outcome": "success"} for i in range(60)],
        )
        page = write_index(tmp_path).read_text()
        assert "12 further runs" in page and "rows.json" in page

    def test_an_empty_outputs_tree_still_renders(self, tmp_path):
        assert "No manifests found" in write_index(tmp_path).read_text()

    def test_a_corrupt_manifest_does_not_break_the_index(self, tmp_path):
        write_manifest(tmp_path / "good", title="good", runs=[])
        (tmp_path / "bad").mkdir()
        (tmp_path / "bad" / MANIFEST).write_text("{not json")
        page = write_index(tmp_path).read_text()
        assert "good" in page

    def test_manifests_survive_numpy_values(self, tmp_path):
        """Campaign rows carry numpy scalars; json cannot serialise them."""
        path = write_manifest(
            tmp_path / "c", title="c",
            runs=[{"label": "x", "outcome": "success",
                   "placement_error_xy": np.float64(0.012), "n": np.int64(3)}],
        )
        data = json.loads(path.read_text())
        assert data["runs"][0]["placement_error_xy"] == pytest.approx(0.012)


class TestManifestProvenance:
    """Every manifest must be able to name the code that produced it.

    Eight campaigns and 140 runs were deleted because none of them could
    (``ROBOTICS_NOTES`` section 7.26). Recording it is therefore opt-*out*: the
    failure mode being guarded against is forgetting, so a caller has to say
    explicitly that it does not want provenance rather than merely omit it.
    """

    def test_a_manifest_records_provenance_without_being_asked(self, tmp_path):
        from tpgpt.reporting.html import write_manifest

        path = write_manifest(tmp_path / "camp", title="t", runs=[])
        info = json.loads(path.read_text())["provenance"]
        assert {"commit", "dirty", "untracked_code", "reproducible"} <= set(info)

    def test_it_can_be_declined_where_the_answer_is_meaningless(self, tmp_path):
        from tpgpt.reporting.html import write_manifest

        path = write_manifest(
            tmp_path / "camp", title="t", runs=[], record_provenance=False
        )
        assert "provenance" not in json.loads(path.read_text())

    def test_an_explicit_value_is_not_overwritten(self, tmp_path):
        from tpgpt.reporting.html import write_manifest

        path = write_manifest(
            tmp_path / "camp", title="t", runs=[],
            provenance={"commit": "deadbee", "reproducible": True},
        )
        assert json.loads(path.read_text())["provenance"]["commit"] == "deadbee"

    def test_the_index_says_loudly_when_results_cannot_be_reproduced(self, tmp_path):
        """A reader must see at a glance whether a table is evidence.

        The index this replaces showed a title and a timestamp, so campaigns
        produced by four different states of the code looked identical and
        directly comparable.
        """
        from tpgpt.reporting.html import write_index, write_manifest

        write_manifest(
            tmp_path / "dirty", title="dirty campaign", runs=[],
            provenance={"commit": "abc1234", "commit_short": "abc1234",
                        "dirty": ["tpgpt/a.py"], "untracked_code": ["tpgpt/b.py"],
                        "reproducible": False},
        )
        page = Path(write_index(tmp_path)).read_text()
        assert "Not reproducible" in page and "tpgpt/b.py" in page
        assert "not as evidence" in page

    def test_the_index_marks_a_reproducible_campaign_as_such(self, tmp_path):
        from tpgpt.reporting.html import write_index, write_manifest

        write_manifest(
            tmp_path / "clean", title="clean campaign", runs=[],
            provenance={"commit": "abc", "commit_short": "abc1234",
                        "branch": "main", "committed": "2026-01-01T00:00:00",
                        "dirty": [], "untracked_code": [], "reproducible": True},
        )
        page = Path(write_index(tmp_path)).read_text()
        assert "Reproducible at" in page and "abc1234" in page

    def test_a_manifest_with_no_provenance_at_all_is_flagged(self, tmp_path):
        """Older manifests predate the field; they must not read as clean."""
        from tpgpt.reporting.html import write_index

        d = tmp_path / "old"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"title": "old", "runs": []}))
        assert "No provenance recorded" in Path(write_index(tmp_path)).read_text()


class TestSettingsRendering:
    """The index must separate what a campaign varied from what it held fixed."""

    def test_varied_and_fixed_are_labelled_distinctly(self):
        from tpgpt.reporting.html import _setting_rows

        rows = dict(_setting_rows({
            "varied": {"gripper": ["panda", "umi"]},
            "fixed": {"obj": "can", "keypoints": ["box"]},
        }))
        assert rows["varied: gripper"] == "panda, umi"
        assert rows["fixed: obj"] == "can"
        assert rows["fixed: keypoints"] == "box"

    def test_the_older_flat_shape_still_renders(self):
        """Manifests written before the settings/results split must not break."""
        from tpgpt.reporting.html import _setting_rows

        assert dict(_setting_rows({"runs": 12, "succeeded": 8}))["runs"] == 12

    def test_a_long_list_of_values_is_summarised_rather_than_dumped(self):
        from tpgpt.reporting.html import _setting_rows

        rows = dict(_setting_rows({"varied": {"seed": list(range(10))}, "fixed": {}}))
        assert "more" in rows["varied: seed"]
