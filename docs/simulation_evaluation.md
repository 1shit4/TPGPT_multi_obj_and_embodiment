# Frozen simulation evaluation

**Completed frozen evaluation: GP 111/120 (92.5%); direct replay 113/120 (94.2%).**

One fixed scripted Panda demonstration transfers to five grippers, four objects and six fresh shared pickup/shelf configurations. All scheduled cells count. Both runs exceed the requested 80% success on at least 100 trials. These are supported, released placement outcomes, not collision-free execution rates.

[Browse synchronized failure and success videos](../outputs/paper_evaluation_v1/evidence_index.html) · [All trial artifacts](../outputs/paper_evaluation_v1/report/index.html) · [Results PDF](../outputs/paper_evaluation_v1/manuscript/simulation_results_preview.pdf)

## Reproducibility

- [Frozen protocol](../outputs/paper_evaluation_v1/protocol.json) and [exact primary cells](../outputs/paper_evaluation_v1/cells_joint_policy.json).
- [Primary checkpoint summary](../outputs/paper_evaluation_v1/joint_policy/summary.json), [subgroup analysis](../outputs/paper_evaluation_v1/analysis.json), and per-cell candidate/filter decisions in `joint_policy/cell_*/result.json`.
- [Single source manifest](../outputs/paper_evaluation_v1/source/source.json), [source demonstration video](../outputs/paper_evaluation_v1/source/demonstration.mp4), and [recording result](../outputs/paper_evaluation_v1/source/recording_result.json). The source scene shelf uses collision-group geometry, hidden in this archived visual-only camera render; the video alone does not show the board. The source is a scripted simulation recording, not a human demonstration.
- [Full suite log](../outputs/paper_evaluation_v1/full_suite.log): 832 passed, seven warnings, zero skips. Code and cell hashes were checked against the frozen protocol.
- [Manuscript LaTeX](simulation_results.tex) and [compiled preview](../outputs/paper_evaluation_v1/manuscript/simulation_results_preview.pdf).

All 120 primary cells executed. There were nine execution failures, no refusals and no setup or diagnostic errors. Seven cells used legacy path fallback; four of those succeeded. None was a forced diagnostic. Final successful placements have median XY error 5.7 mm and 95th percentile 14.8 mm. Success includes released, destination-board-supported, position-stable settling, with every scheduled cell in the denominator.

## Paired results

| Gripper | GP successes / 24 | Replay successes / 24 |
|---|---:|---:|
| Panda | 23 | 23 |
| Robotiq85 | 22 | 24 |
| Robotiq140 | 23 | 24 |
| XArm | 24 | 24 |
| Rethink | 19 | 18 |
| **Total / 120** | **111 (92.5%)** | **113 (94.2%)** |

Both executors succeed in 110 cells and fail in six; replay recovers three GP failures and loses one GP success. All 120 pairs match source, initial scene, candidate cache, effective grasp pose and transported labels. Replay has seven execution failures and no refusals or setup/diagnostic errors. Its seven path fallbacks include six successes. Successful replay placements have median XY error 6.4 mm and 95th percentile 22.9 mm. Replay holds each waypoint for eight control steps and requires four consecutive acquisition contact samples; GP timing and its acquisition gate differ. This is a comparison of complete executor configurations, not an isolated policy-refit ablation.

## Rendered evidence

All 30 normal-selection reproductions match their original outcome and pinned inputs: the union of ten failure pairs and five diverse success controls. All 30 dual-camera timeline sequences were inspected; [manual observations and artifact hashes](../outputs/paper_evaluation_v1/visual_review.json) and [reproduction verification](../outputs/paper_evaluation_v1/evidence_verification.json) are archived. The viewer links each case to videos, event frames, synchronized traces and filter decisions. Reproductions are excluded from the evaluation denominator. Saved final-state views supplement camera occlusions.

## Failure trace review

These observations combine inspected camera sequences with synchronized control-step traces and plots. The final two seconds are the prescribed open-jaw observation, even after a failed/stalled rollout; opening there is not evidence that the policy reached its release phase. They separate measured behavior from causal hypotheses; a checker disagreement alone does not establish which geometry model is correct.

| Cell | Case | Trace evidence | Artifacts |
|---|---|---|
| 007 | panda/gelatin/E02 | No target contact or closure; phase stalls at 0.137. The selected fallback has an unreachable grasp waypoint. | [plot](../outputs/paper_evaluation_v1/trace_review/cell_007.png), [trace](../outputs/paper_evaluation_v1/joint_policy/cell_007/simulation_steps.json), [filter](../outputs/paper_evaluation_v1/joint_policy/cell_007/result.json) |
| 029 | robotiq85/pudding/E06 | Object lifts 74 mm, then phase stalls at 0.385. Selected fallback was rejected for carried-object cloud collision. No robot–shelf/table contact appears in the recorded trace. | [plot](../outputs/paper_evaluation_v1/trace_review/cell_029.png), [trace](../outputs/paper_evaluation_v1/joint_policy/cell_029/simulation_steps.json), [filter](../outputs/paper_evaluation_v1/joint_policy/cell_029/result.json) |
| 031 | robotiq85/gelatin/E02 | Object lifts 185 mm; link 6 contacts the right upper shelf wall and phase stalls at 0.471. Both observed and simulator path diagnostics flagged the selected fallback. | [plot](../outputs/paper_evaluation_v1/trace_review/cell_031.png), [trace](../outputs/paper_evaluation_v1/joint_policy/cell_031/simulation_steps.json), [filter](../outputs/paper_evaluation_v1/joint_policy/cell_031/result.json) |
| 067 | robotiq140/meat/E02 | Initial approach contacts the upper shelf board; phase remains zero and tracking error stays above 130 mm. The selected sampled path was accepted, but its entry connection was unchecked. | [plot](../outputs/paper_evaluation_v1/trace_review/cell_067.png), [trace](../outputs/paper_evaluation_v1/joint_policy/cell_067/simulation_steps.json), [filter](../outputs/paper_evaluation_v1/joint_policy/cell_067/result.json) |
| 108 | rethink/brick/E01 | Grasp detector becomes active briefly, then clears. Hand completes the trajectory; maximum object lift is 3.6 mm. | [plot](../outputs/paper_evaluation_v1/trace_review/cell_108.png), [trace](../outputs/paper_evaluation_v1/joint_policy/cell_108/simulation_steps.json), [filter](../outputs/paper_evaluation_v1/joint_policy/cell_108/result.json) |
| 110 | rethink/brick/E03 | Grasp detector becomes active briefly, then clears. Hand completes the trajectory; maximum object lift is 1.7 mm. | [plot](../outputs/paper_evaluation_v1/trace_review/cell_110.png), [trace](../outputs/paper_evaluation_v1/joint_policy/cell_110/simulation_steps.json), [filter](../outputs/paper_evaluation_v1/joint_policy/cell_110/result.json) |
| 114 | rethink/meat/E01 | Hand completes the trajectory; maximum object lift is 5.4 mm. | [plot](../outputs/paper_evaluation_v1/trace_review/cell_114.png), [trace](../outputs/paper_evaluation_v1/joint_policy/cell_114/simulation_steps.json), [filter](../outputs/paper_evaluation_v1/joint_policy/cell_114/result.json) |
| 118 | rethink/meat/E05 | Hand completes the trajectory; maximum object lift is 7.8 mm. | [plot](../outputs/paper_evaluation_v1/trace_review/cell_118.png), [trace](../outputs/paper_evaluation_v1/joint_policy/cell_118/simulation_steps.json), [filter](../outputs/paper_evaluation_v1/joint_policy/cell_118/result.json) |
| 119 | rethink/meat/E06 | Hand completes the trajectory; maximum object lift is 6.1 mm. | [plot](../outputs/paper_evaluation_v1/trace_review/cell_119.png), [trace](../outputs/paper_evaluation_v1/joint_policy/cell_119/simulation_steps.json), [filter](../outputs/paper_evaluation_v1/joint_policy/cell_119/result.json) |

The Rethink brick E01/E03 videos show contact near the upper object edge followed by the fingers rising while the brick stays on the table. The three shared meat failures similarly leave the can on the table after brief contact/tilting. This supports failed acquisition, without identifying a unique mechanical cause.

Cell **113 (Rethink/brick/E06)** is the replay-only failure: it successfully lifts and carries the brick, then loses it while closure is still commanded. See the [loss sequence](../outputs/paper_evaluation_v1/replay_brick_loss.jpg), [replay trace plot](../outputs/paper_evaluation_v1/trace_review_replay/cell_113.png), and paired videos in the viewer. It must not be grouped with initial missed grasps.

For cell 067, the [pinned non-executing diagnostic](../outputs/paper_evaluation_v1/entry_diagnostic.json) checked **103 entry interpolation poses and still accepted the connection**, with simulator state restored. The real approach contacts the shelf. The [saved final-state view](../outputs/paper_evaluation_v1/evidence_joint_policy/cell_006/final_state.png) and [diagnostic transparent-shelf view](../outputs/paper_evaluation_v1/entry_contact_transparent.png) supplement the occluded side camera. The sampled entry check therefore does not certify this executed connection. Differences in checked versus controlled motion and geometry coverage remain hypotheses. No headline outcome was changed by this diagnostic.

## Contact audit

The [contact audit](../outputs/paper_evaluation_v1/contact_audit.json) examines every primary control-step trace. Twelve trials have robot/gripper contact with shelf/table geometry: ten successful Robotiq140 trials and failed cells 031 and 067. In the successes, the named contact geometries are fingertips or finger pads against the table or shelf board. These are contact records, not measured forces or an automatic classification of forbidden contact. The deepest recorded signed penetration among successful cells is 4.50 mm in cell 060 (Robotiq140/brick/E01); that cell was reproduced and inspected as a normal-selection success control. The placement criterion does not penalize these intermediate contacts. The [replay contact audit](../outputs/paper_evaluation_v1/replay_contact_audit.json) records 14 contact-bearing trials, all successful.

## Interpretation boundaries

Five grippers are mounted on the same Panda arm. Four target objects and the local workspace were chosen during development. Six fresh shared configurations and a dedicated grasp cache define the evaluation; development runs and rendered reproductions must not be pooled with these 120 trials. Target geometry is a full scan at the simulator pose, scene obstacles are observed clouds, and a legacy simulator place-zone filter remains in selection. Discrete checks and fallback execution do not certify collision-free motion. The GP-reference executor does not execute transported compliance.

The [development ledger](simulation_development.md), [original filter audit](grasp_selection_audit.md), and [next ten-object evaluation design](next_ycb_evaluation.md) remain separate from this frozen evaluation.

## Draft corrections

- Describe full target scans at simulator poses; only obstacle clouds are camera observations, and the legacy place-zone filter also uses simulator geometry.
- Report legacy fallback execution explicitly; these rates do not describe strict feasibility selection.
- Describe stride-four sampling with critical endpoints and early rejection, not continuous collision certification. The entry diagnostic has a demonstrated false negative against executed motion.
- Identify the source as one development-selected scripted Panda/box recording, reused unchanged in evaluation. Do not attribute it to a human or the unavailable historical 94/210 experiment.
- Separate development choices, 120 prospective cells per executor, 30 rendered reproductions and non-executing diagnostics. Do not pool their denominators or claim GP superiority over replay.

The [completion receipt](../outputs/paper_evaluation_v1/completion_audit.json) verifies frozen inputs, paired identities, every rendered reproduction, video/trace synchronization, the test log and final document hashes. The PDF pages were visually inspected. Viewer links and JavaScript syntax were checked; browser playback was not automated.
