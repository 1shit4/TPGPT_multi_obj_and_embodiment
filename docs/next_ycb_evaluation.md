# Next YCB evaluation design (not part of the 120-cell result)

The current evaluation measures a local four-object domain. It does not validate the 74-object sweep. Broader development and evaluation must remain separate from its reported result.

## Fixed candidate object set and asset gate

Use this ten-object set for the next development campaign, then freeze its asset hashes before drawing evaluation conditions. The choice covers boxes, cans, a curved object, a handled vessel, a long tool and a bottle. It deliberately retains the previously difficult sugar, hammer, banana and mug cases. Missing assets are acquisition work, not completed trials or evidence of graspability.

| Repository name | YCB asset | Local scan and built MJCF now | Role |
|---|---|---|---|
| pudding | 008_pudding_box | Available | Small box |
| gelatin | 009_gelatin_box | Available | Narrow box |
| brick | 061_foam_brick | Available | Light rectangular object, modeled as rigid |
| meat | 010_potted_meat_can | Available | Squat can |
| sugar | 004_sugar_box | Available | Tall, heavier box |
| mug | 025_mug | Available | Handle and cavity |
| hammer | 048_hammer | Available | Long object with eccentric mass |
| banana | 011_banana | Available | Curved object |
| tuna | 007_tuna_fish_can | Missing | Flat can |
| mustard | 006_mustard_bottle | Missing | Bottle; narrow-hand compatibility must be measured |

The repository registry contains names and nominal masses for the missing assets; that is not an asset inventory. For each acquired asset record source, file hashes, measured scan extents, mesh scale, configured mass, decomposition settings and every convex part. Preserve scale and mass rather than making an object fit a hand. Report per-gripper incompatibility as refusal within the scheduled denominator. Do not remove an object because its evaluation rate is low.

## Development gate

Reuse the final archived Panda/one-box demonstration; do not collect a target-specific demonstration. Use development pickup poses and shelf locations distinct from the final evaluation draws. Diagnose initial rest stability, grasp establishment, target palm contact, entry motion, carried-object collisions, release and post-release support. Inspect both camera views and contact traces for each new failure type.

Retain the seven campaign grippers only after checking their actual assets and kinematic mappings. In particular, audit the half-roll alignment assumption for the nonsymmetric three-finger hand before including it. A gripper failure is not permission to replace its fingers or change object physics. Any controller or selection development must end before evaluation conditions are sampled.

Compare legacy and strict selection with the same candidate draws, effective poses and transported labels. Preserve refusal outcomes. Strict evaluation must validate the entry connection and cannot restore rejected candidates; its sampled checks still do not certify continuous motion. Compare full target scans with an observed-target-cloud condition explicitly rather than calling both perception settings equivalent.

## Fresh evaluation

After development, publish the ten-object asset inventory, code and source hashes, gripper list, controller settings, filter modes, success criterion and sampling bounds. Draw six new pickup/shelf configurations with a separately recorded seed, shared across all object–gripper pairs. Determine the bounds from development reachability and publish them before the draw; do not reuse E01–E06 from the current local evaluation.

Seven validated grippers × ten fixed objects × six configurations gives 420 scheduled cells per declared executor/filter condition. Run serial CPU physics with per-cell checkpoints and a dedicated candidate cache. Keep one primary method and a paired replay baseline; count each method separately rather than pooling repeated scenes into a larger success claim. Include setup errors, refusals and execution failures in all-scheduled success, and also report conditional execution success, object/gripper/condition breakdowns and paired transitions.

Use the present released, board-supported, position-stable placement criterion unless a replacement is declared before the evaluation. Any wider-workspace or pose-orientation requirement must be named and frozen before results are observed. Render all failure types and representative successes, record reproduction mismatches, and retain failed evaluations. The 74-object expansion can follow this asset and evaluation procedure; it is not implied by the four-object rate.
