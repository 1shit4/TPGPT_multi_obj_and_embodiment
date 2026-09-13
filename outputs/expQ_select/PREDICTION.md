# Prediction registered before Experiment Q was run

Recorded from `outputs/expQ_select` (selection only, no physics) so it
cannot be written after the outcome is known.

Experiment O pooled 60 cells and found the chosen grasp's approach gap
from the demonstration separating cleanly: **21 cells placed at a median
6.1 deg, maximum 37.8; 39 missed at a median 73.9, minimum 2.8.** No cell
that placed exceeded 37.8 deg. The new filter set never looks at the
demonstration, so nothing in it bounds that gap -- it bounds the angle
from *straight down*, which is a fact about the table.

| hand | object | chosen | score | gap from demo | offset | tried | O run ii gap | O run iii gap | changed grasp? |
|---|---|---|---|---|---|---|---|---|---|
| yumi | cereal | #10 | 0.860 | **75.8** | 62.6 mm | 1 | 13.5 | 1.7 | yes |
| yumi | milk | #70 | 0.614 | **60.9** | 10.1 mm | 3 | 37.8 | 1.0 | yes |
| yumi | can | #95 | 0.449 | **84.5** | 9.2 mm | 1 (fell back) | 2.8 | 1.8 | yes |
| yumi | bread | #18 | 0.703 | **3.6** | 20.9 mm | 1 | 29.6 | 3.6 | no |
| xarm | cereal | #72 | 0.688 | **12.5** | 11.4 mm | 1 | 17.0 | 4.9 | yes |
| xarm | milk | #21 | 0.841 | **52.4** | 11.2 mm | 1 | 40.3 | 6.1 | yes |
| xarm | can | #35 | 0.774 | **17.6** | 14.9 mm | 2 | 21.2 | 91.8 | yes |
| xarm | bread | #79 | 0.603 | **3.9** | 12.6 mm | 1 | 44.6 | 5.4 | yes |
| panda | cereal | #76 | 0.626 | **14.5** | 6.2 mm | 1 | 16.5 | 6.6 | yes |
| panda | milk | #34 | 0.715 | **65.3** | 14.4 mm | 2 | 3.2 | 3.2 | yes |
| panda | can | #0 | 0.880 | **14.8** | 6.3 mm | 1 | 14.8 | 87.0 | no |
| panda | bread | #51 | 0.601 | **3.3** | 22.4 mm | 1 | 3.3 | 3.3 | no |
| robotiq85 | cereal | #65 | 0.681 | **30.3** | 10.8 mm | 1 | 12.7 | 6.1 | yes |
| robotiq85 | milk | #49 | 0.677 | **47.6** | 11.0 mm | 2 | 32.0 | 29.8 | yes |
| robotiq85 | can | #0 | 0.936 | **5.6** | 14.5 mm | 1 | 5.6 | 82.6 | no |
| robotiq85 | bread | #70 | 0.579 | **6.7** | 9.1 mm | 1 | 10.5 | 4.0 | yes |
| robotiq140 | cereal | #24 | 0.844 | **7.7** | 12.6 mm | 1 | 14.1 | 4.3 | yes |
| robotiq140 | milk | #24 | 0.903 | **7.2** | 9.9 mm | 1 | 7.2 | 7.2 | no |
| robotiq140 | can | #31 | 0.849 | **15.3** | 2.7 mm | 1 | 30.0 | 73.9 | yes |
| robotiq140 | bread | #9 | 0.781 | **9.9** | 14.6 mm | 1 | 9.9 | 4.6 | no |

**The change is not inert.** 14 of 20 cells execute a candidate
neither of Experiment O's two surviving runs chose, so this is genuinely a
different selection rule and not the same one under another name.

**Prediction: the 6 cells past Experiment O's 37.8 deg boundary fail.**

- panda/milk (65.3 deg)
- robotiq85/milk (47.6 deg)
- xarm/milk (52.4 deg)
- yumi/can (84.5 deg)
- yumi/cereal (75.8 deg)
- yumi/milk (60.9 deg)

If all 6 fail, the ceiling on this configuration is 14/20. If some place, Experiment O's boundary was a property of
its own grasp set rather than of the method, and that is worth more than
the success rate.
