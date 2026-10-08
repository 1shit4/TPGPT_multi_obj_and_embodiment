# The pick grasp's yaw, the twin rule and the destination's direction: when the map folds

A companion to `PLATE_YAW_EXPLAINED.md`, drawn the same way: grey = the demonstration, green =
where the cubes must go, violet = the demonstration after the map's first step (one turn and shift
of everything), orange = what the second step must still move, or space folded. Regenerate with
`~/venvs/tpgpt/bin/python scripts/explain_grasp_yaw.py`.

Four test grasps on the apple -- the demonstration's own grasp turned about the vertical by 0, 90,
180 and 270 deg -- through seven cases. **The twin rule** (figure 3): a grasp more than a quarter
turn from the demonstration's is replaced by its half-turned twin (the same grip, the fingers'
roles swapped); the twin points the opposite way, so 135 deg becomes -45 deg and 180 deg becomes 0.

**Start with the summary:**

![](figures/grasp_yaw/60_summary.png)

## The grasps and the twin rule

**What the twin is: a renumbering, nothing moves.** The cube is centred on the grasp point; built from
the opposite arrow, its corners land on each other's places, so the points stay put and only the
numbers -- the pairing with the demonstration's corners -- change. The robot executes the same
physical grasp either way.

![](figures/grasp_yaw/twin_a.png)

![](figures/grasp_yaw/twin_b.png)

![](figures/grasp_yaw/01_demo_grasp.png)

![](figures/grasp_yaw/02_test_grasps.png)

![](figures/grasp_yaw/03_twin_rule.png)

## Shelf, no twin rule

![](figures/grasp_yaw/04_shelf_off_000_cubes.png)

![](figures/grasp_yaw/05_shelf_off_000_space.png)

![](figures/grasp_yaw/06_shelf_off_090_cubes.png)

![](figures/grasp_yaw/07_shelf_off_090_space.png)

![](figures/grasp_yaw/08_shelf_off_180_cubes.png)

![](figures/grasp_yaw/09_shelf_off_180_space.png)

![](figures/grasp_yaw/10_shelf_off_270_cubes.png)

![](figures/grasp_yaw/11_shelf_off_270_space.png)

## Shelf, the planner's twin rule

![](figures/grasp_yaw/12_shelf_planner_000_cubes.png)

![](figures/grasp_yaw/13_shelf_planner_000_space.png)

![](figures/grasp_yaw/14_shelf_planner_090_cubes.png)

![](figures/grasp_yaw/15_shelf_planner_090_space.png)

![](figures/grasp_yaw/16_shelf_planner_180_cubes.png)

![](figures/grasp_yaw/17_shelf_planner_180_space.png)

![](figures/grasp_yaw/18_shelf_planner_270_cubes.png)

![](figures/grasp_yaw/19_shelf_planner_270_space.png)

## Plate as now, no twin rule

![](figures/grasp_yaw/20_plate_off_000_cubes.png)

![](figures/grasp_yaw/21_plate_off_000_space.png)

![](figures/grasp_yaw/22_plate_off_090_cubes.png)

![](figures/grasp_yaw/23_plate_off_090_space.png)

![](figures/grasp_yaw/24_plate_off_180_cubes.png)

![](figures/grasp_yaw/25_plate_off_180_space.png)

![](figures/grasp_yaw/26_plate_off_270_cubes.png)

![](figures/grasp_yaw/27_plate_off_270_space.png)

## Plate as now, the planner's twin rule

![](figures/grasp_yaw/28_plate_planner_000_cubes.png)

![](figures/grasp_yaw/29_plate_planner_000_space.png)

![](figures/grasp_yaw/30_plate_planner_090_cubes.png)

![](figures/grasp_yaw/31_plate_planner_090_space.png)

![](figures/grasp_yaw/32_plate_planner_180_cubes.png)

![](figures/grasp_yaw/33_plate_planner_180_space.png)

![](figures/grasp_yaw/34_plate_planner_270_cubes.png)

![](figures/grasp_yaw/35_plate_planner_270_space.png)

## Plate, placed yaw turned, no twin rule

![](figures/grasp_yaw/36_plateyaw_off_000_cubes.png)

![](figures/grasp_yaw/37_plateyaw_off_000_space.png)

![](figures/grasp_yaw/38_plateyaw_off_090_cubes.png)

![](figures/grasp_yaw/39_plateyaw_off_090_space.png)

![](figures/grasp_yaw/40_plateyaw_off_180_cubes.png)

![](figures/grasp_yaw/41_plateyaw_off_180_space.png)

![](figures/grasp_yaw/42_plateyaw_off_270_cubes.png)

![](figures/grasp_yaw/43_plateyaw_off_270_space.png)

## Plate, placed yaw turned, the planner's twin rule

![](figures/grasp_yaw/44_plateyaw_planner_000_cubes.png)

![](figures/grasp_yaw/45_plateyaw_planner_000_space.png)

![](figures/grasp_yaw/46_plateyaw_planner_090_cubes.png)

![](figures/grasp_yaw/47_plateyaw_planner_090_space.png)

![](figures/grasp_yaw/48_plateyaw_planner_180_cubes.png)

![](figures/grasp_yaw/49_plateyaw_planner_180_space.png)

![](figures/grasp_yaw/50_plateyaw_planner_270_cubes.png)

![](figures/grasp_yaw/51_plateyaw_planner_270_space.png)

## Plate, placed yaw turned, twin rule against the turned demonstration

![](figures/grasp_yaw/52_plateyaw_turned_000_cubes.png)

![](figures/grasp_yaw/53_plateyaw_turned_000_space.png)

![](figures/grasp_yaw/54_plateyaw_turned_090_cubes.png)

![](figures/grasp_yaw/55_plateyaw_turned_090_space.png)

![](figures/grasp_yaw/56_plateyaw_turned_180_cubes.png)

![](figures/grasp_yaw/57_plateyaw_turned_180_space.png)

![](figures/grasp_yaw/58_plateyaw_turned_270_cubes.png)

![](figures/grasp_yaw/59_plateyaw_turned_270_space.png)
