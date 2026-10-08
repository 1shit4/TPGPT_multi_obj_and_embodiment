# The pick grasp's yaw and the destination's direction: when the map folds

A companion to `PLATE_YAW_EXPLAINED.md`, drawn the same way: grey = the demonstration, green =
where the cubes must go, violet = the demonstration after the map's first step (one turn and shift
of everything), orange = what the second step must still move, or space folded. Regenerate with
`~/venvs/tpgpt/bin/python scripts/explain_grasp_yaw.py`.

Four test grasps on the apple -- the demonstration's own grasp turned about the vertical by 0, 90,
180 and 270 deg -- through four cases. The planner relabels a grasp more than a quarter turn from
the demonstration's as its half-turned twin (the same grip, the fingers' roles swapped; for the
Inspire the demonstration's jaw is described turned instead), so at the shelf no pick cube is ever
more than about a quarter turn off: that is why yaw alone never folds the shelf. For the plate the
first step turns the demonstration round, the twin chosen against the unturned demonstration is
then the wrong one for grasps near 0 and 180 deg, and the fourth case chooses it against the
turned demonstration.

**Start with the summary:**

![](figures/grasp_yaw/35_summary.png)

## The grasps

![](figures/grasp_yaw/01_demo_grasp.png)

![](figures/grasp_yaw/02_test_grasps.png)

## Shelf

![](figures/grasp_yaw/03_shelf_000_cubes.png)

![](figures/grasp_yaw/04_shelf_000_space.png)

![](figures/grasp_yaw/05_shelf_090_cubes.png)

![](figures/grasp_yaw/06_shelf_090_space.png)

![](figures/grasp_yaw/07_shelf_180_cubes.png)

![](figures/grasp_yaw/08_shelf_180_space.png)

![](figures/grasp_yaw/09_shelf_270_cubes.png)

![](figures/grasp_yaw/10_shelf_270_space.png)

## Plate as now

![](figures/grasp_yaw/11_plate_000_cubes.png)

![](figures/grasp_yaw/12_plate_000_space.png)

![](figures/grasp_yaw/13_plate_090_cubes.png)

![](figures/grasp_yaw/14_plate_090_space.png)

![](figures/grasp_yaw/15_plate_180_cubes.png)

![](figures/grasp_yaw/16_plate_180_space.png)

![](figures/grasp_yaw/17_plate_270_cubes.png)

![](figures/grasp_yaw/18_plate_270_space.png)

## Plate, placed yaw turned

![](figures/grasp_yaw/19_plate_yaw_000_cubes.png)

![](figures/grasp_yaw/20_plate_yaw_000_space.png)

![](figures/grasp_yaw/21_plate_yaw_090_cubes.png)

![](figures/grasp_yaw/22_plate_yaw_090_space.png)

![](figures/grasp_yaw/23_plate_yaw_180_cubes.png)

![](figures/grasp_yaw/24_plate_yaw_180_space.png)

![](figures/grasp_yaw/25_plate_yaw_270_cubes.png)

![](figures/grasp_yaw/26_plate_yaw_270_space.png)

## Plate, placed yaw turned and twin chosen against the turned demonstration

![](figures/grasp_yaw/27_plate_roll_000_cubes.png)

![](figures/grasp_yaw/28_plate_roll_000_space.png)

![](figures/grasp_yaw/29_plate_roll_090_cubes.png)

![](figures/grasp_yaw/30_plate_roll_090_space.png)

![](figures/grasp_yaw/31_plate_roll_180_cubes.png)

![](figures/grasp_yaw/32_plate_roll_180_space.png)

![](figures/grasp_yaw/33_plate_roll_270_cubes.png)

![](figures/grasp_yaw/34_plate_roll_270_space.png)
