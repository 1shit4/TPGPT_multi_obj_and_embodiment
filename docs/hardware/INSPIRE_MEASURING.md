# Measuring the Inspire hand, step by step

Every step says **what** to do, **what you should see**, and **what to send** back. Do them in
order: each one needs the one before it. You need a ruler (mm), a phone camera, the five objects
(apple, banana, mug, pudding box, foam brick), and the emergency stop within reach.

Background (why each number matters): `INSPIRE_HAND.md` §4. Nothing here changes the robot's
settings; the numbers go into the config afterwards, and Claude does that.

---

## Before anything: how to open a terminal

Every time this guide says "open a terminal", open a new terminal window and paste these two lines
first, then press Enter:

```bash
source /opt/ros/jazzy/setup.bash
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh && source ~/thor_ws/install/setup.bash
```

The robot container must already be running (as for the Franka Hand runs).

---

## Step 1 -- let the computer talk to the hand

The hand talks over a small USB box (the "CH340"). Its driver needs a second driver
(`usbserial`) loaded first. That was the "Unknown symbol" error.

1. Open a terminal and paste:
   ```bash
   sudo modprobe usbserial
   sudo insmod ~/ch341/ch341.ko
   ls /dev/ttyUSB0
   ```
2. **You should see** `/dev/ttyUSB0` printed.
3. If you see `File exists`, it was already loaded: that is fine.
4. If `ls` says `No such file`: unplug the hand's USB cable, plug it back in, and run `ls /dev/ttyUSB0` again.

(This must be done again after every reboot. Optional, so it loads by itself at boot -- paste once:
`sudo cp ~/ch341/ch341.ko /lib/modules/$(uname -r)/updates/ && sudo depmod -a && echo ch341 | sudo tee /etc/modules-load.d/ch341.conf`.)

---

## Step 2 -- make the fingers move

1. Is the hand's 24 V power on? (Its power supply must be switched on.)
2. Open a terminal ("**Terminal A**", leave it running) and paste:
   ```bash
   ros2 launch ycb_pointcloud_pipeline inspire_hand.launch.py
   ```
3. **You should see** a line like `Inspire hand on /dev/ttyUSB0 (ID 2, 115200 baud): angles [...]`.
   If you see `no/short reply` instead: the hand is not powered, or the cable is loose.
4. Open another terminal ("**Terminal B**") and paste, one line at a time, watching the hand:
   ```bash
   ros2 service call /inspire_hand/open std_srvs/srv/Trigger
   ros2 service call /inspire_hand/close std_srvs/srv/Trigger
   ros2 service call /inspire_hand/open std_srvs/srv/Trigger
   ```
5. **You should see** the hand open (fingers straight, thumb across the palm), close into a fist,
   then open again. Each command prints `success=True`.

**Send:** "Step 2 works", or the error text.

---

## Step 3 -- where the fingertips go (ruler only, no photos)

**Done 2026-10-05** (readings and what they changed: `INSPIRE_HAND.md` section 7). To repeat it,
read the same points as then: the very end of the middle fingertip (part A), the centres of the
thumb's and the index finger's tips (part B).

Why: the hand reports each finger as a number from 0 (closed) to 1000 (open). The model says where
the fingertip is for each number. Measuring the fingertip at a few numbers checks that. One
millimetre of ruler is about half a degree of finger angle, so a ruler is good enough.

The commands move the fingers **before** the thumb when closing and the thumb **before** the
fingers when opening, so the thumb no longer gets in the way.

**About the thumb's sideways swing.** That is the thumb **rotation**, the last of the six numbers:
- **0** = the thumb points out of the palm, facing the fingers: like a hand holding a can. Every
  grasp uses this.
- **1000** = the thumb swung back flat beside the palm, pointing outwards: like a flat "stop"
  hand. It is used only to get the thumb out of the way: for the pointing pose, and for Part A
  below.

1. **Stop the hand program**: go to Terminal A and press **Ctrl-C** (these commands need the USB
   line to themselves).

**Part A -- the fingers** (the thumb straight and swung out of the way)

2. For each line below: paste it in Terminal B, wait until the hand stops, then measure.
   ```bash
   ros2 run ycb_pointcloud_pipeline inspire_hand set 1000 1000 1000 1000 1000 1000
   ros2 run ycb_pointcloud_pipeline inspire_hand set 750 750 750 750 1000 1000
   ros2 run ycb_pointcloud_pipeline inspire_hand set 500 500 500 500 1000 1000
   ros2 run ycb_pointcloud_pipeline inspire_hand set 250 250 250 250 1000 1000
   ros2 run ycb_pointcloud_pipeline inspire_hand set 0 0 0 0 1000 1000
   ```
3. **The measurement** (the same for every line):
   - Hold the ruler straight down, along the hand, with **0 at the seam where the beige paper
     tape ends and the clear tape begins** (the same 0 as in your photos).
   - Look at the **middle finger's very tip** (the end of its black last segment) and read the
     ruler **at the height of that tip**. Keep your eye level with the tip: looking from above or
     below shifts the reading.
   - When the finger is curled, the tip points sideways or up. Still read the height of the tip.

**Part B -- the thumb** (the fingers open)

4. For each line below: paste it, wait, then measure.
   ```bash
   ros2 run ycb_pointcloud_pipeline inspire_hand set 1000 1000 1000 1000 1000 0
   ros2 run ycb_pointcloud_pipeline inspire_hand set 1000 1000 1000 1000 750 0
   ros2 run ycb_pointcloud_pipeline inspire_hand set 1000 1000 1000 1000 500 0
   ros2 run ycb_pointcloud_pipeline inspire_hand set 1000 1000 1000 1000 250 0
   ```
5. **The measurement:** the straight-line distance from **the thumb's very tip** to **the index
   finger's very tip**. Put the ruler's 0 on one tip and read at the other.

6. Open the hand when done:
   ```bash
   ros2 run ycb_pointcloud_pipeline inspire_hand set 1000 1000 1000 1000 1000 0
   ```

**Send:** the nine numbers, like this:

| | your reading | the model says |
|---|---|---|
| A: fingers 1000 | | about 218 mm |
| A: fingers 750 | | 201 |
| A: fingers 500 | | 162 |
| A: fingers 250 | | 118 |
| A: fingers 0 | | 90 |
| B: thumb 1000 | | 121 |
| B: thumb 750 | | 94 |
| B: thumb 500 | | 72 |
| B: thumb 250 | | 64 |

(Part A's numbers may all be shifted by the same few mm: where the hand's own base face really is
compared with the seam. That shift is itself useful; it is not an error.)

---

## Step 4 -- two lengths with the ruler, and how loose the hand is

**Done 2026-10-05:** the printed mount 16 mm, the palm 81 mm, the hand rocks ~5 mm at the
fingertips across the palm only (`INSPIRE_HAND.md` section 2).

The hand stays open (from step 3.6). Read the ruler to the nearest millimetre.

What you see from the robot down (user, 2026-10-05):
- **beige paper tape**: it covers the robot's flange *and* the 3D-printed mount, so the seam
  between those two is hidden;
- **clear tape**: it covers the hand's own round base ring (52 mm across, 8.1 mm thick by the
  vendor's drawing). The ring is screwed on, but its holes have no thread, so the screws do not
  grip and the tape holds it. It is still slightly loose;
- then the hand's palm and fingers.

The **seam** below means the line where the beige tape ends and the clear tape begins: that is
where the mount ends and the hand's ring begins (the hand model's zero).

- **A. The beige band:** from its top edge to the seam. (It holds part of the robot's flange
  plus the mount, so it is not the mount's thickness by itself. The robot touches in step 7 find
  that. If you know the printed mount's thickness from its print file, send it: that is better
  than any ruler reading.)
- **B. The palm's width:** across the palm at its widest, from the little-finger side to the
  index-finger side.
- (The hand's length is your step 3 reading at 1000, so it is not asked again.)

**How loose (about one minute).** Why: the robot touches in step 7 measure where the fingers are
once. If the hand can rock on the ring, it rocks a different way in every pose (its weight pulls
it), and later grasps are off by that much. Each 1 mm of rock at the ring's edge is about 4 mm at
the fingertips (they are 218 mm away, the ring's edge 26 mm from its middle).
1. Hold the beige band firmly with one hand, so the robot does not move.
2. With the other hand, take the palm and rock it **gently** (no force: as if testing a loose
   tooth) left and right, then forward and back, then try to turn it about its own axis.
3. Watch the middle fingertip **compared with the beige band**: how far does it move between
   the two ends of each rock? About 0, about 1 mm, about 5 mm, or more?

**Send:** A and B in mm, and the three rock answers (left-right, forward-back, turn).

(Expected: B about 81 mm.)

---

## Step 5 -- closing on each object

**Done 2026-10-05:** `simultaneous` holds every object; `sequential` pushes them away. Forces:
400 g, the pudding box 150 (`INSPIRE_HAND.md` section 7).

Why: this checks that each finger stops when it touches, and lets you pick how hard each object
is squeezed.

1. Start the hand program again in Terminal A:
   ```bash
   ros2 launch ycb_pointcloud_pipeline inspire_hand.launch.py
   ```
2. For **each object**, do this:
   1. Hold the object in the hand's opening, between the thumb and the four fingers, about where a
      hand would hold it. Keep your fingers clear of the hand's fingers.
   2. In Terminal B:
      ```bash
      ros2 param set /inspire_hand close_mode simultaneous
      ros2 service call /inspire_hand/close std_srvs/srv/Trigger
      ```
   3. Look at the hand: did it hold the object? Copy the printed answer (it starts with
      `success=True, message='{"mode": ...`).
   4. Open it:
      ```bash
      ros2 service call /inspire_hand/open std_srvs/srv/Trigger
      ```
   5. Do 1-4 again with `ros2 param set /inspire_hand close_mode sequential`.

   `simultaneous` moves the thumb and the fingers together (how GraspGen-X's model of the hand
   closes); `sequential` closes the fingers first and the thumb last. If the thumb gets in the
   fingers' way in `simultaneous`, write that down: it is part of the answer.
3. The squeeze is 400 g per finger. If an object is crushed or slips, try another force and close
   again:
   ```bash
   ros2 param set /inspire_hand force_g 250
   ```
   (Any number from 0 to 1000.)

**Send:** for each object and each mode: did it hold (yes / no / slipped), the printed message, and
the force you liked.

---

## Step 6 -- where the palm faces at home

1. Leave the hand program running in Terminal A (the homing opens the hand through it).
2. Make sure nothing is between the arm and its home pose. Hand on the emergency stop.
3. Open "**Terminal C**" and paste:
   ```bash
   ros2 run ycb_pointcloud_pipeline joint_homing
   ```
   (If the hand program is not running, add ` --ros-args -p open_gripper:=false` at the end.)
4. When the arm has stopped, **stand behind the robot, looking the same way the robot faces**
   (towards the table's far side). Answer:
   - Do the fingers point **down**?
   - Does the palm face **away from you** (forward), **towards you**, **left** or **right**?
   - Is the thumb on your **left** or your **right**?
5. Optional: a photo from the front of the robot, if anything looks unlike the answers above.

**Send:** the three answers.

---

## Step 7 -- touching the table and the slot marks (Claude sends the go-ahead first)

Do this after Claude has put the step-4 numbers into the config: the program below shows you how
far off its estimate is, and that only helps once the estimate is close.

1. Hand program running in Terminal A. In Terminal B, put the hand in the pointing pose (index
   straight, other fingers curled):
   ```bash
   ros2 service call /inspire_hand/point std_srvs/srv/Trigger
   ```
2. In Terminal C, let the arm float (you move it by hand). The extra word keeps the hand pointing:
   ```bash
   ros2 run ycb_pointcloud_pipeline free_drive --ros-args -p open_gripper:=false
   ```
   Type `yes` when it asks. **The arm is now floating: hold it.**
3. Open "**Terminal D**" and paste:
   ```bash
   source ~/venvs/tpgpt/bin/activate
   ros2 run --prefix python3 ycb_pointcloud_pipeline inspire_touch
   ```
4. **Six table touches**, each one done like this:
   1. Guide the arm so the **index fingertip** rests on the table: touching, not pressing.
      Hold the **robot** (its last links), never the hand: the hand rocks a little on its ring
      (step 4), and pushing it moves the fingertip without the robot knowing.
   2. Hold it still. In Terminal D type `t` and press Enter.
   3. Lift it off.

   Change the hand's tilt each time:
   1. straight down;
   2. tilted towards you (about the angle of a clock hand moving 5 minutes);
   3. tilted away;
   4. tilted left;
   5. tilted right;
   6. straight down, but the whole hand turned a quarter turn about its own axis.
5. **The slot marks**: rest the index fingertip on the top-left slot mark (on the top board, the
   same mark the Franka Hand touched), type `p top_left`, Enter. The same on the top-right mark:
   `p top_right`.
6. Type `f` and press Enter: it prints the fit. Then type `q` and press Enter (it saves).
7. Stop free_drive in Terminal C with **Ctrl-C** (the arm then holds where it is).

**Send:** what `f` printed, and the file name it saved (`~/inspire_mount/touches_<time>.json`).

---

## Step 8 -- the hand's weight (later; Claude will send a small tool)

The robot is still set up (in Desk) for the Franka Hand: 0.73 kg. The Inspire hand and adapter
weigh about the same, but their weight sits further from the flange. The robot can weigh them
itself from a few poses held still, so no scale is needed. Claude will send the steps.

---

## What happens with what you send

| step | becomes |
|---|---|
| 3 | the finger angle for each number (the hand model's joint map), and where the hand's base face is |
| 4 | a starting guess for the mount, and whether the hand's play matters |
| 5 | how hard each object is squeezed, and which closing order to use |
| 6 | which way the hand is turned on the flange |
| 7 | the exact adapter thickness, turn and offset (to a couple of mm) |

After that come the grasp tests: the robot tries one grasp at a time on the table.
