#!/usr/bin/env python3
"""Inspire RH56 hand over RS485: protocol, the open/close grasp, and a CLI.

No ROS in here, so the same code runs from a terminal and inside
``inspire_hand_node`` (the ROS 2 wrapper the pipeline uses).

The hand on the FR3 is an **RH56DFX-2R, a right hand** (2026-10-04). Commands are
ANGLE_SET register values (0-1000, 1000 = straight/open, 0 = bent)::

    dof            open    close
    little..index  1000    0      the force threshold stops each finger on contact
    thumb bend     1000    0      the force threshold stops it on contact
    thumb rotation 0       0      opposed: pointing out of the palm, facing the fingers

**Every change of pose keeps the thumb and the fingers out of each other's way**
(:meth:`InspireHand.goto`; the user, 2026-10-05: moved together, the thumb obstructs the
fingers). Like a fist: closing, the fingers go first and the thumb bends over them last;
opening, the thumb lifts off first and then the fingers open. The thumb rotates only while
it is straight and the fingers are open.

**open** (:meth:`InspireHand.grasp_open`): right after a close that **held something**
(:attr:`CloseReport.held`), everything opens at once, in one stage: the object is between the thumb and the fingers, so they cannot meet,
and a staged open (the thumb first) leaves the fingers pressing the object alone, which
shifts it at the release (the user, 2026-10-05). Otherwise -- a fist closed on air, any
other pose, the thumb to rotate -- it is ``goto(OPEN_POSE)``.

**close**, two ways (``close_mode``):

* ``simultaneous`` (**the one to use**: the user, 2026-10-05, on all five objects):
  thumb rotation confirmed opposed, then the four fingers and the thumb bend together --
  how GraspGen-X's description closes the hand (``close`` drives the thumb pitch and the
  fingers in one pose). Without an object in between, the thumb can meet the fingers.
* ``sequential``: the four fingers close first, then the thumb. **It pushes the object
  away**: nothing opposes the fingers until the thumb comes (the user, 2026-10-05). Kept
  only to show that.

Either returns a :class:`CloseReport`: which DOFs were stopped by something
(**contact**), which closed all the way (**on air**), which were still moving at the
timeout, and the actuator errors. A close that touches nothing is not a success.

**Something is held** (:attr:`CloseReport.held`) when a finger stopped more than
:data:`CONTACT_MARGIN` beyond where an empty close leaves it (:data:`EMPTY_CLOSE_ANGLES`):
an empty close stops the middle and index fingers on the thumb, so the thumb and a finger
stopping proves nothing. On every hold of 2026-10-05 (17 closes, five objects) a finger
stopped 190-590 units beyond that. The thumb is not used: in a palm grasp it closes over
the fingers just as in the empty fist. **It errs towards "held"**: the empty fist is not
repeatable (at 150 g the middle finger stopped at 140 once and 349 once -- read as held),
and the reference is kept low so that no real grasp reads as empty (the mug's middle finger
stops at 398-425). Its two uses tolerate that: the release checks that the hand opened and
otherwise opens in order, and the executor's ``abort_on_empty_close`` only saves time.

**Contact is judged by where a DOF stopped, not by its STATUS.** The manual (2.4.21) has
STATUS 3 for "stopped on the force threshold", but this hand reports 2 ("reached the
target") for a DOF stopped by an object as well (2026-10-05: every DOF of every grasp on
five objects read 2, stopped 400-800 units short of 0 at 400-800 g). A close commands 0;
a DOF that ends more than :data:`CONTACT_MARGIN` short of it was stopped by something.
The hand reaches a free target to within ~7 units.

Each DOF's angle in the URDF joint the hand model uses comes from the ``joint_map``
block of config/inspire_hand.yaml (:func:`joint_map_from_config`,
:func:`register_to_joint`): linear per DOF, ``q = q_at_0 + (q_at_1000 - q_at_0) * u /
1000``. ``tpgpt.hardware.inspire_model`` reads the same block; a test keeps the two
in agreement.

Register convention (RH56 manual, section 2.4): six shorts per group in the
order little, ring, middle, index, thumb bend, thumb rotation. ``-1`` = leave
that DOF alone, EXCEPT thumb rotation (ANGLE_SET(5) accepts only 0-1000: -1
drove it from 55 to 985 on the real hand), so :meth:`InspireHand.set_angles`
resends its measured angle instead.

Wire protocol (manual 2.2): 115200 8N1, frames
    write: EB 90 id len 12 addrL addrH data... checksum
    read:  EB 90 id 04  11 addrL addrH nbytes  checksum
checksum = low byte of the sum of everything after the EB 90 header.

CLI (the hand in this lab answers on ID 2 at 115200; ``scan`` finds it)::

    ros2 run ycb_pointcloud_pipeline inspire_hand ports
    ros2 run ycb_pointcloud_pipeline inspire_hand scan
    ros2 run ycb_pointcloud_pipeline inspire_hand status
    ros2 run ycb_pointcloud_pipeline inspire_hand --speed 300 open
    ros2 run ycb_pointcloud_pipeline inspire_hand --speed 300 --force 400 close
    ros2 run ycb_pointcloud_pipeline inspire_hand --close-mode simultaneous close
    ros2 run ycb_pointcloud_pipeline inspire_hand set 500 500 500 500 500 0   # hold these (M4)
    ros2 run ycb_pointcloud_pipeline inspire_hand cycle --hold 3
    ros2 run ycb_pointcloud_pipeline inspire_hand --dry-run close   # print frames only

Port, ID, baud, speed, force and close mode default to config/inspire_hand.yaml
(installed copy; ``--config`` for another). Needs ``pyserial`` and ``pyyaml``.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import asdict, dataclass, field

# --- register map (manual section 2.4) ---------------------------------------
HAND_ID_REG = 1000  # register ADDRESS of the ID (1 byte, value default 1, range 1-254);
                    # the ID to talk to is --id, not this
REDU_RATIO = 1002  # 1 byte, baud: 0 = 115200 (default), 1 = 57600, 2 = 19200
CLEAR_ERROR = 1004
VOLTAGE = 1472     # 1 short, read only (unit not stated in the manual)
ANGLE_SET = 1486
FORCE_SET = 1498
SPEED_SET = 1522
ANGLE_ACT = 1546
FORCE_ACT = 1582
ERROR = 1606
STATUS = 1612
TEMP = 1618

DOF_NAMES = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rot")
FINGERS = DOF_NAMES[:4]
BENDING = DOF_NAMES[:5]                # the DOFs that close onto an object
#: The URDF joint each DOF drives (GraspGen-X's ``inspire_hand`` description).
URDF_JOINTS = ("pinky_proximal_joint", "ring_proximal_joint", "middle_proximal_joint",
               "index_proximal_joint", "thumb_proximal_pitch_joint",
               "thumb_proximal_yaw_joint")
CLOSE_MODES = ("sequential", "simultaneous")
STATUS_TEXT = {
    0: "opening", 1: "closing", 2: "at target", 3: "stopped on force",
    5: "current limit", 6: "locked rotor", 7: "actuator fault",
}
STATUS_STOPPED_ON_FORCE = 3   # the manual's; this hand reports 2 instead (module docstring)
#: A closing DOF that stops more than this short of its target (register units) was stopped
#: by something. The hand reaches a free target to within ~7 (2026-10-05: [997, 996, 999,
#: 1000, 243] for 1000 and 250); 30 is ~2.5 deg of finger, ~3 mm at the fingertip.
CONTACT_MARGIN = 30
#: Where an EMPTY simultaneous close stops each DOF (the user, 2026-10-05, 400 g, speed 500):
#: little and ring curl fully; the thumb lands on the middle and index fingers and stops
#: them -- so "the thumb and a finger stopped" is also true of an empty fist. Config
#: ``grasp.empty_close_angles`` overrides it.
EMPTY_CLOSE_ANGLES = [0, 0, 180, 361, 493, 0]
ERROR_BITS = ("locked rotor", "over temperature", "over current",
              "motor abnormal", "communication")
HOLD = -1  # ANGLE_SET value that leaves a DOF where it is (not valid for thumb rotation)
BAUDS = (115200, 57600, 19200)  # REDU_RATIO 0, 1, 2

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_ID = 1          # this lab's hand since 2026-10-05 (it answered at 2 on 09-15); factory default
DEFAULT_SPEED = 500     # 0-1000, 1000 = 0.8 s full stroke
DEFAULT_FORCE = 400     # grams per finger before it stops
DEFAULT_CLOSE_MODE = "simultaneous"   # the user, 2026-10-05: sequential pushes the object

# --- the fixed grasp ------------------------------------------------------------
# Thumb rotation (manual 2.4.11, top view): 1000 = beta 165 deg, thumb swung out
# nearly flat with the palm; 0 = beta 90 deg, thumb pointing out of the palm,
# facing the fingers (opposition). A round grasp needs the opposed thumb.
THUMB_OPPOSED = 0
OPEN_POSE = [1000, 1000, 1000, 1000, 1000, THUMB_OPPOSED]   # wide C-shape
CLOSE_POSE = [0, 0, 0, 0, 0, THUMB_OPPOSED]                 # stops on contact
#: Index straight, three fingers curled, thumb straight and swung out flat beside the
#: palm: for touching with the index tip (tpgpt.hardware.inspire_mount.POINT_POSES).
POINT_POSE = [0, 0, 0, 1000, 1000, 1000]
#: The same with the little finger straight instead: its tip is 68 mm across the palm from
#: the index tip, which is what tells the mount's yaw from its sideways offset.
POINT_POSE_LITTLE = [1000, 0, 0, 0, 1000, 1000]
POINT_POSES = {"index": POINT_POSE, "little": POINT_POSE_LITTLE}
#: goto() leaves a DOF alone when it is already this close to its target (register units).
GOTO_TOLERANCE = 20

#: Register -> URDF angle: the ruler fit of 2026-10-05 (config/inspire_hand.yaml
#: joint_map explains; about +-5 mm per reading, so still flagged unverified). The
#: config's block overrides it.
DEFAULT_JOINT_MAP = {
    "little": (1.470, 0.037), "ring": (1.470, 0.031), "middle": (1.470, 0.032),
    "index": (1.470, 0.029), "thumb_bend": (0.770, 0.120), "thumb_rot": (1.308, 0.0),
}


def joint_map_from_config(cfg: dict | None) -> dict:
    """``{dof: (q_at_0, q_at_1000, verified)}`` from the config's ``joint_map`` block.

    Missing DOFs take :data:`DEFAULT_JOINT_MAP`, flagged unverified.
    """
    block = (cfg or {}).get("joint_map", {}) or {}
    out = {}
    for name in DOF_NAMES:
        entry = block.get(name)
        if entry is None:
            q0, q1000 = DEFAULT_JOINT_MAP[name]
            out[name] = (float(q0), float(q1000), False)
        else:
            out[name] = (float(entry["q_at_0"]), float(entry["q_at_1000"]),
                         bool(entry.get("verified", False)))
    return out


def register_to_joint(values, joint_map: dict) -> list[float]:
    """Register values (0-1000, DOF order) -> URDF joint angles, rad (DOF order)."""
    return [joint_map[n][0] + (joint_map[n][1] - joint_map[n][0]) * float(u) / 1000.0
            for n, u in zip(DOF_NAMES, values)]


def joint_to_register(angles, joint_map: dict) -> list[int]:
    """URDF joint angles, rad (DOF order) -> register values, clipped to 0-1000."""
    out = []
    for n, q in zip(DOF_NAMES, angles):
        q0, q1000 = joint_map[n][0], joint_map[n][1]
        u = 1000.0 * (float(q) - q0) / (q1000 - q0)
        out.append(int(round(min(max(u, 0.0), 1000.0))))
    return out


@dataclass
class CloseReport:
    """What a close did, DOF by DOF (names from :data:`DOF_NAMES`).

    Each DOF is judged at the end of the step that last commanded it to move, since
    a later step that holds it may report its status differently.
    """

    mode: str
    contact: list = field(default_factory=list)    # stopped short of the close target
    on_air: list = field(default_factory=list)     # reached the close target untouched
    on_object: list = field(default_factory=list)  # fingers stopped beyond the empty close
    timed_out: list = field(default_factory=list)  # still moving at the step's timeout
    errors: str = ""
    angles: list = field(default_factory=list)     # ANGLE_ACT after the close
    forces_g: list = field(default_factory=list)   # FORCE_ACT after the close
    status: list = field(default_factory=list)     # STATUS per DOF, from its own step
    force_set_g: list = field(default_factory=list)
    speed: int = DEFAULT_SPEED

    @property
    def fingers_in_contact(self) -> list:
        return [n for n in FINGERS if n in self.contact]

    @property
    def opposed(self) -> bool:
        """The thumb and at least one finger stopped -- on the object or on each other:
        an empty fist is opposed too. :attr:`held` is the test for an object."""
        return "thumb_bend" in self.contact and bool(self.fingers_in_contact)

    @property
    def held(self) -> bool:
        """Something is in the hand: a finger stopped beyond the empty close
        (module docstring)."""
        return bool(self.on_object)

    @property
    def ok(self) -> bool:
        """No actuator error and nothing left moving."""
        return not self.errors and not self.timed_out

    def as_dict(self) -> dict:
        d = asdict(self)
        d["opposed"] = self.opposed
        d["held"] = self.held
        return d

    def to_json(self) -> str:
        return json.dumps(self.as_dict())

    def summary(self) -> str:
        contact = ", ".join(self.contact) or "nothing"
        text = (f"{self.mode} close: "
                + (f"HOLDING (on the object: {', '.join(self.on_object)})" if self.held
                   else "NOTHING HELD")
                + f"; stopped: {contact}")
        if self.on_air:
            text += f"; closed on air: {', '.join(self.on_air)}"
        if self.timed_out:
            text += f"; still moving at the timeout: {', '.join(self.timed_out)}"
        if self.errors:
            text += f"; actuator error ({self.errors})"
        return text


# --- protocol ------------------------------------------------------------------
def checksum(body: bytes) -> int:
    return sum(body) & 0xFF


def write_frame(hand_id: int, address: int, data: bytes) -> bytes:
    body = bytes([hand_id, len(data) + 3, 0x12, address & 0xFF, address >> 8]) + data
    return b"\xEB\x90" + body + bytes([checksum(body)])


def read_frame(hand_id: int, address: int, nbytes: int) -> bytes:
    body = bytes([hand_id, 0x04, 0x11, address & 0xFF, address >> 8, nbytes])
    return b"\xEB\x90" + body + bytes([checksum(body)])


def shorts(values) -> bytes:
    return b"".join(struct.pack("<h", int(v)) for v in values)


class InspireHand:
    def __init__(self, port: str | None = DEFAULT_PORT, hand_id: int = DEFAULT_ID,
                 baud: int = 115200, timeout: float = 0.2, dry_run: bool = False,
                 log=print):
        self.id = hand_id
        self.dry_run = dry_run
        #: Where an empty close stops each DOF (:data:`EMPTY_CLOSE_ANGLES`; the node sets
        #: the config's).
        self.empty_close = list(EMPTY_CLOSE_ANGLES)
        #: The last close held something and the hand has not moved since (grasp_open).
        self.holding = False
        self.log = log
        self.ser = None
        if not dry_run:
            import serial  # pyserial
            self.ser = serial.Serial(port, baud, bytesize=8, parity="N",
                                     stopbits=1, timeout=timeout)

    def close_port(self):
        if self.ser is not None:
            self.ser.close()

    # low level
    def _transact(self, frame: bytes, reply_len: int) -> bytes:
        if self.dry_run:
            self.log("  tx " + frame.hex(" "))
            return b""
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        reply = self.ser.read(reply_len)
        if len(reply) != reply_len or reply[:2] != b"\x90\xEB":
            raise IOError(
                f"no/short reply ({len(reply)}/{reply_len} bytes: {reply.hex(' ')}). "
                "Check port, 24 V power, A/B wiring, hand ID and baud rate.")
        if reply[-1] != checksum(reply[2:-1]):
            raise IOError(f"bad checksum in reply {reply.hex(' ')}")
        return reply

    def write(self, address: int, data: bytes):
        reply = self._transact(write_frame(self.id, address, data), 9)
        if reply and reply[7] != 1:
            raise IOError(f"hand refused write to register {address}")

    def read(self, address: int, nbytes: int) -> bytes:
        reply = self._transact(read_frame(self.id, address, nbytes), 8 + nbytes)
        return reply[7:7 + nbytes] if reply else bytes(nbytes)

    def read_shorts(self, address: int) -> list[int]:
        return list(struct.unpack("<6h", self.read(address, 12)))

    # mid level
    def set_speed(self, speed: int):
        self.write(SPEED_SET, shorts([speed] * 6))

    def set_force(self, grams):
        """One threshold (grams) for every DOF, or six, in DOF order."""
        values = list(grams) if isinstance(grams, (list, tuple)) else [grams] * 6
        if len(values) != 6 or not all(0 <= int(v) <= 1000 for v in values):
            raise ValueError(f"force must be one value or six, each 0-1000 g: {grams!r}")
        self.write(FORCE_SET, shorts(values))

    def set_angles(self, angles: list[int]):
        angles = list(angles)
        if angles[5] == HOLD and not self.dry_run:
            # ANGLE_SET(5), thumb rotation, only accepts 0-1000 (manual 2.4.11):
            # -1 does not hold it, the hand drives it to ~1000. Hold = resend
            # its current angle instead.
            angles[5] = self.read_shorts(ANGLE_ACT)[5]
        self.write(ANGLE_SET, shorts(angles))

    def clear_error(self):
        self.write(CLEAR_ERROR, bytes([1]))

    def status(self) -> dict:
        return {
            "angle": self.read_shorts(ANGLE_ACT),
            "force_g": self.read_shorts(FORCE_ACT),
            "status": list(self.read(STATUS, 6)),
            "error": list(self.read(ERROR, 6)),
            "temp_c": list(self.read(TEMP, 6)),
            "voltage_raw": struct.unpack("<h", self.read(VOLTAGE, 2))[0],
        }

    def wait(self, timeout: float, on_status=None) -> dict:
        """Poll until no DOF is still moving (STATUS 0/1) or ``timeout`` passes.

        Returns the last status, with ``"timed_out"``: the DOFs still moving when the
        time ran out (empty when everything stopped). ``on_status`` is called with
        every status read, so a caller can publish the hand while it moves.
        """
        if self.dry_run:
            return {}
        end = time.monotonic() + timeout
        time.sleep(0.05)
        while True:
            st = self.status()
            if on_status is not None:
                on_status(st)
            moving = [n for n, s in zip(DOF_NAMES, st["status"]) if s in (0, 1)]
            if not moving:
                st["timed_out"] = []
                return st
            if time.monotonic() > end:
                st["timed_out"] = moving
                return st

    def move(self, label: str, angles: list[int], timeout: float, verbose: bool = True,
             on_status=None) -> dict:
        if verbose:
            self.log(f"{label}: ANGLE_SET {dict(zip(DOF_NAMES, angles))}")
        self.set_angles(angles)
        st = self.wait(timeout, on_status)
        if verbose:
            print_status(st, self.log)
        return st

    # poses
    def goto(self, target, speed: int = DEFAULT_SPEED, force=DEFAULT_FORCE, timeout: float = 3.0,
             verbose: bool = True, on_status=None, tolerance: int = GOTO_TOLERANCE) -> dict:
        """Move to the register values ``target`` (DOF order) without the thumb and the
        fingers meeting (module docstring). Each step waits for the last:

        1. the thumb lifts: its bend opens to the target if that is straighter, and fully
           if the thumb is about to rotate;
        2. the fingers that open, open -- all curled ones, if the thumb is about to rotate;
        3. the thumb rotates (only straight, the fingers open);
        4. the fingers that close, close;
        5. the thumb bends to its target, last.

        Returns the last status. A step with nothing to move is skipped.
        """
        target = [int(v) for v in target]
        if len(target) != 6 or not all(0 <= v <= 1000 for v in target):
            raise ValueError(f"a pose is six values 0-1000: {target!r}")
        self.holding = False
        self.set_speed(speed)
        self.set_force(force)
        if self.dry_run:
            return self.move("goto", target, timeout, verbose, on_status)
        now = self.read_shorts(ANGLE_ACT)
        rotate = abs(target[5] - now[5]) > tolerance
        st = {}

        def step(label, values):
            nonlocal st
            st = self.move(label, values, timeout, verbose, on_status)
            for k, v in enumerate(values):
                if v != HOLD:
                    now[k] = v

        bend_up = 1000 if rotate else target[4]
        if bend_up > now[4] + tolerance:
            step("thumb lifts", [HOLD] * 4 + [bend_up, HOLD])
        opening = {k: (1000 if rotate and now[k] < 1000 - tolerance else target[k])
                   for k in range(4)}
        opening = {k: v for k, v in opening.items() if v > now[k] + tolerance}
        if opening:
            step("fingers open", [opening.get(k, HOLD) for k in range(4)] + [HOLD, HOLD])
        if rotate:
            step("thumb rotates", [HOLD] * 5 + [target[5]])
        closing = {k: target[k] for k in range(4) if target[k] < now[k] - tolerance}
        if closing:
            step("fingers close", [closing.get(k, HOLD) for k in range(4)] + [HOLD, HOLD])
        if target[4] < now[4] - tolerance:
            step("thumb bends", [HOLD] * 4 + [target[4], HOLD])
        return self.move("pose", target, timeout, verbose, on_status)

    # the grasp
    def grasp_open(self, speed: int = DEFAULT_SPEED, force=DEFAULT_FORCE,
                   timeout: float = 3.0, verbose: bool = True, on_status=None) -> dict:
        """Open. Right after a close that held something (:attr:`holding`), with the
        thumb still opposed: everything opens at once, so the release does not push the
        object. Otherwise (a fist, any other pose): :meth:`goto` ``OPEN_POSE``, the thumb
        first."""
        if self.holding and not self.dry_run:
            self.holding = False
            if abs(self.read_shorts(ANGLE_ACT)[5] - OPEN_POSE[5]) <= GOTO_TOLERANCE:
                self.set_speed(speed)
                self.set_force(force)
                st = self.move("open: release, all at once", OPEN_POSE, timeout, verbose,
                               on_status)
                short = [n for n, a, t in zip(DOF_NAMES, st["angle"], OPEN_POSE)
                         if abs(int(a) - t) > CONTACT_MARGIN]
                if not short:
                    return st
                # The thumb blocked the fingers: 'held' was an empty fist after all (it can
                # read so: the user's 150 g empty close, middle at 349). Open in order.
                self.log(f"open: {', '.join(short)} not open after the release; "
                         "opening in order (the thumb first)")
        return self.goto(OPEN_POSE, speed, force, timeout, verbose, on_status)

    def close(self, mode: str = DEFAULT_CLOSE_MODE, speed: int = DEFAULT_SPEED,
              force=DEFAULT_FORCE, timeout: float = 3.0, verbose: bool = True,
              on_status=None) -> CloseReport:
        """Close onto an object, ``sequential`` or ``simultaneous`` (module docstring).

        Every bending DOF stops on its force threshold. Returns the :class:`CloseReport`;
        raises IOError on a serial failure only.
        """
        if mode not in CLOSE_MODES:
            raise ValueError(f"close mode must be one of {CLOSE_MODES}, not {mode!r}")
        self.set_speed(speed)
        self.set_force(force)
        forces = list(force) if isinstance(force, (list, tuple)) else [int(force)] * 6
        report = CloseReport(mode=mode, force_set_g=[int(f) for f in forces], speed=int(speed))
        if self.dry_run:
            self.move("thumb rotation opposed", [HOLD] * 5 + [THUMB_OPPOSED], timeout, verbose)
            if mode == "sequential":
                self.move("close: fingers", CLOSE_POSE[:4] + [HOLD, THUMB_OPPOSED], timeout,
                          verbose)
                self.move("close: thumb", [HOLD] * 4 + CLOSE_POSE[4:], timeout, verbose)
            else:
                self.move("close: fingers and thumb", CLOSE_POSE, timeout, verbose)
            return report
        if mode == "sequential":
            steps = [("thumb rotation opposed", [HOLD] * 5 + [THUMB_OPPOSED], ("thumb_rot",)),
                     ("close: fingers", CLOSE_POSE[:4] + [HOLD, THUMB_OPPOSED], FINGERS),
                     ("close: thumb", [HOLD] * 4 + CLOSE_POSE[4:], ("thumb_bend",))]
        else:
            steps = [("thumb rotation opposed", [HOLD] * 5 + [THUMB_OPPOSED], ("thumb_rot",)),
                     ("close: fingers and thumb", CLOSE_POSE, BENDING)]
        status, stopped_at = [None] * 6, [None] * 6
        st = {}
        for label, angles, moved in steps:
            st = self.move(label, angles, timeout, verbose, on_status)
            for name in moved:
                k = DOF_NAMES.index(name)
                status[k], stopped_at[k] = st["status"][k], st["angle"][k]
                if name in st.get("timed_out", []):
                    report.timed_out.append(name)
        report.status = [s if s is not None else st["status"][k] for k, s in enumerate(status)]
        report.angles = list(st["angle"])
        report.forces_g = list(st["force_g"])
        report.errors = error_text(st)
        for k, name in enumerate(BENDING):
            if name in report.timed_out:
                continue
            if stopped_short(stopped_at[k], CLOSE_POSE[k], report.status[k]):
                report.contact.append(name)
            else:
                report.on_air.append(name)
            if name in FINGERS and int(stopped_at[k]) - int(self.empty_close[k]) > CONTACT_MARGIN:
                report.on_object.append(name)
        self.holding = report.held and not report.errors
        return report

    def point(self, speed: int = DEFAULT_SPEED, force=DEFAULT_FORCE, timeout: float = 3.0,
              verbose: bool = True, on_status=None, finger: str = "index") -> dict:
        """A pointing pose for measuring the mount by touch (:data:`POINT_POSES`: the
        ``index`` or the ``little`` finger straight): the thumb swings out flat beside the
        palm, then the other three fingers curl (:meth:`goto`)."""
        if finger not in POINT_POSES:
            raise ValueError(f"point with one of {sorted(POINT_POSES)}, not {finger!r}")
        return self.goto(POINT_POSES[finger], speed, force, timeout, verbose, on_status)

    def grasp_close(self, speed: int = DEFAULT_SPEED, force=DEFAULT_FORCE,
                    timeout: float = 3.0, verbose: bool = True) -> dict:
        """The sequential close, returning the final status (the 2026-09-15 interface)."""
        self.close("sequential", speed, force, timeout, verbose)
        return {} if self.dry_run else self.status()


def default_config_path() -> str:
    """Installed config/inspire_hand.yaml, else the source tree's."""
    import os
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory("ycb_pointcloud_pipeline"), "config",
                            "inspire_hand.yaml")
        if os.path.isfile(path):
            return path
    except Exception:  # noqa: BLE001 - package not installed
        pass
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                        "inspire_hand.yaml")


def load_config(path: str | None = None) -> dict:
    """The ``inspire_hand:`` block of config/inspire_hand.yaml (or of ``path``)."""
    import os

    import yaml
    path = os.path.expanduser(path) if path else default_config_path()
    with open(path) as f:
        data = yaml.safe_load(f)
    cfg = data.get("inspire_hand", data)
    cfg["_path"] = path
    return cfg


def stopped_short(angle: int, target: int, status: int | None = None) -> bool:
    """A closing DOF was stopped by something: it ended more than :data:`CONTACT_MARGIN`
    short of its target, or reported the manual's STATUS 3 (module docstring)."""
    return status == STATUS_STOPPED_ON_FORCE or int(angle) - int(target) > CONTACT_MARGIN


def contact_dofs(st: dict, target=CLOSE_POSE) -> list[str]:
    """The bending DOFs a status after a close shows stopped by something
    (:func:`stopped_short` against ``target``, the close's)."""
    if not st:
        return []
    return [n for k, n in enumerate(BENDING)
            if stopped_short(st["angle"][k], target[k], st["status"][k])]



def closure(st: dict) -> list[float]:
    """Per DOF from ANGLE_ACT, ``(1000 - a) / 1000`` clipped to 0..1.

    For the five bending DOFs that is 0 = straight, 1 = fully bent. For thumb
    rotation (index 5) it is **opposition**, 1 = opposed (beta 90 deg), 0 = swung
    out (beta 165 deg) -- not a closure: an open hand reads 1 there too.
    """
    if not st:
        return [0.0] * 6
    return [min(max((1000.0 - a) / 1000.0, 0.0), 1.0) for a in st["angle"]]


def error_text(st: dict) -> str:
    parts = []
    for name, err in zip(DOF_NAMES, (st or {}).get("error", [0] * 6)):
        errs = [b for k, b in enumerate(ERROR_BITS) if err >> k & 1]
        if errs:
            parts.append(f"{name}: {', '.join(errs)}")
    return "; ".join(parts)


def print_status(st: dict, log=print):
    if not st:
        return
    log(f"  VOLTAGE register raw = {st['voltage_raw']}")
    log(f"  {'dof':<11}{'angle':>6}{'force g':>9}  {'temp':>4}  status")
    for i, name in enumerate(DOF_NAMES):
        s = st["status"][i]
        err = st["error"][i]
        errs = ", ".join(b for k, b in enumerate(ERROR_BITS) if err >> k & 1)
        log(f"  {name:<11}{st['angle'][i]:>6}{st['force_g'][i]:>9}  "
            f"{st['temp_c'][i]:>4}  {STATUS_TEXT.get(s, s)}"
            + (f"  ERROR: {errs}" if errs else ""))


def scan(port: str, ids) -> int:
    """Try every baud rate and hand ID; stop at the first one that answers.

    Only reads registers, so the hand never moves.
    """
    hand = InspireHand(port, ids.start, BAUDS[0], timeout=0.05)
    try:
        for baud in BAUDS:
            hand.ser.baudrate = baud
            print(f"baud {baud}: trying IDs {ids.start}-{ids.stop - 1} ...", flush=True)
            for hand_id in ids:
                hand.id = hand_id
                try:
                    reply_id = hand.read(HAND_ID_REG, 1)[0]
                    ratio = hand.read(REDU_RATIO, 1)[0]
                    raw_v = struct.unpack("<h", hand.read(VOLTAGE, 2))[0]
                except IOError:
                    continue
                print(f"\nFOUND hand: --id {reply_id} --baud {baud}  "
                      f"(REDU_RATIO={ratio}, VOLTAGE register raw={raw_v})")
                print(f"  use: inspire_hand --port {port} --id {reply_id} --baud {baud} status")
                return 0
    finally:
        hand.close_port()
    print(f"\nNo hand answered on {port} at any baud rate "
          f"{BAUDS} or ID {ids.start}-{ids.stop - 1}. Check the connection:\n"
          "  - hand powered: 24 V DC on GX12 pin 2 (VCC) vs pin 1/5 (GND), >= 2.5 A supply\n"
          "  - RS485 wiring: pin 3 A+ -> adapter A/D+, pin 4 B- -> adapter B/D- "
          "(try swapping A and B)\n"
          "  - common ground: adapter GND to GX12 pin 1 or 5\n"
          "  - correct port (run `ports`) and nothing else has it open")
    return 1


def list_ports():
    from serial.tools import list_ports as lp
    ports = list(lp.comports())
    if not ports:
        print("no serial ports found (is the USB-RS485 adapter plugged in?)")
    for p in ports:
        print(f"{p.device:<16} {p.description}  [{p.hwid}]")


def _config_defaults(argv) -> dict:
    """Port, ID, baud, speed, force and close mode from the config file, if readable."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    try:
        cfg = load_config(known.config)
    except Exception:  # noqa: BLE001 - no yaml installed / no config: built-in defaults
        return {}
    serial, grasp = cfg.get("serial", {}), cfg.get("grasp", {})
    return {k: v for k, v in {
        "port": serial.get("port"), "id": serial.get("hand_id"), "baud": serial.get("baud"),
        "speed": grasp.get("speed"), "force": grasp.get("force_g"),
        "close_mode": grasp.get("close_mode"),
        "timeout": grasp.get("step_timeout_s")}.items() if v is not None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["ports", "scan", "status", "open", "close", "cycle",
                                        "thumb", "home", "point", "point-little", "set", "clear-error",
                                        "round-open", "round-close", "round-cycle"],
                    help="round-* are aliases of open/close/cycle; set takes six values")
    ap.add_argument("values", nargs="*", type=int,
                    help="set: little ring middle index thumb_bend thumb_rot, 0-1000 "
                         "(fingers before the thumb when closing, after it when opening)")
    ap.add_argument("--config", default=None,
                    help="config/inspire_hand.yaml to take defaults from (default: installed)")
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--id", type=int, default=DEFAULT_ID,
                    help=f"HAND_ID (factory 1; this lab's hand answers on {DEFAULT_ID}, see `scan`)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--max-id", type=int, default=254, help="scan: highest ID tried")
    ap.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="0-1000, 1000 = 0.8 s stroke")
    ap.add_argument("--force", type=int, default=DEFAULT_FORCE,
                    help="per-finger stop threshold in grams, 0-1000")
    ap.add_argument("--close-mode", choices=CLOSE_MODES, default=DEFAULT_CLOSE_MODE)
    ap.add_argument("--thumb-rot", type=int, default=THUMB_OPPOSED,
                    help="thumb: rotation value to swing to and back from 1000")
    ap.add_argument("--hold", type=float, default=2.0, help="cycle: seconds closed")
    ap.add_argument("--timeout", type=float, default=3.0, help="motion wait per step, s")
    ap.add_argument("--dry-run", action="store_true", help="print frames only")
    ap.set_defaults(**_config_defaults(argv))
    args = ap.parse_args(argv)
    command = args.command.replace("round-", "")

    if command == "ports":
        list_ports()
        return 0
    if command == "scan":
        try:
            return scan(args.port, range(1, args.max_id + 1))
        except Exception as exc:  # serial.SerialException, PermissionError
            print(f"cannot open {args.port}: {exc}", file=sys.stderr)
            return 1

    try:
        hand = InspireHand(args.port, args.id, args.baud, dry_run=args.dry_run)
    except Exception as exc:  # serial.SerialException, PermissionError
        print(f"cannot open {args.port}: {exc}", file=sys.stderr)
        if "ermission" in str(exc):
            print("  -> sudo usermod -aG dialout $USER, then log out and in",
                  file=sys.stderr)
        return 1

    try:
        if command == "status":
            print_status(hand.status())
            return 0
        if command == "clear-error":
            hand.clear_error()
            print("errors cleared")
            return 0
        if command == "home":
            hand.goto([1000] * 6, args.speed, args.force, args.timeout)   # straight, thumb out
        elif command == "thumb":
            # Only the thumb rotation moves, fingers stay put: shows which way it turns.
            hand.set_speed(args.speed)
            hand.set_force(args.force)
            hand.move("thumb rotation -> value", [HOLD] * 5 + [args.thumb_rot], args.timeout)
            time.sleep(1.0)
            hand.move("thumb rotation -> 1000", [HOLD] * 5 + [1000], args.timeout)
        elif command == "open":
            hand.grasp_open(args.speed, args.force, args.timeout)
        elif command == "point":
            hand.point(args.speed, args.force, args.timeout)
        elif command == "point-little":
            hand.point(args.speed, args.force, args.timeout, finger="little")
        elif command == "set":
            if len(args.values) != 6 or not all(0 <= v <= 1000 for v in args.values):
                print("set needs six values 0-1000: little ring middle index thumb_bend "
                      "thumb_rot", file=sys.stderr)
                return 2
            st = hand.goto(args.values, args.speed, args.force, args.timeout)
            if st:
                print(f"holding {args.values}; ANGLE_ACT {st['angle']}")
        elif command in ("close", "cycle"):
            if command == "cycle":
                hand.grasp_open(args.speed, args.force, args.timeout)
                time.sleep(args.hold)
            report = hand.close(args.close_mode, args.speed, args.force, args.timeout)
            if not args.dry_run:
                print(report.summary())
                print("opposed (thumb + a finger on the object): "
                      + ("yes" if report.opposed else "NO"))
            if command == "cycle":
                time.sleep(args.hold)
                hand.move("release", OPEN_POSE, args.timeout)
        return 0
    except IOError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        hand.close_port()


if __name__ == "__main__":
    sys.exit(main())
