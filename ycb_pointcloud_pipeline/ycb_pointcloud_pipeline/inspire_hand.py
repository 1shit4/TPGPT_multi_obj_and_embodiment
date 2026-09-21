#!/usr/bin/env python3
"""Inspire RH56 hand over RS485: protocol, the fixed open/close grasp, and a CLI.

No ROS in here, so the same code runs from a terminal and inside
``inspire_hand_node`` (the ROS 2 wrapper the pipeline uses).

The grasp (verified on the real hand, 2026-09-15, gripping round objects from
the side with the object close to the palm) is ONE fixed open and ONE fixed
close, in ANGLE_SET register values (0-1000, 1000 = straight/open, 0 = bent)::

    dof            open    close
    little..index  1000    0      the force threshold stops each finger on contact
    thumb bend     1000    0      bends last, locking the object against the fingers
    thumb rotation 0       0      opposed: pointing out of the palm, facing the fingers

Sequences (the order is what keeps the thumb out of the index finger):

* **open**: fingers and thumb bend straighten first (thumb rotation held), then
  the straight thumb swings into opposition over the open palm.
* **close**: thumb rotation confirmed opposed, the four fingers wrap and push the
  object against the straight thumb, then the thumb bends to lock it.

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
    ros2 run ycb_pointcloud_pipeline inspire_hand cycle --hold 3
    ros2 run ycb_pointcloud_pipeline inspire_hand --dry-run close   # print frames only

Needs only ``pyserial``.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

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
STATUS_TEXT = {
    0: "opening", 1: "closing", 2: "at target", 3: "stopped on force",
    5: "current limit", 6: "locked rotor", 7: "actuator fault",
}
STATUS_STOPPED_ON_FORCE = 3
ERROR_BITS = ("locked rotor", "over temperature", "over current",
              "motor abnormal", "communication")
HOLD = -1  # ANGLE_SET value that leaves a DOF where it is (not valid for thumb rotation)
BAUDS = (115200, 57600, 19200)  # REDU_RATIO 0, 1, 2

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_ID = 2          # this lab's hand (factory default is 1)
DEFAULT_SPEED = 500     # 0-1000, 1000 = 0.8 s full stroke
DEFAULT_FORCE = 400     # grams per finger before it stops

# --- the fixed grasp ------------------------------------------------------------
# Thumb rotation (manual 2.4.11, top view): 1000 = beta 165 deg, thumb swung out
# nearly flat with the palm; 0 = beta 90 deg, thumb pointing out of the palm,
# facing the fingers (opposition). A round grasp needs the opposed thumb.
THUMB_OPPOSED = 0
OPEN_POSE = [1000, 1000, 1000, 1000, 1000, THUMB_OPPOSED]   # wide C-shape
CLOSE_POSE = [0, 0, 0, 0, 0, THUMB_OPPOSED]                 # stops on contact


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

    def set_force(self, grams: int):
        self.write(FORCE_SET, shorts([grams] * 6))

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

    def wait(self, timeout: float) -> dict:
        """Poll until no DOF is still moving (STATUS 0/1) or timeout."""
        if self.dry_run:
            return {}
        end = time.monotonic() + timeout
        time.sleep(0.05)
        while True:
            st = self.status()
            if not any(s in (0, 1) for s in st["status"]) or time.monotonic() > end:
                return st

    def move(self, label: str, angles: list[int], timeout: float, verbose: bool = True) -> dict:
        if verbose:
            self.log(f"{label}: ANGLE_SET {dict(zip(DOF_NAMES, angles))}")
        self.set_angles(angles)
        st = self.wait(timeout)
        if verbose:
            print_status(st, self.log)
        return st

    # the fixed grasp
    def grasp_open(self, speed: int = DEFAULT_SPEED, force: int = DEFAULT_FORCE,
                   timeout: float = 3.0, verbose: bool = True) -> dict:
        """Fingers and thumb bend straight (thumb rotation held), then thumb opposed."""
        self.set_speed(speed)
        self.set_force(force)
        self.move("open fingers + thumb bend", [1000] * 5 + [HOLD], timeout, verbose)
        return self.move("open (thumb opposed)", OPEN_POSE, timeout, verbose)

    def grasp_close(self, speed: int = DEFAULT_SPEED, force: int = DEFAULT_FORCE,
                    timeout: float = 3.0, verbose: bool = True) -> dict:
        """Thumb opposed, four fingers wrap, then the thumb bends. Stops on contact."""
        self.set_speed(speed)
        self.set_force(force)
        self.move("thumb rotation opposed", [HOLD] * 5 + [THUMB_OPPOSED], timeout, verbose)
        self.move("close: fingers", CLOSE_POSE[:4] + [HOLD, THUMB_OPPOSED], timeout, verbose)
        return self.move("close: thumb", [HOLD] * 4 + CLOSE_POSE[4:], timeout, verbose)


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


def contact_dofs(st: dict) -> list[str]:
    """DOFs that stopped on their force threshold, i.e. touched something."""
    if not st:
        return []
    return [n for n, s in zip(DOF_NAMES, st["status"]) if s == STATUS_STOPPED_ON_FORCE]


def closure(st: dict) -> list[float]:
    """0 = straight, 1 = fully bent, per DOF, from ANGLE_ACT."""
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["ports", "scan", "status", "open", "close", "cycle",
                                        "thumb", "home", "clear-error",
                                        "round-open", "round-close", "round-cycle"],
                    help="round-* are aliases of open/close/cycle")
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--id", type=int, default=DEFAULT_ID,
                    help=f"HAND_ID (factory 1; this lab's hand answers on {DEFAULT_ID}, see `scan`)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--max-id", type=int, default=254, help="scan: highest ID tried")
    ap.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="0-1000, 1000 = 0.8 s stroke")
    ap.add_argument("--force", type=int, default=DEFAULT_FORCE,
                    help="per-finger stop threshold in grams, 0-1000")
    ap.add_argument("--thumb-rot", type=int, default=THUMB_OPPOSED,
                    help="thumb: rotation value to swing to and back from 1000")
    ap.add_argument("--hold", type=float, default=2.0, help="cycle: seconds closed")
    ap.add_argument("--timeout", type=float, default=3.0, help="motion wait per step, s")
    ap.add_argument("--dry-run", action="store_true", help="print frames only")
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
            hand.set_speed(args.speed)
            hand.set_force(args.force)
            hand.move("home (everything straight, thumb out)", [1000] * 6, args.timeout)
        elif command == "thumb":
            # Only the thumb rotation moves, fingers stay put: shows which way it turns.
            hand.set_speed(args.speed)
            hand.set_force(args.force)
            hand.move("thumb rotation -> value", [HOLD] * 5 + [args.thumb_rot], args.timeout)
            time.sleep(1.0)
            hand.move("thumb rotation -> 1000", [HOLD] * 5 + [1000], args.timeout)
        elif command == "open":
            hand.grasp_open(args.speed, args.force, args.timeout)
        elif command in ("close", "cycle"):
            if command == "cycle":
                hand.grasp_open(args.speed, args.force, args.timeout)
                time.sleep(args.hold)
            st = hand.grasp_close(args.speed, args.force, args.timeout)
            if st:
                held = contact_dofs(st)
                print("contact on: " + (", ".join(held) if held else
                                        "nothing (no object, every finger closed fully)"))
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
