"""Inspire hand driver against a fake hand on a pseudo-terminal, and its config.

No hardware and no ROS: the fake answers the RH56 register protocol (manual 2.2)
and records every ANGLE_SET, so the open/close sequences can be checked frame by
frame.
"""

import os
import pty
import struct
import threading
import tty

import pytest

from ycb_pointcloud_pipeline import inspire_hand as ih


class FakeHand:
    """RH56 register protocol on a pty. Thumb rotation (DOF 5) is driven to 1000 by a
    -1 set value, as the real hand did."""

    def __init__(self, hand_id=2):
        self.id = hand_id
        self.master, slave = pty.openpty()
        tty.setraw(slave)
        self.port = os.ttyname(slave)
        self._slave = slave
        self.angle = [1000, 1000, 1000, 1000, 1000, 55]
        self.angle_sets = []
        self._stop = False
        threading.Thread(target=self._serve, daemon=True).start()

    def close(self):
        self._stop = True
        os.close(self.master)
        os.close(self._slave)

    def _reply(self, address, data):
        body = bytes([self.id, len(data) + 3, 0x11, address & 0xFF, address >> 8]) + data
        os.write(self.master, b"\x90\xEB" + body + bytes([ih.checksum(body)]))

    def _ack(self, address):
        body = bytes([self.id, 0x04, 0x12, address & 0xFF, address >> 8, 1])
        os.write(self.master, b"\x90\xEB" + body + bytes([ih.checksum(body)]))

    def _serve(self):
        buf = b""
        while not self._stop:
            try:
                buf += os.read(self.master, 256)
            except OSError:
                return
            while len(buf) >= 4 and buf[:2] == b"\xEB\x90":
                length = buf[3]
                total = 5 + length
                if len(buf) < total:
                    break
                frame, buf = buf[:total], buf[total:]
                hid, cmd, address = frame[2], frame[4], frame[5] | frame[6] << 8
                if hid != self.id:
                    continue
                if cmd == 0x12:
                    data = frame[7:-1]
                    if address == ih.ANGLE_SET:
                        values = list(struct.unpack("<6h", data))
                        self.angle_sets.append(values)
                        for k, v in enumerate(values):
                            if v == -1:
                                if k == 5:
                                    self.angle[5] = 1000   # the real hand's behaviour
                            else:
                                self.angle[k] = v
                    self._ack(address)
                else:
                    n = frame[7]
                    if address == ih.ANGLE_ACT:
                        data = struct.pack("<6h", *self.angle)
                    elif address == ih.STATUS:
                        data = bytes([2] * 6)
                    else:
                        data = bytes(n)
                    self._reply(address, data[:n])


@pytest.fixture
def fake():
    hand = FakeHand()
    yield hand
    hand.close()


def test_frames_match_the_manual_example():
    # Manual 2.2.1: read ANGLE_ACT of hand 1 -> EB 90 01 04 11 0A 06 0C 32
    assert ih.read_frame(1, ih.ANGLE_ACT, 12).hex(" ") == "eb 90 01 04 11 0a 06 0c 32"


def test_close_wraps_fingers_first_then_bends_the_thumb(fake):
    hand = ih.InspireHand(fake.port, 2, timeout=0.5, log=lambda *_: None)
    try:
        hand.grasp_close(timeout=1.0, verbose=False)
    finally:
        hand.close_port()
    assert fake.angle_sets == [
        [-1, -1, -1, -1, -1, ih.THUMB_OPPOSED],
        [0, 0, 0, 0, -1, ih.THUMB_OPPOSED],
        [-1, -1, -1, -1, 0, ih.THUMB_OPPOSED],
    ]


def test_open_straightens_then_opposes_and_never_sends_minus_one_to_the_thumb(fake):
    hand = ih.InspireHand(fake.port, 2, timeout=0.5, log=lambda *_: None)
    try:
        st = hand.grasp_open(timeout=1.0, verbose=False)
    finally:
        hand.close_port()
    first, second = fake.angle_sets
    assert first[:5] == [1000] * 5 and first[5] == 55       # held at its measured angle
    assert second == ih.OPEN_POSE
    assert all(s[5] != -1 for s in fake.angle_sets)
    assert fake.angle[5] == ih.THUMB_OPPOSED
    assert ih.contact_dofs(st) == []


def test_a_wrong_id_is_a_connection_error(fake):
    hand = ih.InspireHand(fake.port, 7, timeout=0.05, log=lambda *_: None)
    try:
        with pytest.raises(IOError, match="no/short reply"):
            hand.status()
    finally:
        hand.close_port()


def test_contact_and_closure_helpers():
    st = {"status": [3, 3, 2, 2, 3, 2], "angle": [400, 1000, 0, 500, 1200, -5],
          "error": [0, 0, 1, 0, 0, 0]}
    assert ih.contact_dofs(st) == ["little", "ring", "thumb_bend"]
    assert ih.closure(st) == [0.6, 0.0, 1.0, 0.5, 0.0, 1.0]
    assert ih.error_text(st) == "middle: locked rotor"


def test_config_builds_the_mount():
    cfg = ih.load_config(os.path.join(os.path.dirname(__file__), "..", "config",
                                      "inspire_hand.yaml"))
    assert cfg["handedness"] == "right"
    assert cfg["serial"]["hand_id"] == 2
    ee_mod = pytest.importorskip("tpgpt.hardware.end_effector")
    ee = ee_mod.inspire_from_config(cfg)
    assert ee.controller_frame == "fr3_hand_tcp"
    assert not ee.is_parallel_jaw
