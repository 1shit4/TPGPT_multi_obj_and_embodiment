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
    -1 set value, as the real hand did.

    ``contact_at[k]``: a DOF commanded below this angle stops there, STATUS 3 (on
    force), FORCE_ACT = its threshold. ``stuck[k]``: a commanded DOF reports STATUS 1
    (moving) for ever. A DOF held with -1 has its STATUS reset to 2 by the next
    ANGLE_SET -- the worst case for judging a DOF by the final status, which the
    driver must not do.
    """

    def __init__(self, hand_id=2, contact_at=None, stuck=None):
        self.id = hand_id
        self.master, slave = pty.openpty()
        tty.setraw(slave)
        self.port = os.ttyname(slave)
        self._slave = slave
        self.angle = [1000, 1000, 1000, 1000, 1000, 55]
        self.status = [2] * 6
        self.force = [0] * 6
        self.force_set = [400] * 6
        self.contact_at = list(contact_at or [None] * 6)
        self.stuck = list(stuck or [False] * 6)
        self.angle_sets = []
        self.force_sets = []
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

    def _set_angles(self, values):
        self.angle_sets.append(values)
        for k, v in enumerate(values):
            if v == -1:
                if k == 5:
                    self.angle[5] = 1000   # the real hand's behaviour
                else:
                    self.status[k] = 2      # held: status reset (worst case)
                continue
            if self.stuck[k]:
                self.status[k] = 1
                continue
            stop = self.contact_at[k]
            if stop is not None and v < stop:
                self.angle[k], self.status[k], self.force[k] = stop, 3, self.force_set[k]
            else:
                self.angle[k], self.status[k], self.force[k] = v, 2, 0

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
                        self._set_angles(list(struct.unpack("<6h", data)))
                    elif address == ih.FORCE_SET:
                        self.force_set = list(struct.unpack("<6h", data))
                        self.force_sets.append(self.force_set)
                    self._ack(address)
                else:
                    n = frame[7]
                    if address == ih.ANGLE_ACT:
                        data = struct.pack("<6h", *self.angle)
                    elif address == ih.FORCE_ACT:
                        data = struct.pack("<6h", *self.force)
                    elif address == ih.STATUS:
                        data = bytes(self.status)
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


def test_grasp_close_keeps_its_interface(fake):
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
    try:
        ee = ee_mod.inspire_from_config(cfg)
    except FileNotFoundError as exc:                       # no GraspGen-X checkout
        pytest.skip(str(exc))
    assert ee.controller_frame == "fr3_hand_tcp"
    assert ee.kind == "multi_finger" and ee.graspgen == "inspire_hand"
    assert not ee.model_spec["mount_measured"]             # until M1-M3


# --------------------------------------------------------------------------- close
# DOF order: little, ring, middle, index, thumb_bend, thumb_rot
LITTLE, RING, MIDDLE, INDEX, THUMB = range(5)


def _hand(fake, timeout=0.5):
    return ih.InspireHand(fake.port, 2, timeout=timeout, log=lambda *_: None)


def _close(fake, mode, timeout=0.6, force=400):
    hand = _hand(fake)
    try:
        return hand.close(mode, force=force, timeout=timeout, verbose=False)
    finally:
        hand.close_port()


def test_the_simultaneous_close_opposes_the_thumb_then_closes_everything_at_once(fake):
    _close(fake, "simultaneous")
    assert fake.angle_sets == [[-1, -1, -1, -1, -1, ih.THUMB_OPPOSED], [0, 0, 0, 0, 0, 0]]


def test_the_sequential_close_keeps_the_verified_order(fake):
    _close(fake, "sequential")
    assert fake.angle_sets == [
        [-1, -1, -1, -1, -1, ih.THUMB_OPPOSED],
        [0, 0, 0, 0, -1, ih.THUMB_OPPOSED],
        [-1, -1, -1, -1, 0, ih.THUMB_OPPOSED],
    ]


@pytest.mark.parametrize("mode", ih.CLOSE_MODES)
def test_the_report_names_contact_and_air_dof_by_dof(mode):
    # An object between the thumb and the index and middle fingers; little and ring miss it.
    contact = [None, None, 420, 380, 300, None]
    fake = FakeHand(contact_at=contact)
    try:
        report = _close(fake, mode, force=350)
    finally:
        fake.close()
    assert report.contact == ["middle", "index", "thumb_bend"]
    assert report.on_air == ["little", "ring"]
    assert report.timed_out == [] and report.errors == ""
    assert report.opposed and report.ok
    assert report.force_set_g == [350] * 6 and fake.force_sets[-1] == [350] * 6
    assert report.angles[MIDDLE] == 420 and report.forces_g[INDEX] == 350
    assert ih.json.loads(report.to_json())["opposed"] is True


def test_the_sequential_report_survives_the_hand_resetting_a_held_finger():
    # The fingers stop on the object in step 2; step 3 holds them (-1) and the fake
    # then reads them "at target". Judged at their own step, they stay in contact.
    fake = FakeHand(contact_at=[500, 500, 500, 500, 300, None])
    try:
        report = _close(fake, "sequential")
    finally:
        fake.close()
    assert fake.status[:4] == [2, 2, 2, 2]
    assert report.contact == ["little", "ring", "middle", "index", "thumb_bend"]


def test_fingers_without_the_thumb_are_not_opposed():
    fake = FakeHand(contact_at=[None, None, 400, 400, None, None])
    try:
        report = _close(fake, "simultaneous")
    finally:
        fake.close()
    assert report.fingers_in_contact == ["middle", "index"]
    assert "thumb_bend" in report.on_air
    assert not report.opposed and report.ok


def test_closing_on_air_is_reported_as_such(fake):
    report = _close(fake, "simultaneous")
    assert report.contact == [] and report.on_air == list(ih.BENDING)
    assert not report.opposed and report.ok
    assert "closed on air" in report.summary()


def test_a_dof_still_moving_at_the_timeout_is_an_error():
    fake = FakeHand(stuck=[False, False, False, True, False, False])
    try:
        report = _close(fake, "simultaneous", timeout=0.3)
    finally:
        fake.close()
    assert report.timed_out == ["index"]
    assert not report.ok
    assert "index" not in report.contact + report.on_air


def test_wait_names_what_is_still_moving():
    fake = FakeHand(stuck=[True, False, False, False, False, False])
    hand = _hand(fake)
    seen = []
    try:
        hand.set_angles([0, 0, 0, 0, 0, 0])
        st = hand.wait(0.2, on_status=seen.append)
    finally:
        hand.close_port()
        fake.close()
    assert st["timed_out"] == ["little"]
    assert len(seen) >= 1 and seen[-1] is st


def test_force_is_one_value_or_six(fake):
    hand = _hand(fake)
    try:
        hand.set_force([100, 200, 300, 400, 500, 600])
        assert fake.force_sets[-1] == [100, 200, 300, 400, 500, 600]
        with pytest.raises(ValueError):
            hand.set_force([100, 200])
        with pytest.raises(ValueError):
            hand.set_force(1200)
    finally:
        hand.close_port()


def test_an_unknown_close_mode_is_refused(fake):
    hand = _hand(fake)
    try:
        with pytest.raises(ValueError, match="close mode"):
            hand.close("squeeze")
    finally:
        hand.close_port()


# --------------------------------------------------------------------- joint map
def _config():
    return ih.load_config(os.path.join(os.path.dirname(__file__), "..", "config",
                                       "inspire_hand.yaml"))


def test_the_joint_map_is_read_from_the_config_and_flagged_until_measured():
    jm = ih.joint_map_from_config(_config())
    assert set(jm) == set(ih.DOF_NAMES)
    assert jm["thumb_rot"][:2] == (1.308, 0.0)
    assert not any(ok for _, _, ok in jm.values())          # until measured (M4)
    # Anything missing from the config falls back to the manual-derived default, unverified.
    assert ih.joint_map_from_config({})["index"] == (1.698, 0.029, False)


def test_registers_and_joint_angles_round_trip():
    jm = ih.joint_map_from_config(_config())
    q = ih.register_to_joint([1000, 1000, 1000, 1000, 1000, 0], jm)
    assert q == pytest.approx([0.037, 0.031, 0.032, 0.029, 0.028, 1.308])  # open, opposed
    q = ih.register_to_joint([0, 0, 0, 0, 0, 0], jm)
    assert q == pytest.approx([1.770, 1.699, 1.676, 1.698, 0.690, 1.308])
    u = [0, 250, 500, 750, 1000, 333]
    assert ih.joint_to_register(ih.register_to_joint(u, jm), jm) == u
    assert ih.joint_to_register([9.0] * 6, jm) == [0, 0, 0, 0, 0, 0]       # clipped


def test_the_cli_takes_its_defaults_from_the_config():
    path = os.path.join(os.path.dirname(__file__), "..", "config", "inspire_hand.yaml")
    d = ih._config_defaults(["--config", path, "status"])
    assert d["id"] == 2 and d["baud"] == 115200 and d["close_mode"] in ih.CLOSE_MODES


# -------------------------------------------------------------------- the node
def test_node_force_validation():
    node = pytest.importorskip("ycb_pointcloud_pipeline.inspire_hand_node")
    assert node.check_force(400, []) is None
    assert node.check_force(400, [1, 2, 3, 4, 5, 6]) is None
    assert "0-1000" in node.check_force(1500, [])
    assert "six" in node.check_force(400, [1, 2])


def test_node_joint_states_are_urdf_joints_in_radians():
    node = pytest.importorskip("ycb_pointcloud_pipeline.inspire_hand_node")
    jm = ih.joint_map_from_config(_config())
    st = {"angle": [1000, 1000, 0, 500, 1000, 0], "force_g": [0, 0, 380, 0, 0, 0]}
    msg = node.joint_state_message(st, jm, _stamp())
    assert list(msg.name) == list(ih.URDF_JOINTS)
    assert msg.position[2] == pytest.approx(1.676) and msg.position[3] == pytest.approx(0.8635)
    assert msg.position[5] == pytest.approx(1.308)
    assert msg.effort[2] == 380.0


def _stamp():
    from builtin_interfaces.msg import Time
    return Time()


def test_the_cli_sets_six_values_thumb_rotation_first(fake):
    assert ih.main(["--port", fake.port, "--id", "2", "--timeout", "0.3",
                    "set", "500", "500", "500", "500", "500", "300"]) == 0
    assert fake.angle_sets[-2] == [-1, -1, -1, -1, -1, 300]
    assert fake.angle_sets[-1] == [500, 500, 500, 500, 500, 300]
    assert ih.main(["--port", fake.port, "set", "500", "500"]) == 2
