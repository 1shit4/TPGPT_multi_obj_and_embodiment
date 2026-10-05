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

    ``contact_at[k]``: a DOF commanded below this angle stops there with FORCE_ACT = its
    threshold and STATUS ``contact_status``: 2 ("at target"), as the real hand reports a
    force stop (2026-10-05), or 3, the manual's. ``stuck[k]``: a commanded DOF reports
    STATUS 1 (moving) for ever. A DOF held with -1 has its STATUS reset to 2 by the next
    ANGLE_SET -- the worst case for judging a DOF by the final status, which the
    driver must not do.
    """

    def __init__(self, hand_id=2, contact_at=None, stuck=None, contact_status=2,
                 obstruct_together=False):
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
        self.contact_status = contact_status
        # The thumb lying on the fingers: an ANGLE_SET that opens fingers and the thumb
        # bend together leaves the fingers where they are (what the user saw, 2026-10-05).
        self.obstruct_together = obstruct_together
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
        if self.obstruct_together and values[4] > self.angle[4] and any(
                v != -1 and v > self.angle[k] for k, v in enumerate(values[:4])):
            values = [-1] * 4 + list(values[4:])           # the fingers are blocked
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
                self.angle[k], self.status[k], self.force[k] = stop, self.contact_status, self.force_set[k]
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


def test_open_from_rest_only_swings_the_thumb_and_never_sends_minus_one_to_it(fake):
    hand = ih.InspireHand(fake.port, 2, timeout=0.5, log=lambda *_: None)
    try:
        st = hand.grasp_open(timeout=1.0, verbose=False)
    finally:
        hand.close_port()
    # From the fake's rest (fingers straight, thumb rotation 55): the straight thumb rotates
    # into opposition, nothing else moves; then the whole pose is confirmed.
    assert fake.angle_sets == [[-1, -1, -1, -1, -1, ih.THUMB_OPPOSED], ih.OPEN_POSE]
    assert all(s[5] != -1 for s in fake.angle_sets)
    assert fake.angle[5] == ih.THUMB_OPPOSED
    assert not hand.holding


def _goto(fake, start, target):
    fake.angle = list(start)
    hand = ih.InspireHand(fake.port, 2, timeout=0.5, log=lambda *_: None)
    try:
        hand.goto(target, timeout=0.5, verbose=False)
    finally:
        hand.close_port()
    return fake.angle_sets


def test_closing_moves_the_fingers_before_the_thumb(fake):
    """The user, 2026-10-05: moved together the thumb obstructs the fingers. A fist: the
    fingers close first, the thumb bends over them last."""
    sets = _goto(fake, [1000, 1000, 1000, 1000, 1000, 0], [0, 0, 0, 0, 0, 0])
    assert sets == [[0, 0, 0, 0, -1, 0], [-1, -1, -1, -1, 0, 0], [0, 0, 0, 0, 0, 0]]


def test_opening_lifts_the_thumb_before_the_fingers(fake):
    sets = _goto(fake, [0, 0, 0, 0, 0, 0], [1000, 1000, 1000, 1000, 1000, 0])
    assert sets == [[-1, -1, -1, -1, 1000, 0], [1000, 1000, 1000, 1000, -1, 0],
                    [1000, 1000, 1000, 1000, 1000, 0]]


def test_the_thumb_rotates_only_straight_with_the_fingers_open(fake):
    # From a fist to the pointing pose: thumb up, every finger open, rotate, then curl three.
    sets = _goto(fake, [0, 0, 0, 0, 0, 0], ih.POINT_POSE)
    assert sets == [[-1, -1, -1, -1, 1000, 0], [1000, 1000, 1000, 1000, -1, 0],
                    [-1, -1, -1, -1, -1, 1000], [0, 0, 0, -1, -1, 1000], ih.POINT_POSE]


def test_a_pose_is_six_values(fake):
    hand = ih.InspireHand(fake.port, 2, timeout=0.5, log=lambda *_: None)
    try:
        with pytest.raises(ValueError):
            hand.goto([0, 0, 0])
    finally:
        hand.close_port()


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
    # Stopped short of the close's 0 (index at 500, status 2 as this hand reports a force
    # stop), or the manual's STATUS 3 (ring, at 1000): contact. Little at 400 too.
    assert ih.contact_dofs(st) == ["little", "ring", "index", "thumb_bend"]
    assert ih.closure(st) == [0.6, 0.0, 1.0, 0.5, 0.0, 1.0]
    assert ih.error_text(st) == "middle: locked rotor"


def test_config_builds_the_mount():
    cfg = ih.load_config(os.path.join(os.path.dirname(__file__), "..", "config",
                                      "inspire_hand.yaml"))
    assert cfg["handedness"] == "right"
    assert cfg["serial"]["hand_id"] == 1                    # 2 until 2026-10-05
    ee_mod = pytest.importorskip("tpgpt.hardware.end_effector")
    try:
        ee = ee_mod.inspire_from_config(cfg)
    except FileNotFoundError as exc:                       # no GraspGen-X checkout
        pytest.skip(str(exc))
    assert ee.controller_frame == "fr3_hand_tcp"
    assert ee.kind == "multi_finger" and ee.graspgen == "inspire_hand"
    # Measured by touch (M3, 2026-10-05, both fingertips, the ring concentric): 137.1 deg, 12.45 mm.
    assert ee.model_spec["mount_measured"]
    assert ee.model_spec["mount_yaw_deg"] == pytest.approx(137.11)
    assert ee.model_spec["adapter_thickness"] == pytest.approx(0.01245)


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


@pytest.mark.parametrize("contact_status", [2, 3])     # this hand's (2026-10-05); the manual's
@pytest.mark.parametrize("mode", ih.CLOSE_MODES)
def test_the_report_names_contact_and_air_dof_by_dof(mode, contact_status):
    # An object between the thumb and the index and middle fingers; little and ring miss it.
    contact = [None, None, 420, 380, 300, None]
    fake = FakeHand(contact_at=contact, contact_status=contact_status)
    try:
        report = _close(fake, mode, force=350)
    finally:
        fake.close()
    assert report.contact == ["middle", "index", "thumb_bend"]
    assert report.on_air == ["little", "ring"]
    assert report.timed_out == [] and report.errors == ""
    assert report.opposed and report.ok and report.held
    assert report.on_object == ["middle"]          # index 380: within 30 of the empty fist's 361
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


def test_the_default_close_is_simultaneous():
    """The user, 2026-10-05: sequential pushes the object away; simultaneous holds it."""
    assert ih.DEFAULT_CLOSE_MODE == "simultaneous"
    assert ih.load_config(os.path.join(os.path.dirname(__file__), "..", "config",
                                       "inspire_hand.yaml"))["grasp"]["close_mode"] == "simultaneous"


def test_a_grasp_seen_on_the_real_hand_is_contact_not_air():
    """The 2026-10-05 banana: every DOF read STATUS 2, stopped 216-508 short of 0."""
    fake = FakeHand(contact_at=[245, 216, 243, 406, 508, None])
    try:
        report = _close(fake, "simultaneous")
    finally:
        fake.close()
    assert report.status[:5] == [2] * 5
    assert report.contact == list(ih.BENDING) and report.on_air == []
    assert report.opposed and report.held
    assert report.on_object == ["little", "ring", "middle", "index"]


def test_an_empty_fist_is_not_held_though_the_thumb_stops_two_fingers(fake):
    """The user's empty close, 2026-10-05: [0, 0, 180, 361, 493] -- the thumb landed on the
    middle and index fingers. 'Opposed', and nothing in the hand."""
    fake.contact_at = [None, None, 180, 361, 493, None]
    hand = _hand(fake)
    try:
        report = hand.close("simultaneous", timeout=0.6, verbose=False)
        assert report.contact == ["middle", "index", "thumb_bend"]
        assert report.opposed and not report.held and report.on_object == []
        assert "NOTHING HELD" in report.summary()
        assert not hand.holding
        n = len(fake.angle_sets)
        hand.grasp_open(timeout=0.6, verbose=False)
    finally:
        hand.close_port()
    assert fake.angle_sets[n][:5] == [-1, -1, -1, -1, 1000]     # the ordered open


def test_a_release_the_thumb_blocks_is_redone_in_order():
    """The user's second 150 g empty close (2026-10-05): middle 349, read as held; the
    one-stage open left the fingers under the thumb. The open checks and opens in order."""
    fake = FakeHand(contact_at=[None, None, 349, 361, 491, None], obstruct_together=True)
    hand = _hand(fake)
    try:
        assert hand.close("simultaneous", timeout=0.6, verbose=False).held   # the false 'held'
        n = len(fake.angle_sets)
        hand.grasp_open(timeout=0.6, verbose=False)
    finally:
        hand.close_port()
        fake.close()
    sets = fake.angle_sets[n:]
    assert sets[0] == ih.OPEN_POSE                                 # the one-stage try
    # Then in order from where it got to (here the thumb was already up): no step moves
    # the fingers and the thumb bend together, until the final confirming pose.
    for v in sets[1:-1]:
        assert v[4] == -1 or all(x == -1 for x in v[:4]), v
    assert sets[1][:4] == [1000] * 4
    assert fake.angle[:5] == [1000] * 5                            # and the hand is open


def test_the_mug_close_of_2026_10_05_is_held():
    """Little and ring on air; the mug stops the middle finger 218 beyond the empty fist."""
    fake = FakeHand(contact_at=[None, None, 398, 414, 561, None])
    try:
        report = _close(fake, "simultaneous")
    finally:
        fake.close()
    assert report.on_air == ["little", "ring"]
    assert report.held and report.on_object == ["middle", "index"]


def test_the_config_holds_the_measured_empty_close():
    cfg = ih.load_config(os.path.join(os.path.dirname(__file__), "..", "config",
                                      "inspire_hand.yaml"))
    assert cfg["grasp"]["empty_close_angles"] == ih.EMPTY_CLOSE_ANGLES


def test_after_a_close_that_held_the_release_opens_everything_at_once():
    """The user, 2026-10-05: an open in stages (the thumb first) shifts the object at the
    place. With the object between thumb and fingers they cannot meet: one stage."""
    fake = FakeHand(contact_at=[None, 600, 550, 500, 650, None])
    hand = _hand(fake)
    try:
        assert hand.close("simultaneous", timeout=0.6, verbose=False).opposed
        assert hand.holding
        n = len(fake.angle_sets)
        hand.grasp_open(timeout=0.6, verbose=False)
    finally:
        hand.close_port()
        fake.close()
    assert fake.angle_sets[n:] == [ih.OPEN_POSE]
    assert not hand.holding


def test_after_a_close_on_air_the_open_still_lifts_the_thumb_first(fake):
    """A fist closed on air may have the thumb over the fingers: the ordered open."""
    hand = _hand(fake)
    try:
        assert not hand.close("simultaneous", timeout=0.6, verbose=False).opposed
        n = len(fake.angle_sets)
        hand.grasp_open(timeout=0.6, verbose=False)
    finally:
        hand.close_port()
    sets = fake.angle_sets[n:]
    assert sets[0][:5] == [-1, -1, -1, -1, 1000]                # the thumb lifts first
    assert sets[1][:5] == [1000, 1000, 1000, 1000, -1]          # then the fingers open
    assert sets[-1] == ih.OPEN_POSE


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
    # Anything missing from the config falls back to the default (the ruler fit), unverified.
    assert ih.joint_map_from_config({})["index"] == (1.373, 0.120, False)


def test_registers_and_joint_angles_round_trip():
    jm = ih.joint_map_from_config(_config())
    q = ih.register_to_joint([1000, 1000, 1000, 1000, 1000, 0], jm)
    assert q == pytest.approx([0.128, 0.122, 0.123, 0.120, 0.082, 1.308])  # open, opposed
    q = ih.register_to_joint([0, 0, 0, 0, 0, 0], jm)
    assert q == pytest.approx([1.373, 1.373, 1.373, 1.373, 0.650, 1.308])
    u = [0, 250, 500, 750, 1000, 333]
    assert ih.joint_to_register(ih.register_to_joint(u, jm), jm) == u
    assert ih.joint_to_register([9.0] * 6, jm) == [0, 0, 0, 0, 0, 0]       # clipped


def test_the_cli_takes_its_defaults_from_the_config():
    path = os.path.join(os.path.dirname(__file__), "..", "config", "inspire_hand.yaml")
    d = ih._config_defaults(["--config", path, "status"])
    assert d["id"] == 1 and d["baud"] == 115200 and d["close_mode"] in ih.CLOSE_MODES


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
    assert msg.position[2] == pytest.approx(1.373) and msg.position[3] == pytest.approx(0.7465)
    assert msg.position[5] == pytest.approx(1.308)
    assert msg.effort[2] == 380.0


def _stamp():
    from builtin_interfaces.msg import Time
    return Time()


def test_the_cli_set_goes_through_the_ordered_pose(fake):
    assert ih.main(["--port", fake.port, "--id", "2", "--timeout", "0.3",
                    "set", "500", "500", "500", "500", "500", "300"]) == 0
    # rest (fingers straight, rotation 55): rotate the straight thumb, close the fingers,
    # bend the thumb, confirm
    assert fake.angle_sets == [[-1, -1, -1, -1, -1, 300], [500, 500, 500, 500, -1, 300],
                               [-1, -1, -1, -1, 500, 300], [500, 500, 500, 500, 500, 300]]
    assert ih.main(["--port", fake.port, "set", "500", "500"]) == 2
