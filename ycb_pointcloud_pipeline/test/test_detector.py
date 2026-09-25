"""The detector's bookkeeping: captions, phrase attribution, thresholds, reports.

None of this needs a model. What it pins down is the part that decides whether an
object reaches a topic, and -- since 2026-09-25 -- whether a frame that dropped
one says why.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ycb_pointcloud_pipeline.ycb_detector import (  # noqa: E402
    GroundedSAMDetector, build_captions, phrase_to_name)

TARGETS = [{'name': 'banana', 'prompt': 'banana'},
           {'name': 'apple', 'prompt': 'apple'},
           {'name': 'pudding_box', 'prompt': 'brown chocolate pudding box'}]


def test_joint_caption_is_one_pass_with_every_phrase():
    captions = build_captions(TARGETS, 'joint')
    assert len(captions) == 1
    caption, mapping = captions[0]
    assert caption == 'banana. apple. brown chocolate pudding box.'
    assert mapping['brown chocolate pudding box'] == 'pudding_box'


def test_per_object_caption_is_one_pass_each_and_cannot_cross_attribute():
    captions = build_captions(TARGETS, 'per_object')
    assert len(captions) == 3
    for caption, mapping in captions:
        assert caption.count('.') == 1
        assert len(mapping) == 1
    # A pass for the banana can only ever produce a banana, whatever it matched.
    _, mapping = captions[0]
    assert phrase_to_name('brown box', mapping) is None


def test_unknown_caption_mode_is_refused():
    with pytest.raises(ValueError):
        build_captions(TARGETS, 'one_word_at_a_time')


def test_partial_phrases_map_back_to_the_target():
    mapping = build_captions(TARGETS, 'joint')[0][1]
    assert phrase_to_name('pudding box', mapping) == 'pudding_box'
    assert phrase_to_name('brown chocolate pudding box', mapping) == 'pudding_box'
    assert phrase_to_name('banana', mapping) == 'banana'
    assert phrase_to_name('spatula', mapping) is None


def test_an_ambiguous_phrase_is_not_attributed():
    targets = [{'name': 'red_box', 'prompt': 'red box'},
               {'name': 'brown_box', 'prompt': 'brown box'}]
    mapping = build_captions(targets, 'joint')[0][1]
    assert phrase_to_name('box', mapping) is None       # equal overlap with both


def _detector(thresholds=None, found=None, mode='joint'):
    """A detector with its models replaced by a fixed set of boxes."""
    det = GroundedSAMDetector.__new__(GroundedSAMDetector)
    det.targets = TARGETS
    det.captions = build_captions(TARGETS, mode)
    det.thresholds = {t['name']: (thresholds or {}).get(t['name'], 0.35) for t in TARGETS}
    det.report_floor = 0.05
    det.text_threshold = 0.25
    det.boxes = lambda pil, caption, mapping, size, floor, box_ok=None: (dict(found or {}), [])
    return det


def test_a_near_miss_is_recorded_rather_than_lost():
    det = _detector(found={'pudding_box': {'label': 'pudding_box', 'score': 0.29,
                                           'box': (1, 1, 5, 5), 'phrase': 'pudding box'}})
    assert det.detect_and_segment(np.zeros((8, 8, 3), np.uint8)) == []
    report = det.last_report['pudding_box']
    assert report['accepted'] is False
    assert report['score'] == pytest.approx(0.29)
    assert report['phrase'] == 'pudding box'
    # and the targets that were never seen say so, rather than being absent
    assert det.last_report['banana'] == {'score': 0.0, 'phrase': None, 'threshold': 0.35,
                                         'accepted': False}


class _Fake(dict):
    """Enough of torch and SAM to let an accepted box reach a mask."""

    def to(self, _device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.ones((8, 8), dtype=bool)

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, _name):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getitem__(self, _key):
        return self

    def post_process_masks(self, *a, **k):
        return [[[self]]]


def _with_sam(det):
    det.torch = _Fake()
    det.sam_processor = _Fake()
    det.sam_model = _Fake()
    det.device = 'cpu'
    return det


def test_a_per_object_threshold_admits_what_the_shared_one_refuses():
    found = {'pudding_box': {'label': 'pudding_box', 'score': 0.29, 'box': (1, 1, 5, 5),
                             'phrase': 'pudding box'}}
    assert _detector(found=found).detect_and_segment(np.zeros((8, 8, 3), np.uint8)) == []
    det = _with_sam(_detector(thresholds={'pudding_box': 0.25}, found=found))
    dets = det.detect_and_segment(np.zeros((8, 8, 3), np.uint8))
    assert [d['label'] for d in dets] == ['pudding_box']
    assert det.last_report['pudding_box']['accepted'] is True
    assert dets[0]['mask'].all()


def test_the_report_floor_lets_the_boxes_call_see_below_every_threshold():
    seen = {}
    det = _detector(thresholds={'pudding_box': 0.5})
    det.boxes = lambda pil, caption, mapping, size, floor, box_ok=None: (seen.update(floor=floor) or ({}, []))
    det.detect_and_segment(np.zeros((8, 8, 3), np.uint8))
    assert seen['floor'] == 0.05          # the floor, not the lowest threshold


def test_boxes_reports_an_unattributable_phrase_but_not_an_empty_one():
    """A box with no text is the detector proposing a region, not an ambiguity."""
    det = _detector()
    del det.boxes                      # this one exercises the real method
    det.gd_processor = _Fake()
    det.gd_model = _Fake()
    det.torch = _Fake()
    det.device = 'cpu'

    class Res(dict):
        def __getitem__(self, key):
            return {'boxes': _Scores([[0, 0, 1, 1], [0, 0, 2, 2]]),
                    'scores': _Scores([0.44, 0.31]),
                    'text_labels': ['box', '']}[key]

        def __contains__(self, key):
            return key == 'text_labels'

    det.gd_processor.post_process_grounded_object_detection = lambda *a, **k: [Res()]
    targets = [{'name': 'red_box', 'prompt': 'red box'},
               {'name': 'brown_box', 'prompt': 'brown box'}]
    caption, mapping = build_captions(targets, 'joint')[0]
    found, unmatched = det.boxes(None, caption, mapping, (8, 8), 0.05)
    assert found == {}
    assert unmatched == [{'phrase': 'box', 'score': pytest.approx(0.44)}]


class _Scores(list):
    def tolist(self):
        return list(self)


def test_load_target_objects_reads_an_optional_threshold(tmp_path):
    pytest.importorskip('rclpy')
    from ycb_pointcloud_pipeline.segmentation_node import load_target_objects
    path = tmp_path / 'targets.yaml'
    path.write_text("target_objects:\n  - banana\n  - name: pudding_box\n"
                    "    prompt: brown box\n    threshold: 0.22\n")
    targets = load_target_objects(str(path))
    assert targets[0] == {'name': 'banana', 'prompt': 'banana'}
    assert targets[1]['threshold'] == pytest.approx(0.22)


# ---------------------------------------------------------------- prompt_bench
class _Boxes:
    """A detector that answers every caption with fixed boxes per phrase."""

    def __init__(self, by_prompt):
        self.by_prompt = by_prompt

    def boxes(self, pil, caption, mapping, size, floor):
        found = {}
        for prompt, name in mapping.items():
            if prompt in self.by_prompt:
                score, box = self.by_prompt[prompt]
                found[name] = {'score': score, 'box': box}
        return found, []


BENCH_TARGETS = [{'name': 'pudding_box', 'prompt': 'brown chocolate pudding box'},
                 {'name': 'foam_brick', 'prompt': 'red foam brick'}]
FRAME = [('f0', np.zeros((8, 8, 3), np.uint8))]
BRICK = (10, 10, 50, 40)
PUDDING = (100, 10, 150, 80)


def test_a_broad_prompt_that_lands_on_the_other_box_is_reported_as_confused():
    from ycb_pointcloud_pipeline.prompt_bench import score_prompt
    detector = _Boxes({'brown box': (0.81, BRICK), 'red foam brick': (0.66, BRICK)})
    s, confused, who = score_prompt(detector, FRAME, BENCH_TARGETS, 'pudding_box',
                                    'brown box', 'joint')
    assert s[0] == pytest.approx(0.81)      # a high score...
    assert confused[0]                      # ...on the wrong object
    assert who == ['foam_brick']


def test_a_prompt_on_its_own_object_is_not_confused():
    from ycb_pointcloud_pipeline.prompt_bench import score_prompt
    detector = _Boxes({'brown chocolate pudding box': (0.44, PUDDING),
                       'red foam brick': (0.66, BRICK)})
    s, confused, _ = score_prompt(detector, FRAME, BENCH_TARGETS, 'pudding_box',
                                  'brown chocolate pudding box', 'per_object')
    assert s[0] == pytest.approx(0.44)
    assert not confused[0]


def test_iou_is_zero_for_a_missing_box():
    from ycb_pointcloud_pipeline.prompt_bench import iou
    assert iou(None, BRICK) == 0.0
    assert iou(BRICK, BRICK) == pytest.approx(1.0)
    assert iou(BRICK, PUDDING) == 0.0


def test_a_target_dropped_for_sharing_a_box_says_which_target_took_it():
    """One prompt landing on another object: the loser used to vanish silently."""
    same = (10, 10, 50, 40)
    found = {'pudding_box': {'label': 'pudding_box', 'score': 0.52, 'box': same,
                             'phrase': 'brown chocolate pudding box'},
             'apple': {'label': 'apple', 'score': 0.61, 'box': same, 'phrase': 'apple'}}
    det = _with_sam(_detector(found=found))
    dets = det.detect_and_segment(np.zeros((8, 8, 3), np.uint8))
    assert [d['label'] for d in dets] == ['apple']
    lost = det.last_report['pudding_box']
    assert lost['accepted'] is True                 # it cleared its threshold...
    assert lost['lost_to'] == 'apple'               # ...and lost the box to the apple
    assert lost['overlap'] == pytest.approx(1.0)


# ------------------------------------------------------------------ size gate
class _Res(dict):
    """A Grounding DINO result: boxes, scores and their phrases."""

    def __init__(self, boxes, scores, labels):
        super().__init__()
        self.b, self.s, self.l = boxes, scores, labels

    def __getitem__(self, key):
        return {'boxes': _Scores(self.b), 'scores': _Scores(self.s), 'text_labels': self.l}[key]

    def __contains__(self, key):
        return key == 'text_labels'


def _real_boxes(result):
    det = _detector()
    del det.boxes
    det.gd_processor, det.gd_model, det.torch, det.device = _Fake(), _Fake(), _Fake(), 'cpu'
    det.gd_processor.post_process_grounded_object_detection = lambda *a, **k: [result]
    return det


SHELF = (100, 50, 700, 450)        # 600 x 400 px
BOX = (300, 200, 360, 260)         # 60 x 60 px


def test_a_box_too_big_for_the_object_loses_to_a_real_one_that_scored_lower():
    """The shelf scored 0.40 and the real pudding box 0.33 on the same frame (2026-09-25)."""
    det = _real_boxes(_Res([list(SHELF), list(BOX)], [0.40, 0.33],
                           ['brown chocolate pudding box', 'brown chocolate pudding box']))
    sizes = {SHELF: 0.62, BOX: 0.09}
    det.last_oversized = {}
    found, _ = det.boxes(None, *build_captions(TARGETS, 'joint')[0], (480, 848), 0.05,
                         box_ok=lambda name, box: (sizes[box] <= 0.21, sizes[box]))
    assert found['pudding_box']['box'] == BOX
    assert found['pudding_box']['score'] == pytest.approx(0.33)
    assert det.last_oversized['pudding_box']['size_m'] == pytest.approx(0.62)


def test_with_the_object_absent_the_oversized_box_is_reported_not_published():
    det = _real_boxes(_Res([list(SHELF)], [0.40], ['brown chocolate pudding box']))
    det.thresholds['pudding_box'] = 0.25
    out = det.detect_and_segment(np.zeros((8, 8, 3), np.uint8),
                                 box_ok=lambda name, box: (False, 0.62))
    assert out == []
    r = det.last_report['pudding_box']
    assert r['score'] == 0.0 and not r['accepted']
    assert r['oversized'] == {'score': pytest.approx(0.40), 'size_m': 0.62,
                              'phrase': 'brown chocolate pudding box'}


def _node_module():
    pytest.importorskip('rclpy')
    from ycb_pointcloud_pipeline import segmentation_node
    return segmentation_node


def test_box_size_is_the_larger_side_at_the_depth_inside_it():
    box_size_m = _node_module().box_size_m
    depth = np.zeros((480, 848), np.uint16)
    depth[200:260, 300:360] = 1000                       # 1.0 m, in mm
    # 60 px at 1 m with fx = 600 px -> 0.10 m
    assert box_size_m(BOX, depth, 600.0, 600.0) == pytest.approx(0.10)
    depth[:] = 1200                                      # the whole shelf patch at 1.2 m
    assert box_size_m(SHELF, depth, 600.0, 600.0) == pytest.approx(1.2)


def test_a_box_with_no_depth_has_no_size_and_is_not_refused_for_it():
    box_size_m = _node_module().box_size_m
    assert box_size_m(BOX, np.zeros((480, 848), np.uint16), 600.0, 600.0) is None


def test_load_target_objects_reads_max_size(tmp_path):
    load = _node_module().load_target_objects
    path = tmp_path / 't.yaml'
    path.write_text("target_objects:\n  - name: pudding_box\n    prompt: x\n    max_size: 0.142\n")
    assert load(str(path))[0]['max_size'] == pytest.approx(0.142)


def test_every_listed_object_has_a_size():
    """A target without max_size is never size-checked: the shelf could be it."""
    load = _node_module().load_target_objects
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'target_objects.yaml')
    missing = [t['name'] for t in load(path) if 'max_size' not in t]
    assert not missing, f"no max_size for {missing}"
