"""
Object detectors/segmenters for the RGB-D -> point cloud pipeline.

GroundedSAMDetector: Grounding DINO (open-vocabulary, text-prompted boxes)
followed by SAM (box -> pixel mask). Both models are loaded through
HuggingFace `transformers`, so there is no CUDA extension to compile on
Jetson Thor and checkpoints are downloaded automatically on first run
(cached in ~/.cache/huggingface).

The text prompts come from config/target_objects.yaml, so only the objects
listed there are ever searched for.
"""

import os
# transformers probes for TF/Flax backends on import; a stray/partial
# jax or tensorflow install anywhere on the system crashes that probe.
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np


class DummyDetector:
    """No model dependencies. Fixed center-box mask for the first target.
    Use to verify camera topics / point cloud plumbing before loading models."""

    def __init__(self, targets, **kwargs):
        self.label = targets[0]['name']

    def detect_and_segment(self, rgb_image):
        h, w = rgb_image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = True
        return [{"label": self.label, "score": 1.0, "mask": mask,
                 "box": (w // 3, h // 3, 2 * w // 3, 2 * h // 3)}]


#: Boxes below the accept threshold are still reported, down to this score, so a
#: miss can be told from a near miss. Diagnostic only: nothing is published from it.
REPORT_FLOOR = 0.05


def build_captions(targets, mode="joint"):
    """``[(caption, {prompt: name})]`` for Grounding DINO.

    ``joint`` puts every phrase in one caption and runs one forward pass -- what
    this node has always done. ``per_object`` gives each phrase a caption of its
    own, at one forward pass each: a phrase's score depends on the caption it
    sits in, because the text tokens compete, so an object that scores badly
    beside four others may score well alone. Which is true on this bench is a
    measurement, not a guess -- ``scripts/prompt_bench.py`` makes it.
    """
    pairs = [(t['prompt'].lower().strip(), t['name']) for t in targets]
    if mode == "joint":
        return [(" ".join(f"{prompt}." for prompt, _ in pairs), dict(pairs))]
    if mode == "per_object":
        return [(f"{prompt}.", {prompt: name}) for prompt, name in pairs]
    raise ValueError(f"unknown caption_mode {mode!r}; use 'joint' or 'per_object'")


def phrase_to_name(phrase, prompt_to_name):
    """Map a matched phrase back to a target name, or ``None``.

    Grounding DINO returns the part of the caption it matched, which is often
    only part of a prompt ("can" for "tomato soup can"), so pick the prompt with
    the best word overlap. A tie means two prompts explain the phrase equally
    well and the detection cannot be attributed; those are dropped, and the
    node reports them rather than losing them silently.
    """
    phrase = phrase.lower().strip().rstrip('.')
    if phrase in prompt_to_name:
        return prompt_to_name[phrase]
    words = set(phrase.split())
    if not words:
        return None
    scored = sorted(((len(words & set(prompt.split())) / len(words | set(prompt.split())), name)
                     for prompt, name in prompt_to_name.items()), reverse=True)
    if scored[0][0] == 0 or (len(scored) > 1 and scored[0][0] == scored[1][0]):
        return None
    return scored[0][1]


def _box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class GroundedSAMDetector:
    """
    targets: list of {"name": str, "prompt": str} from target_objects.yaml.
    Returns detections labelled with the target *name* (not the raw phrase
    Grounding DINO matched), keeping at most one instance per target.
    """

    def __init__(self, targets, grounding_model="IDEA-Research/grounding-dino-base",
                 sam_model="facebook/sam-vit-large", box_threshold=0.35,
                 text_threshold=0.25, device=None, logger=None, caption_mode="joint",
                 thresholds=None, report_floor=REPORT_FLOOR, load_sam=True):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.targets = targets
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log = logger.info if logger else print
        if self.device == "cpu":
            (logger.warn if logger else print)(
                "CUDA not available -- running detection on CPU (slow)")

        log(f"Loading {grounding_model} on {self.device} ...")
        self.gd_processor = AutoProcessor.from_pretrained(grounding_model)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model).to(self.device).eval()

        # The boxes are Grounding DINO's; SAM only turns a box into a mask. Scoring
        # prompts (prompt_bench) needs the boxes alone, and SAM is the larger model.
        self.sam_processor = self.sam_model = None
        if load_sam:
            from transformers import SamModel, SamProcessor
            log(f"Loading {sam_model} on {self.device} ...")
            self.sam_processor = SamProcessor.from_pretrained(sam_model)
            self.sam_model = SamModel.from_pretrained(sam_model).to(self.device).eval()

        # Grounding DINO expects lowercase phrases, each terminated by "."
        self.prompt_to_name = {t['prompt'].lower().strip(): t['name'] for t in targets}
        self.caption_mode = caption_mode
        self.captions = build_captions(targets, caption_mode)
        self.caption = self.captions[0][0]           # what a joint pass sends; kept for the record
        #: Accept score per target: its own ``threshold:`` in target_objects.yaml,
        #: else ``box_threshold``. One flaky object should not have to drag the
        #: threshold down for every other object in the caption.
        self.thresholds = {t['name']: float((thresholds or {}).get(t['name'], box_threshold))
                           for t in targets}
        self.report_floor = float(report_floor)
        #: Filled every frame: one entry per target, accepted or not. A target that
        #: is missing from a frame is the thing we most need to know about, and it
        #: used to leave no trace at all.
        self.last_report = {}
        #: Boxes whose phrase matched two prompts equally well, so they could not be
        #: attributed. Also silent before.
        self.last_unmatched = []
        log(f"caption mode {caption_mode}, {len(self.captions)} pass(es) per frame: "
            + " | ".join(caption for caption, _ in self.captions))

    def _phrase_to_name(self, phrase):
        """Deprecated: use :func:`phrase_to_name`. Kept so callers do not break."""
        return phrase_to_name(phrase, self.prompt_to_name)

    def boxes(self, pil, caption, mapping, size, floor):
        """The best box per target name in one caption, at or above ``floor``."""
        torch = self.torch
        best, unmatched = {}, []
        with torch.inference_mode():
            inputs = self.gd_processor(images=pil, text=caption,
                                       return_tensors="pt").to(self.device)
            outputs = self.gd_model(**inputs)
            res = self.gd_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=floor,
                text_threshold=self.text_threshold, target_sizes=[size])[0]
        phrases = res["text_labels"] if "text_labels" in res else res["labels"]
        for box, score, phrase in zip(res["boxes"].tolist(), res["scores"].tolist(), phrases):
            name = phrase_to_name(phrase, mapping)
            if name is None:
                # An empty phrase is a box whose text never cleared text_threshold --
                # the detector proposing a region with nothing to call it, of which
                # there are several on any frame. Only a phrase that exists and
                # cannot be attributed is worth reporting.
                if phrase.strip():
                    unmatched.append({"phrase": phrase, "score": float(score)})
                continue
            if name not in best or score > best[name]["score"]:
                best[name] = {"label": name, "score": float(score), "box": tuple(box),
                              "phrase": phrase, "caption": caption}
        return best, unmatched

    def detect_and_segment(self, rgb_image):
        """rgb_image: HxWx3 uint8 (RGB). Returns list of
        {"label", "score", "mask" (HxW bool), "box" (x0, y0, x1, y1)}.

        Detection is run at :data:`REPORT_FLOOR` and filtered at each target's own
        threshold afterwards, which accepts exactly what the single threshold used
        to accept and additionally records, in :attr:`last_report`, how close every
        target came. An object that appears on one frame and not the next is the
        case this node exists to explain.
        """
        from PIL import Image
        pil = Image.fromarray(rgb_image)
        h, w = rgb_image.shape[:2]
        floor = min([self.report_floor] + list(self.thresholds.values()))

        best, unmatched = {}, []
        for caption, mapping in self.captions:
            found, missed = self.boxes(pil, caption, mapping, (h, w), floor)
            unmatched.extend(missed)
            for name, d in found.items():
                if name not in best or d["score"] > best[name]["score"]:
                    best[name] = d
        self.last_report = {
            t["name"]: {"score": best.get(t["name"], {}).get("score", 0.0),
                        "phrase": best.get(t["name"], {}).get("phrase"),
                        "threshold": self.thresholds[t["name"]],
                        "accepted": (best.get(t["name"], {}).get("score", 0.0)
                                     >= self.thresholds[t["name"]])}
            for t in self.targets}
        self.last_unmatched = sorted(unmatched, key=lambda u: -u["score"])[:5]

        # One physical object detected under two names -> keep higher score
        dets = sorted((d for d in best.values() if d["score"] >= self.thresholds[d["label"]]),
                      key=lambda d: -d["score"])
        kept = []
        for d in dets:
            # Two targets on one box: one physical object was found under two
            # names, and only the higher score can be right. The loser used to
            # vanish without a trace -- the very case of one object's prompt landing
            # on another object -- so it is written into the report.
            clash = next((k for k in kept if _box_iou(d["box"], k["box"]) >= 0.7), None)
            if clash is None:
                kept.append(d)
            else:
                self.last_report[d["label"]].update(
                    lost_to=clash["label"], lost_to_score=clash["score"],
                    overlap=_box_iou(d["box"], clash["box"]))
        if not kept:
            return []
        if self.sam_model is None:
            raise RuntimeError(
                "this detector was built with load_sam=False, so it can score boxes "
                "(see boxes(), prompt_bench) but cannot produce masks")

        torch = self.torch
        with torch.inference_mode():
            sam_inputs = self.sam_processor(
                pil, input_boxes=[[list(d["box"]) for d in kept]],
                return_tensors="pt").to(self.device)
            sam_out = self.sam_model(**sam_inputs, multimask_output=False)
            masks = self.sam_processor.image_processor.post_process_masks(
                sam_out.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu())[0]  # (N, 1, H, W)

        for d, m in zip(kept, masks):
            d["mask"] = m[0].numpy().astype(bool)
        return kept
