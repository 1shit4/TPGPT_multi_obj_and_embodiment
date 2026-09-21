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
                 text_threshold=0.25, device=None, logger=None):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  SamModel, SamProcessor)

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

        log(f"Loading {sam_model} on {self.device} ...")
        self.sam_processor = SamProcessor.from_pretrained(sam_model)
        self.sam_model = SamModel.from_pretrained(sam_model).to(self.device).eval()

        # Grounding DINO expects lowercase phrases, each terminated by "."
        self.prompt_to_name = {t['prompt'].lower().strip(): t['name'] for t in targets}
        self.caption = " ".join(f"{p}." for p in self.prompt_to_name)

    def _phrase_to_name(self, phrase):
        """Map a matched phrase back to a target name. Grounding DINO sometimes
        returns only part of a prompt ("can" for "tomato soup can"), so pick
        the prompt with the best word overlap and drop ambiguous ties."""
        phrase = phrase.lower().strip().rstrip('.')
        if phrase in self.prompt_to_name:
            return self.prompt_to_name[phrase]
        words = set(phrase.split())
        if not words:
            return None
        scored = []
        for prompt, name in self.prompt_to_name.items():
            pw = set(prompt.split())
            scored.append((len(words & pw) / len(words | pw), name))
        scored.sort(reverse=True)
        if scored[0][0] == 0 or (len(scored) > 1 and scored[0][0] == scored[1][0]):
            return None
        return scored[0][1]

    def detect_and_segment(self, rgb_image):
        """rgb_image: HxWx3 uint8 (RGB). Returns list of
        {"label", "score", "mask" (HxW bool), "box" (x0, y0, x1, y1)}."""
        from PIL import Image
        torch = self.torch
        pil = Image.fromarray(rgb_image)
        h, w = rgb_image.shape[:2]

        with torch.inference_mode():
            inputs = self.gd_processor(images=pil, text=self.caption,
                                       return_tensors="pt").to(self.device)
            outputs = self.gd_model(**inputs)
            res = self.gd_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=[(h, w)])[0]

        phrases = res["text_labels"] if "text_labels" in res else res["labels"]
        # Best instance per target name, highest score first
        best = {}
        for box, score, phrase in zip(res["boxes"].tolist(), res["scores"].tolist(), phrases):
            name = self._phrase_to_name(phrase)
            if name is None:
                continue
            if name not in best or score > best[name]["score"]:
                best[name] = {"label": name, "score": score, "box": tuple(box)}

        # One physical object detected under two names -> keep higher score
        dets = sorted(best.values(), key=lambda d: -d["score"])
        kept = []
        for d in dets:
            if all(_box_iou(d["box"], k["box"]) < 0.7 for k in kept):
                kept.append(d)
        if not kept:
            return []

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
