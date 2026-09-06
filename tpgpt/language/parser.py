"""Deterministic natural-language task parsing.

Turns "put the milk carton on the top shelf" into a pair of scene entity ids.
No model, no network, no API key: the same prompt and the same scene always
produce the same answer, and a failure is a missing word in
:mod:`tpgpt.language.vocabulary` rather than a sampling accident.

The design rule throughout is **never guess**. An unmatched or ambiguous prompt
returns a :class:`TaskSpec` with ``ok`` false, carrying the candidates that were
considered and why they were rejected, so a wrong answer is inspectable. The one
place a default is applied -- a level named without a lateral slot -- says so in
the rationale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tpgpt.language.vocabulary import (
    LATERAL_ALIASES,
    LEVEL_ALIASES,
    PICK_VERBS,
    PLACEMENT_PREPOSITIONS,
    RECEPTACLE_NOUNS,
    STOPWORDS,
)
from tpgpt.perception.scene_graph import SceneEntity, SceneGraph

#: Lateral slot assumed when a prompt names a level but not a position.
DEFAULT_LATERAL = "middle"


@dataclass
class TaskSpec:
    """The parsed intent, or a structured explanation of why parsing failed."""

    ok: bool
    prompt: str
    object_id: str | None = None
    destination_id: str | None = None
    object_label: str | None = None
    destination_label: str | None = None
    pick_phrase: str = ""
    place_phrase: str = ""
    rationale: list[str] = field(default_factory=list)
    candidates: dict = field(default_factory=dict)
    error: str | None = None

    def describe(self) -> str:
        if not self.ok:
            return f"parse failed: {self.error}"
        return (
            f"move {self.object_id} ({self.object_label}) "
            f"-> {self.destination_id} ({self.destination_label})"
        )


def normalise(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", text.lower())).strip()


def content_tokens(phrase: str) -> list[str]:
    """Tokens carrying selective information."""
    return [t for t in normalise(phrase).split() if t and t not in STOPWORDS]


def split_prompt(prompt: str) -> tuple[str, str]:
    """Split into a pick phrase and a place phrase at a placement preposition.

    The prepositions are tried longest first, so "on top of the shelf" splits at
    "on top of" rather than at the bare "on", which would otherwise leave
    "top of" glued to the destination and make "top" look like two words.
    """
    text = normalise(prompt)
    for verb in sorted(PICK_VERBS, key=len, reverse=True):
        if text.startswith(verb + " "):
            text = text[len(verb) + 1:]
            break
    for preposition in PLACEMENT_PREPOSITIONS:
        match = re.search(rf"\b{re.escape(preposition)}\b", text)
        if match:
            return text[: match.start()].strip(), text[match.end():].strip()
    return text, ""


def _score_entity(tokens: list[str], entity: SceneEntity) -> tuple[float, str]:
    """Best normalised overlap between the phrase and any name of the entity."""
    best, best_name = 0.0, ""
    for name in entity.all_names:
        name_tokens = set(content_tokens(name))
        if not name_tokens:
            continue
        overlap = len(name_tokens & set(tokens))
        if not overlap:
            continue
        # Reward covering the entity's name, and prefer longer matched names so
        # "milk carton" beats a bare "milk" when both appear.
        score = overlap / len(name_tokens) + 0.01 * overlap
        if score > best:
            best, best_name = score, name
    return best, best_name


def resolve_object(phrase: str, scene: SceneGraph) -> tuple[SceneEntity | None, dict, str]:
    """Match a phrase against the scene's objects.

    Returns:
        ``(entity, scores, note)``. ``entity`` is ``None`` when nothing matched
        or two objects tied.
    """
    tokens = content_tokens(phrase)
    if not tokens:
        return None, {}, "no object words in the prompt"

    scores, names = {}, {}
    for entity in scene.objects:
        score, name = _score_entity(tokens, entity)
        if score > 0:
            scores[entity.entity_id] = round(score, 4)
            names[entity.entity_id] = name
    if not scores:
        return None, scores, f"no object matched {phrase!r}"

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < 1e-9:
        tied = [entity_id for entity_id, score in ranked if score == ranked[0][1]]
        return None, scores, f"{phrase!r} matches {tied} equally well"
    chosen = scene.by_id(ranked[0][0])
    return chosen, scores, f"matched {phrase!r} to {chosen.label} via {names[chosen.entity_id]!r}"


def resolve_receptacle(phrase: str, scene: SceneGraph) -> tuple[SceneEntity | None, dict, str]:
    """Match a phrase against the scene's receptacles, structurally.

    A slot is identified by two independent fields -- its level and its lateral
    position -- so they are resolved separately instead of by comparing the
    phrase against a combinatorial list of names.
    """
    tokens = set(content_tokens(phrase))
    if not tokens:
        return None, {}, "no destination words in the prompt"

    receptacles = scene.receptacles
    if not receptacles:
        return None, {}, "the scene has no receptacles"

    levels = {e.metadata.get("level") for e in receptacles}
    laterals = {e.metadata.get("lateral") for e in receptacles}

    def match_field(aliases: dict, valid: set) -> set:
        return {
            key
            for key, words in aliases.items()
            if key in valid and tokens & set(words)
        }

    level_hits = match_field(LEVEL_ALIASES, levels)
    lateral_hits = match_field(LATERAL_ALIASES, laterals)
    mentions_receptacle = bool(tokens & set(RECEPTACLE_NOUNS))

    detail = {
        "levels_matched": sorted(level_hits),
        "laterals_matched": sorted(lateral_hits),
        "receptacle_noun": mentions_receptacle,
    }

    if len(level_hits) > 1:
        return None, detail, f"{phrase!r} names more than one level: {sorted(level_hits)}"
    if len(lateral_hits) > 1:
        return None, detail, f"{phrase!r} names more than one slot: {sorted(lateral_hits)}"
    if not level_hits and not lateral_hits:
        return None, detail, f"no shelf level or slot named in {phrase!r}"

    notes = []
    if not level_hits:
        return None, detail, f"{phrase!r} names a slot but not which level"
    level = level_hits.pop()

    if lateral_hits:
        lateral = lateral_hits.pop()
    else:
        lateral = DEFAULT_LATERAL
        notes.append(f"no slot named, defaulting to {DEFAULT_LATERAL!r}")

    for entity in receptacles:
        if entity.metadata.get("level") == level and entity.metadata.get("lateral") == lateral:
            note = f"matched {phrase!r} to {entity.label}"
            return entity, detail, "; ".join([note, *notes])
    return None, detail, f"the scene has no {level} shelf {lateral} slot"


def parse_task(prompt: str, scene: SceneGraph) -> TaskSpec:
    """Parse a task prompt against a scene.

    Args:
        prompt: e.g. ``"put the milk carton on the top shelf"``.
        scene: What is actually present.

    Returns:
        A :class:`TaskSpec`. Check ``ok`` before using the ids; on failure,
        ``error`` says what went wrong and ``candidates`` shows what was
        considered.
    """
    pick_phrase, place_phrase = split_prompt(prompt)
    spec = TaskSpec(ok=False, prompt=prompt, pick_phrase=pick_phrase, place_phrase=place_phrase)

    obj, object_scores, object_note = resolve_object(pick_phrase, scene)
    spec.candidates["objects"] = object_scores
    spec.rationale.append(object_note)
    if obj is None:
        spec.error = object_note
        return spec

    if not place_phrase:
        spec.error = (
            f"no destination in {prompt!r}; expected one of "
            f"{PLACEMENT_PREPOSITIONS[:4]} followed by a shelf"
        )
        spec.rationale.append(spec.error)
        return spec

    destination, receptacle_detail, receptacle_note = resolve_receptacle(place_phrase, scene)
    spec.candidates["receptacles"] = receptacle_detail
    spec.rationale.append(receptacle_note)
    if destination is None:
        spec.error = receptacle_note
        return spec

    spec.ok = True
    spec.object_id, spec.object_label = obj.entity_id, obj.label
    spec.destination_id, spec.destination_label = destination.entity_id, destination.label
    return spec
