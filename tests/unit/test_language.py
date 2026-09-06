"""Deterministic prompt parsing.

No model and no network, so these are exact-answer tests: the same prompt and
scene must always give the same ids.
"""

import numpy as np
import pytest

from tpgpt.language.parser import DEFAULT_LATERAL, content_tokens, parse_task, split_prompt
from tpgpt.language.vocabulary import OBJECT_ALIASES, receptacle_aliases_for
from tpgpt.perception.scene_graph import SceneEntity, SceneGraph


@pytest.fixture
def scene():
    """A scene graph matching the tabletop shelf, built without a simulator."""
    entities = [
        SceneEntity(f"obj_{i:03d}", label, "object", np.zeros(3),
                    aliases=OBJECT_ALIASES[label], instance=label)
        for i, label in enumerate(("milk", "can", "cereal", "bread"), start=1)
    ]
    index = 1
    for level in ("bottom", "top"):
        for lateral in ("left", "middle", "right"):
            entities.append(
                SceneEntity(
                    f"slot_{index:03d}", f"{level} shelf {lateral}", "receptacle",
                    np.zeros(3), aliases=receptacle_aliases_for(level, lateral),
                    metadata={"level": level, "lateral": lateral},
                )
            )
            index += 1
    return SceneGraph(entities=entities)


class TestSplitting:
    @pytest.mark.parametrize(
        "prompt,pick,place",
        [
            ("put the milk on the top shelf", "the milk", "the top shelf"),
            ("place the can onto the bottom shelf", "the can", "the bottom shelf"),
            ("move the cereal into the top slot", "the cereal", "the top slot"),
        ],
    )
    def test_leading_verb_is_stripped_and_phrase_split(self, prompt, pick, place):
        assert split_prompt(prompt) == (pick, place)

    def test_longest_preposition_wins(self):
        """'on top of' must not be split at the bare 'on'.

        Splitting at 'on' would leave 'top of ...' as the destination and, worse,
        strand 'top' where a naive matcher could read it as part of the object.
        """
        _, place = split_prompt("put the can on top of the bottom shelf")
        assert place == "the bottom shelf"

    def test_missing_destination_gives_an_empty_place_phrase(self):
        assert split_prompt("pick up the milk")[1] == ""

    def test_stopwords_are_dropped(self):
        assert content_tokens("the milk carton please") == ["milk", "carton"]


class TestObjectResolution:
    def test_canonical_name(self, scene):
        assert parse_task("put the milk on the top shelf", scene).object_id == "obj_001"

    @pytest.mark.parametrize(
        "phrase,expected",
        [("milk carton", "obj_001"), ("soda can", "obj_002"),
         ("cereal box", "obj_003"), ("loaf", "obj_004")],
    )
    def test_aliases_resolve(self, scene, phrase, expected):
        assert parse_task(f"put the {phrase} on the top shelf", scene).object_id == expected

    def test_unknown_object_fails_with_its_reason(self, scene):
        spec = parse_task("put the banana on the top shelf", scene)
        assert not spec.ok
        assert "banana" in spec.error
        assert spec.object_id is None

    def test_pronoun_does_not_silently_pick_something(self, scene):
        assert not parse_task("move it to the top shelf", scene).ok


class TestDestinationResolution:
    @pytest.mark.parametrize(
        "phrase,level,lateral",
        [
            ("the top shelf left", "top", "left"),
            ("the bottom shelf right", "bottom", "right"),
            ("the upper level right slot", "top", "right"),
            ("the lowest tier", "bottom", DEFAULT_LATERAL),
            ("the middle of the bottom rack", "bottom", "middle"),
        ],
    )
    def test_level_and_lateral_are_resolved_independently(self, scene, phrase, level, lateral):
        spec = parse_task(f"put the milk on {phrase}", scene)
        assert spec.ok, spec.error
        entity = scene.by_id(spec.destination_id)
        assert entity.metadata["level"] == level
        assert entity.metadata["lateral"] == lateral

    def test_a_level_without_a_slot_defaults_and_says_so(self, scene):
        """Underspecification is normal speech, so it resolves -- but visibly."""
        spec = parse_task("put the milk on the top shelf", scene)
        assert spec.ok
        assert scene.by_id(spec.destination_id).metadata["lateral"] == DEFAULT_LATERAL
        assert any("defaulting" in note for note in spec.rationale)

    def test_a_slot_without_a_level_is_refused(self, scene):
        """'the left slot' is genuinely ambiguous across two levels."""
        spec = parse_task("put the milk on the left slot", scene)
        assert not spec.ok
        assert "level" in spec.error

    def test_naming_two_levels_is_refused(self, scene):
        spec = parse_task("put the milk on the top bottom shelf", scene)
        assert not spec.ok
        assert "more than one level" in spec.error

    def test_missing_destination_is_reported(self, scene):
        spec = parse_task("put the milk somewhere", scene)
        assert not spec.ok
        assert "no destination" in spec.error

    def test_a_bare_shelf_is_refused(self, scene):
        assert not parse_task("put the milk on the shelf", scene).ok


class TestDeterminism:
    def test_the_same_prompt_always_parses_the_same_way(self, scene):
        prompt = "move the cereal box onto the upper level right slot"
        results = {
            (parse_task(prompt, scene).object_id, parse_task(prompt, scene).destination_id)
            for _ in range(5)
        }
        assert results == {("obj_003", "slot_006")}

    def test_case_and_punctuation_are_ignored(self, scene):
        plain = parse_task("put the milk on the top shelf", scene)
        noisy = parse_task("Please PUT the Milk, on the top shelf!", scene)
        assert (plain.object_id, plain.destination_id) == (noisy.object_id, noisy.destination_id)

    def test_failures_carry_the_candidates_considered(self, scene):
        spec = parse_task("put the banana on the top shelf", scene)
        assert "objects" in spec.candidates
        assert spec.rationale
