"""Names the parser accepts for the things in a scene.

Deliberately explicit rather than learned. Every accepted phrasing is written
down here, so a prompt that fails to parse fails visibly and the fix is to add
the word -- not to retrain or re-prompt anything. That is what makes the whole
pipeline reproducible offline.
"""

from __future__ import annotations

#: Alternative names for each spawnable object.
OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "milk": ("milk carton", "carton", "milk box"),
    "can": ("soda can", "tin", "coke can", "drink can"),
    "cereal": ("cereal box", "cornflakes", "breakfast box"),
    "bread": ("loaf", "bread loaf", "bun"),
    "bottle": ("water bottle", "flask"),
    "lemon": ("citrus", "fruit"),
    # Added with the fixed object set of ROBOTICS_NOTES 7.42. Two of these are
    # deliberately not boxes: a fleet measured only on cartons and a can cannot
    # show that more fingers buy anything.
    "hammer": ("mallet", "claw hammer"),
    "mug": ("cup", "hollow cylinder", "tumbler"),
    "pot": ("pan", "saucepan", "cooking pot"),
    "wrench": ("spanner", "ratcheting wrench"),
    "nut_square": ("square nut", "square peg"),
    "nut_round": ("round nut", "round peg"),
}

#: Words naming a shelf level.
LEVEL_ALIASES: dict[str, tuple[str, ...]] = {
    "top": ("top", "upper", "higher", "highest", "second"),
    "bottom": ("bottom", "lower", "low", "lowest", "first", "under"),
}

#: Words naming a lateral slot.
LATERAL_ALIASES: dict[str, tuple[str, ...]] = {
    "left": ("left", "leftmost", "left-hand"),
    "middle": ("middle", "centre", "center", "central", "middle-most"),
    "right": ("right", "rightmost", "right-hand"),
}

#: Nouns that mean "a shelf slot", so "top shelf" and "upper level" both work.
RECEPTACLE_NOUNS: tuple[str, ...] = (
    "shelf", "shelves", "level", "slot", "tier", "rack", "compartment", "ledge",
)

#: Prepositions that separate what to pick from where to put it.
#:
#: Order matters: the longest match is taken first, so "on top of" is not
#: mistaken for a bare "on" that leaves "top of" in the destination phrase.
PLACEMENT_PREPOSITIONS: tuple[str, ...] = (
    "on top of", "onto", "into", "in to", "on to", "on", "in", "to", "at", "inside",
)

#: Verbs that introduce the object, stripped before matching.
PICK_VERBS: tuple[str, ...] = (
    "put", "place", "move", "pick up", "pick", "take", "set", "bring", "transfer",
    "shift", "relocate", "stow", "shelve",
)

#: Words carrying no selective information.
STOPWORDS: frozenset[str] = frozenset(
    {"the", "a", "an", "please", "could", "you", "would", "kindly", "now", "then"}
)


def receptacle_aliases_for(level: str, lateral: str) -> tuple[str, ...]:
    """A few readable phrasings for one slot, for display and error messages.

    Receptacles are **not** matched by string comparison against this list.
    They have structure -- a level and a lateral position -- and the parser
    resolves those two fields independently
    (:func:`~tpgpt.language.parser.resolve_receptacle`). Enumerating every
    combination as a string would produce hundreds of aliases per slot, and
    "top shelf" would then match all three top slots equally, turning a
    perfectly ordinary instruction into an ambiguity error.
    """
    return (
        f"{level} shelf {lateral}",
        f"{lateral} slot of the {level} shelf",
        f"{level} level {lateral}",
    )
