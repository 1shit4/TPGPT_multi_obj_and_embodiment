"""A scene described as identified entities, for language to resolve against.

Everything downstream speaks **stable ids** (``obj_001``, ``slot_002``) rather
than free text. That is the discipline that keeps a parsing mistake visible: a
wrong id is inspectable and testable, whereas a label passed around as a string
quietly becomes whatever the last piece of code decided it meant.

The graph is built from simulator state. In a real deployment the same
structure would come from a detector; the entity records carry the same fields
either way, so the language and grasping layers do not know or care which.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SceneEntity:
    """One thing in the scene that a prompt could refer to."""

    entity_id: str
    label: str
    kind: str                               # "object" | "receptacle"
    position: np.ndarray
    aliases: tuple[str, ...] = ()
    #: robosuite instance name, for segmentation. Receptacles have none.
    instance: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.label,) + tuple(self.aliases)

    def describe(self) -> str:
        return f"{self.entity_id}  {self.kind:11s} {self.label:22s} {np.round(self.position, 3)}"


@dataclass
class SceneGraph:
    """The objects and destinations available in one scene configuration."""

    entities: list[SceneEntity] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self):
        return iter(self.entities)

    def by_id(self, entity_id: str) -> SceneEntity:
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        raise KeyError(f"unknown entity {entity_id!r}; scene has {[e.entity_id for e in self.entities]}")

    def of_kind(self, kind: str) -> list[SceneEntity]:
        return [e for e in self.entities if e.kind == kind]

    @property
    def objects(self) -> list[SceneEntity]:
        return self.of_kind("object")

    @property
    def receptacles(self) -> list[SceneEntity]:
        return self.of_kind("receptacle")

    def table(self) -> str:
        """Markdown-ish table of the scene, for logs and error messages."""
        lines = [f"{'id':10s} {'kind':11s} {'label':22s} {'position':22s} aliases"]
        for e in self.entities:
            lines.append(
                f"{e.entity_id:10s} {e.kind:11s} {e.label:22s} "
                f"{str(np.round(e.position, 3)):22s} {', '.join(e.aliases)}"
            )
        return "\n".join(lines)


def build_scene_graph(env, object_aliases=None, receptacle_aliases=None) -> SceneGraph:
    """Build a :class:`SceneGraph` from a :class:`~tpgpt.sim.scenes.tabletop_shelf.TabletopShelf`.

    Args:
        env: The environment to read.
        object_aliases: Optional ``{label: (alias, ...)}`` overriding the
            defaults from :mod:`tpgpt.language.vocabulary`.
        receptacle_aliases: As above, for shelf slots.
    """
    from tpgpt.language.vocabulary import OBJECT_ALIASES, receptacle_aliases_for

    object_aliases = object_aliases or OBJECT_ALIASES
    entities: list[SceneEntity] = []

    for index, name in enumerate(env.object_names, start=1):
        entities.append(
            SceneEntity(
                entity_id=f"obj_{index:03d}",
                label=name,
                kind="object",
                position=env.object_position(name),
                aliases=tuple(object_aliases.get(name, ())),
                instance=env.object_instance(name),
            )
        )

    for index, (slot, position) in enumerate(env.slot_poses().items(), start=1):
        level, lateral = slot.split("_", 1)
        aliases = (
            receptacle_aliases.get(slot, ())
            if receptacle_aliases
            else receptacle_aliases_for(level, lateral)
        )
        entities.append(
            SceneEntity(
                entity_id=f"slot_{index:03d}",
                label=f"{level} shelf {lateral}",
                kind="receptacle",
                position=position,
                aliases=tuple(aliases),
                metadata={"level": level, "lateral": lateral, "slot": slot},
            )
        )

    return SceneGraph(entities=entities)
