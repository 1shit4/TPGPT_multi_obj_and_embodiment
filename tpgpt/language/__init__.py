"""Deterministic natural-language task parsing (no model, no network)."""

from tpgpt.language.parser import TaskSpec, parse_task, resolve_object, resolve_receptacle
from tpgpt.language.vocabulary import OBJECT_ALIASES, receptacle_aliases_for

__all__ = [
    "OBJECT_ALIASES",
    "TaskSpec",
    "parse_task",
    "receptacle_aliases_for",
    "resolve_object",
    "resolve_receptacle",
]
