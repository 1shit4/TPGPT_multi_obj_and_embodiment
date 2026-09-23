"""Every ROS parameter a node reads must be one it declared.

A node declares its parameters once (``d = self.declare_parameter``) and reads
them through a one-line lambda (``p = lambda n: self.get_parameter(n).value``).
A typo in either is invisible until the node runs on the robot, where it raises
``ParameterNotDeclaredException`` in the middle of a plan. This reads the
sources instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "ycb_pointcloud_pipeline"
NODES = sorted(p for p in PACKAGE.glob("*.py") if p.name != "__init__.py")


def names(tree: ast.AST, func: str) -> set[str]:
    """Every ``func('literal')`` in the file, whatever it is called on."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        called = node.func
        called = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", None)
        if called == func and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            out.add(node.args[0].value)
    return out


def class_declarations() -> dict[str, set[str]]:
    """Every node class in the package, and the parameters its file declares.

    A node that extends another (``JointHoming(JointTargetPublisher)``) inherits
    its parameters, so the base's declarations count as its own.
    """
    per_class: dict[str, set[str]] = {}
    bases: dict[str, list[str]] = {}
    for path in NODES:
        tree = ast.parse(path.read_text())
        declared = names(tree, "declare_parameter") | names(tree, "d")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                per_class[node.name] = declared
                bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
    for name in per_class:
        seen, queue = set(), list(bases.get(name, []))
        while queue:
            base = queue.pop()
            if base in seen or base not in per_class:
                continue
            seen.add(base)
            per_class[name] = per_class[name] | per_class[base]
            queue.extend(bases.get(base, []))
    return per_class


INHERITED = class_declarations()


@pytest.mark.parametrize("path", NODES, ids=lambda p: p.name)
def test_parameters_read_are_parameters_declared(path):
    tree = ast.parse(path.read_text())
    declared = names(tree, "declare_parameter") | names(tree, "d")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            declared = declared | INHERITED.get(node.name, set())
    read = names(tree, "get_parameter") | names(tree, "p")
    # `p(...)` and `d(...)` are also ordinary function names in some files; only
    # the nodes that declare parameters are checked, and only for what they read.
    if not declared:
        pytest.skip(f"{path.name} declares no parameters")
    missing = sorted(read - declared)
    assert not missing, f"{path.name} reads undeclared parameters: {missing}"
