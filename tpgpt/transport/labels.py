"""Policy labels and their transportation (paper Sec. III-A, III-F, III-G).

The paper is explicit that a demonstration is *not* treated as a trajectory but
as a set of independent state-action pairs (Sec. III-F). That is what allows
labels which were interactively edited or aggregated from human feedback to be
generalised, and it is why the transported velocity is obtained through the
Jacobian rather than by differentiating the transported path.

``PolicyLabels`` therefore stores each label family from Sec. III-A as a plain
array indexed by label, plus the two book-keeping channels a real controller
needs: the gripper command and the belief-of-time phase used by the motor
primitive of ref. [12].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from tpgpt.transport.uncertainty import variance_to_std

_ARRAY_FIELDS = (
    "positions",
    "velocities",
    "orientations",
    "stiffness",
    "damping",
    "gripper",
    "time_belief",
    "position_std",
    "velocity_std",
)


@dataclass
class PolicyLabels:
    """A set of ``M`` policy labels in one task space.

    Attributes:
        positions: ``(M, 3)`` Cartesian positions -- the set ``X`` of Sec. III-A.
        velocities: ``(M, 3)`` Cartesian velocities ``X_dot``.
        orientations: ``(M, 3, 3)`` end-effector rotations ``R`` in ``SO(3)``.
        stiffness: ``(M, 3, 3)`` Cartesian stiffness ``K``, symmetric PSD.
        damping: ``(M, 3, 3)`` Cartesian damping ``D``, symmetric PSD.
        gripper: ``(M,)`` gripper command; ``+1`` closed, ``-1`` open.
        time_belief: ``(M,)`` monotone phase in ``[0, 1]``. Not part of the
            paper's label list, but Sec. V states the real-robot policy takes
            position *and* a belief of time as input (ref. [12]); a policy of
            position alone cannot represent a path that revisits a position.
        position_std: ``(M,)`` transport uncertainty on the positions.
        velocity_std: ``(M, 3)`` transport uncertainty on the velocities (Eq. 12).
        metadata: Free-form provenance (control frequency, scene, seed, ...).
    """

    positions: np.ndarray
    velocities: np.ndarray | None = None
    orientations: np.ndarray | None = None
    stiffness: np.ndarray | None = None
    damping: np.ndarray | None = None
    gripper: np.ndarray | None = None
    time_belief: np.ndarray | None = None
    position_std: np.ndarray | None = None
    velocity_std: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.positions = np.atleast_2d(np.asarray(self.positions, dtype=float))
        for name, shape in (
            ("velocities", (-1, 3)),
            ("orientations", (-1, 3, 3)),
            ("stiffness", (-1, 3, 3)),
            ("damping", (-1, 3, 3)),
            ("gripper", (-1,)),
            ("time_belief", (-1,)),
            ("position_std", (-1,)),
            ("velocity_std", (-1, 3)),
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=float).reshape(shape))
        self.validate()

    def __len__(self) -> int:
        return self.positions.shape[0]

    def validate(self) -> None:
        """Raise if any present label family disagrees on ``M``."""
        n = len(self)
        for name in _ARRAY_FIELDS:
            value = getattr(self, name)
            if value is not None and value.shape[0] != n:
                raise ValueError(
                    f"{name} has {value.shape[0]} labels but positions has {n}"
                )
        if self.positions.shape[1] != 3:
            raise ValueError(f"positions must be (M, 3), got {self.positions.shape}")

    def __getitem__(self, idx) -> "PolicyLabels":
        """Index or slice every present label family consistently."""
        if isinstance(idx, (int, np.integer)):
            idx = slice(idx, idx + 1)
        kwargs = {
            name: (v[idx] if (v := getattr(self, name)) is not None else None)
            for name in _ARRAY_FIELDS
        }
        return PolicyLabels(metadata=dict(self.metadata), **kwargs)

    @property
    def present(self) -> tuple[str, ...]:
        """Names of the label families that are actually populated."""
        return tuple(n for n in _ARRAY_FIELDS if getattr(self, n) is not None)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> Path:
        """Persist to a compressed ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {n: v for n in _ARRAY_FIELDS if (v := getattr(self, n)) is not None}
        np.savez_compressed(path, __metadata__=np.array(repr(self.metadata)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PolicyLabels":
        """Load labels written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            kwargs = {n: data[n] for n in _ARRAY_FIELDS if n in data}
            meta = {}
            if "__metadata__" in data:
                import ast

                try:
                    meta = ast.literal_eval(str(data["__metadata__"]))
                except (ValueError, SyntaxError):  # pragma: no cover - defensive
                    meta = {}
        return cls(metadata=meta, **kwargs)

    def describe(self) -> str:
        """One-line human-readable summary."""
        return f"PolicyLabels(M={len(self)}, families={list(self.present)})"


def transport_labels(transport_map, labels: PolicyLabels) -> PolicyLabels:
    """Transport a full label set through ``phi`` (paper Sec. III-F, III-G).

    Applies, for whichever families are present:

    * positions   -- ``X_hat = phi(X)``
    * velocities  -- Eq. (9) ``xdot_hat = J xdot``, with Eq. (12) uncertainty
    * orientations -- Eq. (11) ``R_hat = J_perp R``
    * stiffness / damping -- ``J_perp M J_perp^T`` (Sec. III-G)

    The gripper command and the time belief are scalars attached to the label,
    not spatial quantities, so they carry over unchanged.

    Args:
        transport_map: A fitted :class:`~tpgpt.transport.maps.TransportMap`.
        labels: Source-space labels.

    Returns:
        A new :class:`PolicyLabels` in the target space, carrying the transport
        uncertainties.
    """
    X = labels.positions
    X_hat, pos_std = transport_map.transport_positions(X, return_std=True)

    velocities = velocity_std = None
    if labels.velocities is not None:
        velocities, velocity_std = transport_map.transport_velocities(
            X, labels.velocities, return_std=True
        )

    orientations = (
        transport_map.transport_orientations(X, labels.orientations)
        if labels.orientations is not None
        else None
    )
    stiffness = (
        transport_map.transport_stiffness(X, labels.stiffness)
        if labels.stiffness is not None
        else None
    )
    damping = (
        transport_map.transport_damping(X, labels.damping)
        if labels.damping is not None
        else None
    )

    metadata = dict(labels.metadata)
    metadata["transported"] = True
    report = transport_map.check_diffeomorphism(X)
    metadata["jacobian_fraction_positive"] = report.fraction_positive
    metadata["jacobian_consistent_sign"] = report.consistent_sign
    metadata["keypoint_residual_max"] = float(transport_map.keypoint_residual().max())

    return PolicyLabels(
        positions=X_hat,
        velocities=velocities,
        orientations=orientations,
        stiffness=stiffness,
        damping=damping,
        gripper=None if labels.gripper is None else labels.gripper.copy(),
        time_belief=None if labels.time_belief is None else labels.time_belief.copy(),
        position_std=pos_std,
        velocity_std=velocity_std,
        metadata=metadata,
    )
