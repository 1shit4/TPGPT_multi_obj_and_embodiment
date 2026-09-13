# Can this system hold an object with something other than a two-finger jaw?

**Written for a reader with no prior context.** Every experiment below states
what it asks, the conditions it was measured under, what each column means, the
result, and *why* the result came out that way. Section 1 is background; if you
know the setup, start at section 3.

**Provenance.** `manifest.json` in each output directory records the commit that
produced it. Where it reports `reproducible: false` the tree had uncommitted
changes and the numbers are a scratch experiment, not evidence — that rule is
`ROBOTICS_NOTES.md` §7.26, written after 254 MB of results had to be deleted for
exactly this reason.

**Companion documents.** `outputs/keypoints_sweep/FINDINGS.md` asks which
keypoints should pin the transportation map; `docs/dynamics_execution.md` asks
how a transported policy should be executed. This one asks a third question that
is independent of both: **which hands and which objects can this system grasp at
all?** Nothing here involves a transportation map, a policy, or a shelf.

---

> ### ⚠ Status of every experiment
>
> | # | experiment | status |
> |---|---|---|
> | A | §3 the ruler — what a one-actuator command mis-measures | *running* |
> | B | §4 the Inspire and UMI hands, re-measured and re-calibrated | *not started* |
> | C | §5 two hands paired from halves already on disk | *not started* |
> | D | §6 GraspGen-X descriptions authored from the MuJoCo model | *not started* |
> | E | §7 object census — what the cameras see, per object | *not started* |
> | F | §8 the bench — grasp, close, lift, carry | *not started* |
> | G | §9 the closure sweep, and how wide each pair's band is | *not started* |
> | H | §10 the closure table applied, paired grasp for grasp | *not started* |

---

## 1. Why this document exists

The project's cross-embodiment claim — one demonstration, many hands — rests on
a fleet that is almost entirely **two-fingered**. Nine hands are registered in
`tpgpt/grasp/grippers.py`; eight of them are two-finger jaws. The ninth,
`robotiq3f`, is the whole of the finger-count diversity in the project, and the
only other multi-finger pairing, `inspire`, carried a registry note reading
*"it does not actuate"* and was excluded from every campaign.

That is open item **1h** in `outputs/keypoints_sweep/FINDINGS.md` §8z, whose
stated reason is:

> Only one of the nine registered hands is not two-fingered, and no five-finger
> hand can be run at all. [...] robosuite offers Ability, Fourier, SchunkSvh and
> Jaco hands and **GraspGen-X has a description for none of them**, so there are
> no grasp candidates to filter.

A gripper needs two halves to be usable here. It needs a **robosuite model**, so
MuJoCo can simulate it, and it needs a **GraspGen-X description**, so the grasp
planner can propose poses for it. The assessment above is that the two sets
barely overlap. This document tests that assessment and then acts on it.

The second half of the problem is the objects. Four separate rules for closing
the jaws have now been tried and all four fail somewhere, because the window
between "gripping" and "crushing" is different for every combination of hand and
object and no constant sits inside all of them. That history is summarised in
§9, and it is why the closing rule here is **measured per pair** rather than
chosen.

*(Sections 2 onward are written as each experiment completes.)*
