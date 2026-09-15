"""Attribute every cell of the real-world campaign to a cause, from its trace.

The ordering matters and is not arbitrary: each test is placed where it is
because the state it describes makes every later test meaningless. A hand that
knocked the object away before closing cannot be judged on whether its jaws
gripped; a hand whose jaws closed on air cannot be judged on whether it carried.

Thresholds are stated here rather than buried, because every one of them is a
judgement:

``PUSH``      20 mm of horizontal object motion *before* the jaws are commanded
              shut. Below that is settling and contact compliance; above it the
              object is somewhere other than where the grasp was planned on.
``ARRIVE``    40 mm between the hand and the planned grasp point at the moment
              of closing. The approach axis tolerates 120-135 mm on a parallel
              jaw, so this is deliberately loose -- it is meant to catch "the
              arm never got there", not "the arm was a few millimetres off".
``GRIP``      50 N of peak grip force after closing. Measured on the traces, a
              hand that takes hold registers hundreds of newtons and one that
              brushes the object registers 20-40.
``LIFT``      20 mm of rise. The demonstration lifts about 340 mm.
``HOLD``      60% of the steps between closing and release in contact.
"""
import json, sys, numpy as np, collections
from pathlib import Path

PUSH, ARRIVE, GRIP, LIFT, HOLD = 0.020, 0.040, 50.0, 0.020, 0.60
S = Path("/tmp/claude-1000/-home-ishita-TPGPT/c6962ad8-6825-422a-9fa3-2cbcf5da2ef5/scratchpad")
TR = S / "traces"

man = json.load(open("outputs/campaigns/realworld/real/manifest.json"))["runs"]
fit = {(o["gripper"],o["object"],o["pick_config_id"],o["destination"]): o
       for o in json.load(open(S/"fit.json"))}
slab = {(o["gripper"],o["object"],o["pick_config_id"],o["destination"]): o
        for o in json.load(open(S/"slab.json"))}

def classify(r):
    key=(r["gripper"],r["object"],r["pick_config_id"],r["destination"])
    tag=f"{r['gripper']}__{r['object']}__{r['pick_config_id']}__{r['destination'].replace(' ','_')}"
    f=TR/f"{tag}.npz"
    ev={"cell":"/".join(str(x) for x in key), "outcome":r["outcome"],
        "shelf_fits":fit[key]["fits"], "overflow_mm":round(max(fit[key][x] for x in ("front_mm","back_mm","side_mm","below_mm")),1),
        "slab_mm":None if slab[key]["slab_mm"] is None else round(slab[key]["slab_mm"],1),
        "aperture_mm":slab[key]["aperture_mm"]}
    if not f.exists():
        return {**ev,"cause":"no trace"}
    z=np.load(f)
    grip=z["grip"]; held=z["held"].astype(bool); force=z["grip_force"]
    ox,oy,oz=z["object_x"],z["object_y"],z["object_z"]; hand=z["hand"]
    G=z["grasp"]; n=len(grip)
    shut=np.flatnonzero(grip>0)
    if not len(shut):
        return {**ev,"cause":"jaws never commanded shut"}
    close, release = int(shut[0]), int(shut[-1])
    push=float(np.hypot(ox[close]-ox[0], oy[close]-oy[0]))
    arrive=float(np.linalg.norm(hand[close]-G[:3,3]))
    peak=float(force[close:].max()) if close<n else 0.0
    lift=float(oz.max()-oz[0])
    window=slice(close,max(release,close+1))
    hold=float(held[window].mean()) if release>close else 0.0
    touched=bool(held[close:min(n,close+20)].any())
    # **Did the hand ever get there, and was it there when the jaws shut?**
    # These are different questions and the difference is the whole of the
    # "never arrived" bucket. The executor's gripper schedule is open-loop on
    # the phase: it closes when the clock says to, not when the hand is in
    # position. If the arm is merely *late*, the closest approach happens after
    # the close and the jaws shut on the way in.
    dist = np.linalg.norm(hand - G[:3,3], axis=1)
    best = int(np.argmin(dist))
    # And of the miss at the close, how much is each half of the executor?
    #
    # **Not the plan.** Measured over all 210 cells, the transported path passes
    # 0.0 mm from both the planned grasp and the planned release -- exactly, by
    # construction, because the grasp-pose cube is centred on the grasp and phi
    # interpolates its keypoints exactly. So every millimetre of this miss is
    # downstream of the map, and it splits in two: how far the *integrated
    # attractor* has drifted from the plan it is meant to follow, and how far
    # the arm is trailing that attractor.
    att = z["attractor"]
    plan_miss = float(np.linalg.norm(att[close] - G[:3,3]))
    track_miss = float(np.linalg.norm(hand[close] - att[close]))
    ev.update(closest_mm=round(float(dist.min())*1000,1), closest_step=best,
              closed_early=bool(best > close and dist[close] > dist.min()*2),
              plan_miss_mm=round(plan_miss*1000,1), track_miss_mm=round(track_miss*1000,1))
    ev.update(push_mm=round(push*1000,1), arrive_mm=round(arrive*1000,1),
              peak_force_N=round(peak,1), lift_mm=round(lift*1000,1),
              hold_frac=round(hold,2), close_step=close, release_step=release,
              touched_at_close=touched,
              grasp_idx=int(z["meta"][0]), campaign_idx=int(z["meta"][1]))
    if r["outcome"]=="success":
        ev["cause"]="placed"; return ev
    if push>PUSH:            ev["cause"]="object knocked away before the jaws closed"
    elif arrive>ARRIVE:
        if ev["closest_mm"] <= ARRIVE*1000 and ev["closest_step"] > close:
            ev["cause"]="jaws closed early: the hand reached the grasp later"
        elif ev["plan_miss_mm"] > ev["track_miss_mm"]:
            ev["cause"]="the attractor drifted off the plan"
        else:
            ev["cause"]="the arm lagged too far behind its attractor"
    elif not touched and lift<LIFT: ev["cause"]="jaws closed with nothing between them"
    elif peak<GRIP and lift<LIFT:   ev["cause"]="fingers touched but took no purchase"
    elif lift<LIFT:          ev["cause"]="gripped but the object never rose"
    elif hold<HOLD:          ev["cause"]="lost the object during the carry"
    elif not fit[key]["fits"]: ev["cause"]="shelf refuses the commanded placed pose"
    else:                    ev["cause"]="carried, released in the wrong place"
    return ev

rows=[classify(r) for r in man]
json.dump(rows, open(S/"causes.json","w"), indent=1)
have=[r for r in rows if r["cause"]!="no trace"]
print(f"traces available for {len(have)} of {len(rows)} cells\n")
mism=[r for r in have if r.get("grasp_idx",-1)!=r.get("campaign_idx",-2)]
print(f"re-runs that chose a different grasp than the campaign: {len(mism)} of {len(have)}")
for m in mism[:8]: print("   ", m["cell"], m["grasp_idx"], "vs", m["campaign_idx"])
print()
c=collections.Counter(r["cause"] for r in have)
print("of the cells where the hand was far from the grasp when the jaws shut:")
far=[r for r in have if r.get("arrive_mm") is not None and r["arrive_mm"]>ARRIVE*1000]
if far:
    print(f"   n={len(far)}   median miss at close {np.median([r['arrive_mm'] for r in far]):.0f} mm")
    print(f"   of which the hand DID reach the grasp later: "
          f"{sum(1 for r in far if r['closest_mm']<=ARRIVE*1000 and r['closest_step']>r['close_step'])}")
    print(f"   median closest approach over the whole run: "
          f"{np.median([r['closest_mm'] for r in far]):.0f} mm")
    print(f"   attractor drift off the plan: median {np.median([r['plan_miss_mm'] for r in far]):.0f} mm")
    print(f"   arm lag behind the attractor: median {np.median([r['track_miss_mm'] for r in far]):.0f} mm")
    print(f"   (the transported plan itself passes 0.0 mm from the grasp on all 210 cells)")
print()
print(f"{'cause':46s} {'cells':>6s}")
for k,v in c.most_common(): print(f"{k:46s} {v:6d}")
print()
print("cause by object:")
objs=sorted({r["cell"].split("/")[1] for r in have})
causes=[k for k,_ in c.most_common()]
print(f"{'cause':46s} " + " ".join(f"{o[:7]:>7s}" for o in objs))
for cause in causes:
    print(f"{cause:46s} " + " ".join(
        f"{sum(1 for r in have if r['cause']==cause and r['cell'].split('/')[1]==o):7d}" for o in objs))
print()
print("cause by gripper:")
gs=sorted({r["cell"].split("/")[0] for r in have})
print(f"{'cause':46s} " + " ".join(f"{g[:9]:>9s}" for g in gs))
for cause in causes:
    print(f"{cause:46s} " + " ".join(
        f"{sum(1 for r in have if r['cause']==cause and r['cell'].split('/')[0]==g):9d}" for g in gs))
