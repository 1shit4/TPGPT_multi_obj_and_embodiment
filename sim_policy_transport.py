"""
Policy Transportation Simulation in MuJoCo + Robosuite
=======================================================
Replicates the full pipeline from the original ROS-based code:
  1. demo_recorder.py   -> DemoRecorder    : Record a demo trajectory in sim
  2. keypoint_detector.py -> KeypointGen  : Extract keypoints (cube corners) in sim
  3. transported_traj.py  -> transport_demo: AffineWarper + GPWarper
  4. run_new_traj.py      -> run_new_traj  : Execute transported trajectory

Demo task: Stack cubeA on top of cubeB
  - SOURCE scene: cubeA and cubeB at initial positions
  - TARGET scene: cubeA and cubeB shifted to a new location
  - The policy is recorded on SOURCE and transported to TARGET
"""

import os
import json
import copy
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import polar
from scipy.spatial.transform import Rotation as Rot

# No forced GL backend - let MuJoCo use the system display for live rendering
import robosuite as suite

# ── Version-agnostic controller loader ────────────────────────────────────
# robosuite >= 1.5  uses load_composite_controller_config('BASIC', robot=...)
# robosuite <= 1.4  uses load_controller_config(default_controller='OSC_POSE')
#                   and passes a plain dict
def _get_controller_config():
    try:
        from robosuite.controllers import load_composite_controller_config
        return load_composite_controller_config('BASIC', robot='Panda')
    except (ImportError, AssertionError, Exception):
        pass
    try:
        from robosuite import load_controller_config
        return load_controller_config(default_controller='OSC_POSE')
    except (ImportError, Exception):
        pass
    # Absolute fallback: raw dict that works on robosuite 1.2–1.4
    return {
        "type": "OSC_POSE",
        "input_max": 1, "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150, "damping_ratio": 1, "impedance_mode": "fixed",
        "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
        "position_limits": None, "orientation_limits": None,
        "uncouple_pos_ori": True, "control_delta": True,
    }

# Local imports (same-directory copies of the original files)
from affine_transform import AffineWarper
from warping_transform import GPWarper
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# ─────────────────────────── Configuration ────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DEMO_TRAJ_FILE        = os.path.join(DATA_DIR, "demo_trajectory.json")
SOURCE_KP_FILE        = os.path.join(DATA_DIR, "source_keypoints.json")
TARGET_KP_FILE        = os.path.join(DATA_DIR, "target_keypoints.json")
WARPED_TRAJ_FILE      = os.path.join(DATA_DIR, "warped_trajectory.json")
AFFINE_TRAJ_FILE      = os.path.join(DATA_DIR, "affine_trajectory.json")
SOURCE_AFFINE_KP_FILE = os.path.join(DATA_DIR, "source_affine_keypoints.json")
SOURCE_FINAL_KP_FILE  = os.path.join(DATA_DIR, "source_final_keypoints.json")

CONTROL_FREQ  = 20    # Hz for simulation control
CAMERA_H, CAMERA_W = 480, 640

# ── Rendering ─────────────────────────────────────────────────────────────
# Set RENDER_LIVE=True  to open the MuJoCo viewer window (requires a display)
# Set RENDER_LIVE=False to run headless (faster, saves frames to PNG instead)
RENDER_LIVE = True

# Cube half-size in the Stack env is ~0.02m. Corners are ±0.02 in x/y from center.
CUBE_HALF = 0.02


# ─────────────────────────── Helpers ──────────────────────────────────────

def make_env(source_pos=None, target_pos=None, offscreen=True):
    """
    Create robosuite Stack environment.
    Compatible with robosuite 1.2 through 1.5.
    """
    config = _get_controller_config()

    # Camera names vary by version; 'frontview' is always available
    cam_names = ['frontview', 'agentview'] if offscreen else []

    kwargs = dict(
        env_name='Stack',
        robots='Panda',
        controller_configs=config,
        has_renderer=RENDER_LIVE ,
        has_offscreen_renderer=offscreen,
        use_camera_obs=offscreen,
        camera_names=cam_names,
        camera_heights=CAMERA_H,
        camera_widths=CAMERA_W,
        control_freq=CONTROL_FREQ,
        reward_shaping=True,
    )
    # initialization_noise only exists in newer robosuite
    try:
        import inspect
        if 'initialization_noise' in str(inspect.signature(suite.make)):
            kwargs['initialization_noise'] = None
    except Exception:
        pass

    return suite.make(**kwargs)


def step_and_render(env, action):
    obs, rew, done, info = env.step(action)

    try:
        if RENDER_LIVE and getattr(env, "has_renderer", False):
            env.render()
    except Exception:
        pass

    return obs, rew, done, info


def get_cube_keypoints(cube_pos):
    """
    Simulate ArUco marker corners (like keypoint_detector.py).
    Returns 4 corners of the cube's top face as 3D points in robot base frame.
    Layout: TL, TR, BR, BL (looking from above)
    """
    cx, cy, cz = cube_pos
    top_z = cz + CUBE_HALF
    return np.array([
        [cx - CUBE_HALF, cy + CUBE_HALF, top_z],  # TL
        [cx + CUBE_HALF, cy + CUBE_HALF, top_z],  # TR
        [cx + CUBE_HALF, cy - CUBE_HALF, top_z],  # BR
        [cx - CUBE_HALF, cy - CUBE_HALF, top_z],  # BL
    ])


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  [Saved] {path}")


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def osc_move_to(env, target_pos, target_quat, current_obs, n_steps=80, gripper=-1.0):
    """
    Simple proportional controller: drive EEF to target_pos/quat using OSC delta actions.
    Returns final obs, list of (obs, action) tuples.
    """
    history = []
    obs = current_obs
    for _ in range(n_steps):
        eef_pos  = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']

        # Positional error
        pos_err = target_pos - eef_pos
        pos_action = np.clip(pos_err * 10.0, -1.0, 1.0)

        # Orientation error (using quat difference -> axis-angle)
        rot_curr = Rot.from_quat(eef_quat)
        rot_tgt  = Rot.from_quat(target_quat)
        rot_diff = rot_tgt * rot_curr.inv()
        rot_action = np.clip(rot_diff.as_rotvec() * 3.0, -1.0, 1.0)

        action = np.array([*pos_action, *rot_action, gripper])
        obs, rew, done, info = step_and_render(env, action)
        history.append((obs.copy(), action.copy()))

        if np.linalg.norm(pos_err) < 0.005:
            break
    return obs, history


# ──────────────────────────────────────────────────────────────────────────
# PHASE 1: Record Demo Trajectory  (like demo_recorder.py)
# ──────────────────────────────────────────────────────────────────────────

def record_demo(seed=42):
    """
    Records a pick-and-stack demo:
      - Pick cubeA from its start position
      - Place cubeA on top of cubeB
    Returns trajectory dict + source keypoints dict.
    """
    print("\n═══════════════════════════════════════")
    print("  PHASE 1: Recording Demo Trajectory")
    print("═══════════════════════════════════════")

    env = make_env(offscreen=False)
    obs = env.reset()
    # Seed the sim for reproducibility
    np.random.seed(seed)

    # Get initial cube positions
    cubeA_pos = obs['cubeA_pos'].copy()
    cubeB_pos = obs['cubeB_pos'].copy()
    print(f"  cubeA start: {cubeA_pos}")
    print(f"  cubeB start: {cubeB_pos}")

    # Default EEF orientation: gripper pointing down
    default_quat = np.array([1.0, 0.0, 0.0, 0.0])   # w,x,y,z in robosuite = x,y,z,w scipy

    timestamps  = []
    positions   = []
    orientations = []
    linear_vels  = []
    gripper_states = []
    frames = []

    t_start = time.time()
    prev_pos = None
    prev_t   = None

    def record_step(obs, gripper_held):
        nonlocal prev_pos, prev_t
        t_now = time.time() - t_start
        pos   = obs['robot0_eef_pos'].copy()
        quat  = obs['robot0_eef_quat'].copy()

        if prev_pos is not None and (t_now - prev_t) > 1e-4:
            vel = (pos - prev_pos) / (t_now - prev_t)
        else:
            vel = np.zeros(3)

        timestamps.append(t_now)
        positions.append(pos.tolist())
        orientations.append(quat.tolist())
        linear_vels.append(vel.tolist())
        gripper_states.append(1 if gripper_held else 0)
        prev_pos = pos.copy()
        prev_t   = t_now

        if offscreen_frames:
            frames.append(obs.get('frontview_image', None))

    offscreen_frames = True

    # ── Step 1: Move above cubeA ──
    print("  → Moving above cubeA...")
    above_A = cubeA_pos.copy()
    above_A[2] += 0.15  # hover height
    obs, hist = osc_move_to(env, above_A, default_quat, obs, n_steps=60, gripper=-1.0)
    for h_obs, _ in hist:
        record_step(h_obs, False)

    # ── Step 2: Descend to cubeA ──
    print("  → Descending to cubeA...")
    grasp_A = cubeA_pos.copy()
    grasp_A[2] += 0.005
    obs, hist = osc_move_to(env, grasp_A, default_quat, obs, n_steps=60, gripper=-1.0)
    for h_obs, _ in hist:
        record_step(h_obs, False)

    # ── Step 3: Close gripper ──
    print("  → Closing gripper (grasping cubeA)...")
    for _ in range(20):
        action = np.array([0, 0, 0, 0, 0, 0, 1.0])
        obs, _, _, _ = step_and_render(env, action)
        record_step(obs, True)

    # ── Step 4: Lift cubeA ──
    print("  → Lifting cubeA...")
    lift_pos = cubeA_pos.copy()
    lift_pos[2] += 0.20
    obs, hist = osc_move_to(env, lift_pos, default_quat, obs, n_steps=60, gripper=1.0)
    for h_obs, _ in hist:
        record_step(h_obs, True)

    # ── Step 5: Move above cubeB ──
    print("  → Moving above cubeB...")
    above_B = cubeB_pos.copy()
    above_B[2] += 0.22  # higher than cubeB top
    obs, hist = osc_move_to(env, above_B, default_quat, obs, n_steps=80, gripper=1.0)
    for h_obs, _ in hist:
        record_step(h_obs, True)

    # ── Step 6: Lower onto cubeB ──
    print("  → Lowering onto cubeB...")
    place_B = cubeB_pos.copy()
    place_B[2] += 0.065  # just above cubeB top
    obs, hist = osc_move_to(env, place_B, default_quat, obs, n_steps=60, gripper=1.0)
    for h_obs, _ in hist:
        record_step(h_obs, True)

    # ── Step 7: Open gripper ──
    print("  → Opening gripper (releasing cubeA)...")
    for _ in range(20):
        action = np.array([0, 0, 0, 0, 0, 0, -1.0])
        obs, _, _, _ = step_and_render(env, action)
        record_step(obs, False)

    # ── Step 8: Retreat ──
    print("  → Retreating...")
    retreat = place_B.copy()
    retreat[2] += 0.18
    obs, hist = osc_move_to(env, retreat, default_quat, obs, n_steps=40, gripper=-1.0)
    for h_obs, _ in hist:
        record_step(h_obs, False)

    env.close()

    # Normalise timestamps
    t0 = timestamps[0]
    norm_ts = [t - t0 for t in timestamps]

    traj_data = {
        "metadata": {"freq": CONTROL_FREQ, "source": "mujoco_sim_demo"},
        "timestamps": norm_ts,
        "positions": positions,
        "orientations": orientations,
        "velocities_lin": linear_vels,
        "gripper_states": gripper_states,
    }
    save_json(traj_data, DEMO_TRAJ_FILE)

    # Build source keypoints (both cubes, 4 corners each, like keypoint_detector.py)
    kp_A = get_cube_keypoints(cubeA_pos)
    kp_B = get_cube_keypoints(cubeB_pos)
    all_kp = np.vstack([kp_A, kp_B])
    labels = ["CubeA_TL","CubeA_TR","CubeA_BR","CubeA_BL",
              "CubeB_TL","CubeB_TR","CubeB_BR","CubeB_BL"]
    src_kp_data = {
        "type": "source",
        "order_reference": labels,
        "keypoints": [{"label": labels[i], "coords": all_kp[i].tolist()} for i in range(8)],
        "frame": "robot_base",
    }
    save_json(src_kp_data, SOURCE_KP_FILE)
    print(f"  Recorded {len(positions)} trajectory points.")
    return traj_data, src_kp_data, cubeA_pos, cubeB_pos, frames


# ──────────────────────────────────────────────────────────────────────────
# PHASE 2: Generate Target Keypoints  (like keypoint_detector.py)
# ──────────────────────────────────────────────────────────────────────────

def generate_target_keypoints(cubeA_pos_src, cubeB_pos_src):
    """
    Simulate placing cubes at a new (target) location.
    We shift both cubes by an offset to emulate 'object transported to new shelf'.
    """
    print("\n═══════════════════════════════════════")
    print("  PHASE 2: Generating Target Keypoints")
    print("═══════════════════════════════════════")

    # Shift: translate +0.15 in x, +0.10 in y, small rotation via different cubeB placement
    shift = np.array([0.15, 0.10, 0.0])
    cubeA_tgt = cubeA_pos_src + shift
    cubeB_tgt = cubeB_pos_src + shift + np.array([0.01, -0.01, 0.0])  # slight extra offset for cubeB

    print(f"  Source cubeA: {cubeA_pos_src}  -> Target: {cubeA_tgt}")
    print(f"  Source cubeB: {cubeB_pos_src}  -> Target: {cubeB_tgt}")

    kp_A = get_cube_keypoints(cubeA_tgt)
    kp_B = get_cube_keypoints(cubeB_tgt)
    all_kp = np.vstack([kp_A, kp_B])
    labels = ["CubeA_TL","CubeA_TR","CubeA_BR","CubeA_BL",
              "CubeB_TL","CubeB_TR","CubeB_BR","CubeB_BL"]
    tgt_kp_data = {
        "type": "target",
        "order_reference": labels,
        "keypoints": [{"label": labels[i], "coords": all_kp[i].tolist()} for i in range(8)],
        "frame": "robot_base",
    }
    save_json(tgt_kp_data, TARGET_KP_FILE)
    return tgt_kp_data, cubeA_tgt, cubeB_tgt


# ──────────────────────────────────────────────────────────────────────────
# PHASE 3: Policy Transportation  (like transported_traj.py)
# ──────────────────────────────────────────────────────────────────────────

def get_ordered_keypoints(data_dict):
    kp_map = {kp['label']: kp['coords'] for kp in data_dict['keypoints']}
    return np.array([kp_map[label] for label in data_dict['order_reference']])


def transport_policy():
    print("\n═══════════════════════════════════════")
    print("  PHASE 3: Policy Transportation")
    print("═══════════════════════════════════════")

    src_data  = load_json(SOURCE_KP_FILE)
    tgt_data  = load_json(TARGET_KP_FILE)
    demo_traj = load_json(DEMO_TRAJ_FILE)

    S = get_ordered_keypoints(src_data)
    T = get_ordered_keypoints(tgt_data)

    X_demo = np.array(demo_traj['positions'])
    Q_demo = np.array(demo_traj['orientations'])
    V_demo = np.array(demo_traj.get('velocities_lin', np.zeros_like(X_demo)))

    print(f"  Loaded {len(X_demo)} demo points. Fitting Affine...")

    # ── Affine Warp ──
    affine = AffineWarper()
    affine.fit(S, T)
    S_affine = affine.predict(S)
    X_affine = affine.predict(X_demo)
    R_aff    = affine.get_jacobian()
    V_affine = np.dot(V_demo, R_aff.T)

    print("  Fitting GP Warper...")

    # ── GP Warp ──
    kernel = C(1.0) * RBF(length_scale=0.2) + WhiteKernel(noise_level=1e-5)
    gp = GPWarper(kernel=kernel, n_restarts_optimizer=5)
    gp.fit(S_affine, T)

    # Verify keypoints
    S_final = np.array([gp.predict(S_affine[i]) for i in range(len(S_affine))])

    # ── Transport trajectory ──
    print("  Transporting trajectory points...")
    X_final, V_final, Q_final = [], [], []

    for i in range(len(X_affine)):
        x_curr = X_affine[i]
        v_curr = V_affine[i]
        q_curr = Q_demo[i]

        x_new = gp.predict(x_curr)
        X_final.append(x_new.tolist())

        J_gp  = gp.get_jacobian(x_curr)
        v_new = np.dot(J_gp, v_curr)
        V_final.append(v_new.tolist())

        R_warp, _ = polar(J_gp)
        # Use from_matrix (newer scipy) or from_dcm (older)
        try:
            rot_warp = Rot.from_matrix(R_warp)
        except AttributeError:
            rot_warp = Rot.from_dcm(R_warp)
        rot_orig = Rot.from_quat(q_curr)
        q_new = (rot_warp * rot_orig).as_quat()
        Q_final.append(q_new.tolist())

    # Save
    affine_traj = copy.deepcopy(demo_traj)
    warped_traj = copy.deepcopy(demo_traj)

    affine_traj['positions']      = X_affine.tolist()
    affine_traj['velocities_lin'] = V_affine.tolist()
    affine_traj['orientations']   = Q_demo.tolist()
    affine_traj['metadata']['source'] = "policy_transportation_affine"

    warped_traj['positions']      = X_final
    warped_traj['velocities_lin'] = V_final
    warped_traj['orientations']   = Q_final
    warped_traj['metadata']['source'] = "policy_transportation_gp"

    save_json(warped_traj, WARPED_TRAJ_FILE)
    save_json(affine_traj, AFFINE_TRAJ_FILE)
    save_json(S_affine.tolist(), SOURCE_AFFINE_KP_FILE)
    save_json(S_final.tolist(), SOURCE_FINAL_KP_FILE)

    print(f"  Transport complete. {len(X_final)} warped points.")
    return warped_traj, affine_traj


# ──────────────────────────────────────────────────────────────────────────
# PHASE 4: Execute Transported Trajectory  (like run_new_traj.py)
# ──────────────────────────────────────────────────────────────────────────

def set_cube_positions(env, cubeA_pos, cubeB_pos):
    """
    Force-set cube positions in the MuJoCo simulation via qpos manipulation.
    Compatible with both old mujoco-py and new mujoco bindings.
    """
    # Support both mujoco-py (env.sim) and new dm-control style (env.sim)
    sim = env.sim

    def find_joint_qpos_addr(sim, body_name):
        """Find qpos address for the free joint of a body.
        Works with both mujoco-py (robosuite<=1.4) and mujoco>=2.x (robosuite>=1.5).
        """
        # Try common joint name patterns
        for jnt_name in [f'{body_name}_joint0', f'{body_name}_freejoint',
                         f'{body_name}:joint', body_name]:
            try:
                jnt_id = sim.model.joint_name2id(jnt_name)
                return sim.model.jnt_qposadr[jnt_id]
            except Exception:
                pass
        # mujoco-py style: model.name2id('joint', name)
        for jnt_name in [f'{body_name}_joint0', f'{body_name}_freejoint']:
            try:
                jnt_id = sim.model.name2id(jnt_name, 'joint')
                return sim.model.jnt_qposadr[jnt_id]
            except Exception:
                pass
        # Last resort: find via body -> first joint
        try:
            body_id = sim.model.body_name2id(body_name)
            jnt_id  = sim.model.body_jntadr[body_id]
            if jnt_id >= 0:
                return sim.model.jnt_qposadr[jnt_id]
        except Exception:
            pass
        raise RuntimeError(f"Cannot find free joint for body '{body_name}'")

    addrA = find_joint_qpos_addr(sim, 'cubeA')
    addrB = find_joint_qpos_addr(sim, 'cubeB')

    # pos (3) + quat wxyz (4)
    sim.data.qpos[addrA:addrA+3] = cubeA_pos
    sim.data.qpos[addrA+3:addrA+7] = [1, 0, 0, 0]
    sim.data.qpos[addrB:addrB+3] = cubeB_pos
    sim.data.qpos[addrB+3:addrB+7] = [1, 0, 0, 0]
    sim.forward()


def run_transported_trajectory(cubeA_tgt, cubeB_tgt):
    """
    Execute the warped trajectory in a new sim scene where cubes are at TARGET positions.
    Like run_new_traj.py but in simulation.
    """
    print("\n═══════════════════════════════════════")
    print("  PHASE 4: Executing Transported Trajectory")
    print("═══════════════════════════════════════")

    warped_traj = load_json(WARPED_TRAJ_FILE)
    positions    = warped_traj['positions']
    orientations = warped_traj['orientations']
    gripper_states = warped_traj['gripper_states']

    print(f"  Replaying {len(positions)} warped trajectory points...")

    env = make_env(offscreen=True)
    obs = env.reset()

    # Override cube positions to target scene
    set_cube_positions(env, cubeA_tgt, cubeB_tgt)
    # _get_observations() exists in newer robosuite; older uses _get_observation()
    try:
        obs = env._get_observations()
    except AttributeError:
        try:
            obs = env._get_observation()
        except AttributeError:
            obs = env.reset()  # last resort

    frames_new = []
    rewards = []
    prev_gripper = 0

    # We execute each waypoint using the proportional controller
    # (matching the approach of demo_validator / run_new_traj)
    for i in range(len(positions)):
        tgt_pos  = np.array(positions[i])
        
        # print(gripper_states[i+1], gripper_states[i])
        if i+1<len(positions):
            if gripper_states[i+1]==1 and gripper_states[i]==0:
                print(positions[i+1])
                positions[i+1][-1] = positions[i+1][-1] - 2 #why?
                print(positions[i+1])
        tgt_quat = np.array(orientations[i])
        curr_grip = gripper_states[i]

        gripper_cmd = 1.0 if curr_grip == 1 else -1.0

        # Single step proportional move towards waypoint
        eef_pos  = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']

        pos_err = tgt_pos - eef_pos
        pos_action = np.clip(pos_err * 8.0, -1.0, 1.0)

        rot_curr = Rot.from_quat(eef_quat)
        rot_tgt  = Rot.from_quat(tgt_quat)
        rot_diff = (rot_tgt * rot_curr.inv()).as_rotvec()
        rot_action = np.clip(rot_diff * 3.0, -1.0, 1.0)

        action = np.array([*pos_action, *rot_action, gripper_cmd])
        obs, rew, done, info = step_and_render(env, action)
        rewards.append(rew)

        if obs.get('frontview_image') is not None:
            frames_new.append(obs['frontview_image'].copy())

    total_reward = sum(rewards)
    print(f"  Execution complete. Total reward: {total_reward:.3f}")
    env.close()
    return frames_new, rewards


# ──────────────────────────────────────────────────────────────────────────
# PHASE 5: Visualization  (like traj_visualizer.py)
# ──────────────────────────────────────────────────────────────────────────

def visualize_trajectories(frames_demo, frames_transported):
    print("\n═══════════════════════════════════════")
    print("  PHASE 5: Visualizing Results")
    print("═══════════════════════════════════════")

    demo_traj      = load_json(DEMO_TRAJ_FILE)
    warped_traj    = load_json(WARPED_TRAJ_FILE)
    affine_traj    = load_json(AFFINE_TRAJ_FILE)
    src_kp_data    = load_json(SOURCE_KP_FILE)
    tgt_kp_data    = load_json(TARGET_KP_FILE)
    s_affine_raw   = load_json(SOURCE_AFFINE_KP_FILE)
    s_final_raw    = load_json(SOURCE_FINAL_KP_FILE)

    demo_pos   = np.array(demo_traj['positions'])
    warp_pos   = np.array(warped_traj['positions'])
    affine_pos = np.array(affine_traj['positions'])
    S_kp       = get_ordered_keypoints(src_kp_data)
    T_kp       = get_ordered_keypoints(tgt_kp_data)
    S_affine   = np.array(s_affine_raw)
    S_final    = np.array(s_final_raw)

    # ── 3D trajectory comparison plot ──
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('#0d1117')

    # Plot 1: Demo vs Affine
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('#0d1117')
    ax1.plot(*demo_pos.T, color='#58a6ff', lw=2, label='Demo (Source)', alpha=0.9)
    ax1.plot(*affine_pos.T, color='#f78166', lw=2, label='Affine Warped', alpha=0.9)
    ax1.scatter(*S_kp.T, c='#58a6ff', marker='o', s=60, edgecolors='white', lw=0.5, label='Source KP')
    ax1.scatter(*T_kp.T, c='#f78166', marker='^', s=80, edgecolors='white', lw=0.5, label='Target KP')
    ax1.scatter(*S_affine.T, c='#ffa657', marker='s', s=60, edgecolors='white', lw=0.5, label='Affine KP')
    ax1.set_title('Step 1: Affine Alignment', color='white', fontsize=11, pad=10)
    _style_3d_ax(ax1)
    ax1.legend(fontsize=7, loc='upper left', facecolor='#161b22', labelcolor='white', framealpha=0.8)

    # Plot 2: Affine vs GP-Warped
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('#0d1117')
    ax2.plot(*affine_pos.T, color='#ffa657', lw=1.5, label='Affine', alpha=0.6)
    ax2.plot(*warp_pos.T, color='#3fb950', lw=2.5, label='GP Warped (Final)', alpha=0.9)
    ax2.scatter(*S_affine.T, c='#ffa657', marker='s', s=60, edgecolors='white', lw=0.5, label='Affine KP')
    ax2.scatter(*T_kp.T, c='#f78166', marker='^', s=80, edgecolors='white', lw=0.5, label='Target KP')
    ax2.scatter(*S_final.T, c='#3fb950', marker='d', s=60, edgecolors='white', lw=0.5, label='GP Final KP')
    ax2.set_title('Step 2: GP Residual Warping', color='white', fontsize=11, pad=10)
    _style_3d_ax(ax2)
    ax2.legend(fontsize=7, loc='upper left', facecolor='#161b22', labelcolor='white', framealpha=0.8)

    plt.suptitle("Policy Transportation: Demo → Transported Trajectory",
                 color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    traj_plot_path = os.path.join(DATA_DIR, "trajectory_comparison.png")
    plt.savefig(traj_plot_path, dpi=130, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  [Saved] {traj_plot_path}")

    # ── Save frame collages ──
    _save_frame_collage(frames_demo, os.path.join(DATA_DIR, "frames_demo.png"),
                        title="Demo Execution (Source Scene)")
    _save_frame_collage(frames_transported, os.path.join(DATA_DIR, "frames_transported.png"),
                        title="Transported Policy Execution (Target Scene)")

    # ── Keypoint error analysis ──
    _plot_keypoint_errors(S_kp, T_kp, S_affine, S_final)


def _style_3d_ax(ax):
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('#30363d')
        axis._axinfo['grid']['color'] = '#21262d'
    ax.tick_params(colors='#8b949e', labelsize=7)
    ax.set_xlabel('X (m)', color='#8b949e', fontsize=8)
    ax.set_ylabel('Y (m)', color='#8b949e', fontsize=8)
    ax.set_zlabel('Z (m)', color='#8b949e', fontsize=8)


def _save_frame_collage(frames, path, title=""):
    if not frames:
        print(f"  [Warning] No frames to save for: {title}")
        return
    # Pick ~8 evenly spaced frames
    n = min(8, len(frames))
    indices = np.linspace(0, len(frames)-1, n, dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(title, color='white', fontsize=12, y=1.02)
    for idx, ax in zip(indices, axes.flat):
        frame = frames[idx]
        if frame is not None:
            ax.imshow(frame[::-1])  # robosuite images are upside down
        ax.axis('off')
        ax.set_title(f"t={idx}", color='#8b949e', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  [Saved] {path}")


def _plot_keypoint_errors(S_kp, T_kp, S_affine, S_final):
    affine_errs = np.linalg.norm(S_affine - T_kp, axis=1)
    final_errs  = np.linalg.norm(S_final - T_kp, axis=1)
    labels = ["A_TL","A_TR","A_BR","A_BL","B_TL","B_TR","B_BR","B_BL"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.bar(x - 0.2, affine_errs*1000, 0.38, label='Affine only', color='#ffa657', alpha=0.85)
    ax.bar(x + 0.2, final_errs*1000,  0.38, label='Affine + GP',  color='#3fb950', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='#8b949e', fontsize=9)
    ax.set_ylabel('Error (mm)', color='#8b949e')
    ax.set_title('Keypoint Alignment Error After Transportation', color='white', fontsize=11)
    ax.tick_params(colors='#8b949e')
    ax.legend(facecolor='#161b22', labelcolor='white')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#21262d', linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(DATA_DIR, "keypoint_errors.png")
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  [Saved] {path}")
    print(f"\n  Mean Affine error: {affine_errs.mean()*1000:.2f} mm")
    print(f"  Mean GP error:     {final_errs.mean()*1000:.2f} mm")
    print(f"  Improvement:       {(1 - final_errs.mean()/affine_errs.mean())*100:.1f}%")


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║    Policy Transportation in MuJoCo + Robosuite                ║")
    print("║    (AffineWarper + GPWarper pipeline, no ROS required)        ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    t0 = time.time()

    # Phase 1: Record demo in source scene
    _, _, cubeA_src, cubeB_src, frames_demo = record_demo(seed=42)

    # Phase 2: Generate target keypoints (shifted scene)
    _, cubeA_tgt, cubeB_tgt = generate_target_keypoints(cubeA_src, cubeB_src)

    # Phase 3: Transport policy (AffineWarper + GPWarper)
    transport_policy()

    # Phase 4: Execute transported trajectory in target scene
    frames_transported, rewards = run_transported_trajectory(cubeA_tgt, cubeB_tgt)

    # Phase 5: Visualize everything
    visualize_trajectories(frames_demo, frames_transported)

    elapsed = time.time() - t0
    print(f"\n╔═══════════════════════════════════════════════════════════════╗")
    print(f"║  Pipeline complete in {elapsed:.1f}s                              ")
    print(f"║  Outputs in: {DATA_DIR}")
    print(f"╚═══════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()