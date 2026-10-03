"""X6 hardware experiment: configuration, paths, hashing and provenance.

No ROS here, so the session tool, the order sheet and the analysis can use it
without a robot. Everything the X6 tools write goes under

    <experiment.data_root>/<experiment.run_id>/

laid out as described in x6_experiment/experiments.md.

Provenance is **read-only**: only `git rev-parse`, `git status` and
`git ls-files` are run. Nothing is committed, staged or stashed.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml

TO_FILL = 'TO_FILL'

#: File suffixes that count as code when listing untracked files.
CODE_SUFFIXES = ('.py', '.yaml', '.yml', '.sh', '.xml', '.cfg', '.toml', '.json', '.srv', '.msg')
#: Path fragments never counted as code (data, caches, build output).
NON_CODE_PARTS = ('outputs/', '__pycache__', '.egg-info', 'build/', 'install/', 'log/',
                  '.pytest_cache', 'imgs/', 'assets/')

#: The only parameters a user may give on the command line of an experiment
#: trial. Everything else comes from the frozen config.
TRIAL_CLI_PARAMETERS = ('experiment', 'trial_id', 'hand', 'object_name', 'destination_name',
                        'dry_run', 'top_down_grasp', 'x6_config')

FAILURE_STAGES = ('planning', 'approach', 'reach', 'grasp', 'lift', 'carry', 'place', 'settle')

#: How the frozen ``method:`` block reaches the planner node: (block, key) -> planner
#: parameter. Every other planner parameter runs at its node default, as it does outside
#: an experiment. test_x6 checks each value (and type) against the node's own default, so
#: a default changed in the node without re-syncing the config fails there, not on the
#: robot.
PLANNER_METHOD_PARAMETERS = {
    ('perception', 'n_frames'): 'n_frames',
    ('perception', 'max_cloud_age_s'): 'max_cloud_age',
    ('perception', 'workspace_min'): 'workspace_min',
    ('perception', 'workspace_max'): 'workspace_max',
    ('perception', 'voxel_size'): 'voxel_size',
    ('perception', 'outlier_radius'): 'outlier_radius',
    ('perception', 'outlier_min_neighbors'): 'outlier_min_neighbors',
    ('perception', 'max_points'): 'max_points',
    ('perception', 'object_model'): 'object_model',
    ('perception', 'object_model_retries'): 'object_model_retries',
    ('perception', 'use_scene_cloud'): 'use_scene_cloud',
    ('grasp_generation', 'source'): 'grasp_source',
    ('grasp_generation', 'host'): 'graspgen_host',
    ('grasp_generation', 'port'): 'graspgen_port',
    ('grasp_generation', 'planner'): 'graspgen_planner',
    ('grasp_generation', 'grasp_library'): 'grasp_library',
    ('grasp_generation', 'grasp_library_size'): 'grasp_library_size',
    ('grasp_generation', 'num_grasps'): 'graspgen_num_grasps',
    ('grasp_generation', 'topk'): 'graspgen_topk',
    ('grasp_generation', 'timeout_s'): 'graspgen_timeout',
    ('selection', 'max_tilt_deg'): 'grasp_max_tilt_deg',
    ('selection', 'check_visibility'): 'check_visibility',
    ('selection', 'check_centred'): 'check_centred',
    ('selection', 'max_plans'): 'grasp_max_plans',
    ('selection', 'min_path_clearance_m'): 'grasp_min_path_clearance',
    ('selection', 'fallback_top_down'): 'grasp_fallback_top_down',
    ('selection', 'grasp_depth_below_top_m'): 'grasp_depth_below_top',
    ('selection', 'kinematic_criterion'): 'kinematic_criterion',
    ('selection', 'approach_rejects_grasp'): 'approach_rejects_grasp',
    ('selection', 'reach_margin_m'): 'grasp_reach_margin',
    ('selection', 'retreat_clearance_m'): 'grasp_retreat_clearance',
    ('map', 'min_determinant'): 'min_determinant',
}


def planner_method_parameters(method):
    """The planner parameters the frozen method block sets, as ``name -> value``."""
    out = {}
    for (block, key), name in PLANNER_METHOD_PARAMETERS.items():
        value = method[block][key]
        out[name] = [float(v) for v in value] if isinstance(value, list) else value
    return out


def default_config_path():
    """x6_experiment/config/x6_experiment.yaml in the source tree."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / 'x6_experiment' / 'config' / 'x6_experiment.yaml'
        if candidate.is_file():
            return str(candidate)
    # installed copy: ~/thor_ws/install/... -> look in the workspace source tree
    fallback = Path.home() / 'thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment/x6_experiment/config/x6_experiment.yaml'
    return str(fallback)


def expand(path):
    return os.path.abspath(os.path.expanduser(str(path))) if path not in (None, '') else path


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def is_unfilled(value):
    """True for TO_FILL, a string containing TO_FILL, or None."""
    if value is None:
        return True
    if isinstance(value, str):
        return TO_FILL in value
    if isinstance(value, (list, tuple)):
        return any(is_unfilled(v) for v in value)
    if isinstance(value, dict):
        return any(is_unfilled(v) for v in value.values())
    return False


class X6Config:
    """The campaign config plus the hand files it names."""

    def __init__(self, path=None):
        self.path = expand(path or default_config_path())
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f'X6 config not found: {self.path}')
        with open(self.path) as f:
            self.raw = yaml.safe_load(f)
        self.sha256 = sha256_file(self.path)
        self.dir = os.path.dirname(self.path)

    # ------------------------------------------------------------------ sections
    def __getitem__(self, key):
        return self.raw[key]

    @property
    def run_id(self):
        return self.raw['experiment']['run_id']

    @property
    def root(self):
        return os.path.join(expand(self.raw['experiment']['data_root']), self.run_id)

    @property
    def control_rate(self):
        return float(self.raw['experiment']['control_rate_hz'])

    @property
    def method(self):
        return self.raw['method']

    def hand_path(self, hand):
        files = self.raw['hands']['files']
        if hand not in files:
            raise KeyError(f'hand {hand!r} is not in the config (hands.files: {list(files)})')
        return os.path.join(self.dir, files[hand])

    def hand(self, hand):
        path = self.hand_path(hand)
        with open(path) as f:
            data = yaml.safe_load(f)['hand']
        data['_path'] = path
        data['_sha256'] = sha256_file(path)
        return data

    @property
    def shelf_file(self):
        return expand(self.raw['destinations']['shelf_file'])

    def destination_names(self):
        return [str(n) for n in self.raw['destinations']['slots']]

    def destination(self, name):
        """``{'x', 'y', 'surface_height'}`` of a named slot, from the touched slot marks in
        the shelf file (tpgpt.hardware.shelf.slots), the same resolution the planner uses."""
        if name not in self.destination_names():
            raise KeyError(f'destination {name!r} not in config (slots: '
                           f'{self.destination_names()})')
        from tpgpt.hardware.shelf import slots
        with open(self.shelf_file) as f:
            x, y, z = slots(yaml.safe_load(f))[name]
        return {'x': x, 'y': y, 'surface_height': z}

    # ------------------------------------------------------------------ paths
    def trial_dir(self, trial_id, scratch=False):
        sub = 'scratch' if scratch else 'trials'
        return os.path.join(self.root, sub, f'trial_{trial_id}')

    def sessions_dir(self):
        return os.path.join(self.root, 'sessions')

    def demo_copy_dir(self):
        return os.path.join(self.root, 'demonstration')


# ---------------------------------------------------------------------- demo hash
def demo_hashes(source_bundle_dir):
    """sha256 of every file in a source bundle, by file name."""
    d = expand(source_bundle_dir)
    return {name: sha256_file(os.path.join(d, name)) for name in sorted(os.listdir(d))
            if os.path.isfile(os.path.join(d, name))}


# ---------------------------------------------------------------------- provenance
def _git(repo, *args):
    try:
        out = subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True,
                             timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _is_code(path):
    return path.endswith(CODE_SUFFIXES) and not any(part in path for part in NON_CODE_PARTS)


def repo_provenance(repo):
    """SHA, modified tracked files and untracked code files of one repository."""
    repo = expand(repo)
    info = {'repo': repo, 'git_sha': None, 'branch': None, 'dirty': None,
            'modified_files': [], 'untracked_code_files': [], 'error': None}
    if not os.path.isdir(repo):
        info['error'] = 'directory does not exist'
        return info
    top = _git(repo, 'rev-parse', '--show-toplevel')
    if top is None:
        info['error'] = 'not a git repository'
        return info
    if os.path.realpath(top.strip()) != os.path.realpath(repo):
        info['error'] = f'not a repository root (inside {top.strip()})'
    info['git_sha'] = (_git(repo, 'rev-parse', 'HEAD') or '').strip() or None
    info['branch'] = (_git(repo, 'rev-parse', '--abbrev-ref', 'HEAD') or '').strip() or None
    status = _git(repo, 'status', '--porcelain', '--untracked-files=all') or ''
    modified, untracked = [], []
    for line in status.splitlines():
        code, path = line[:2], line[3:]
        if code == '??':
            if _is_code(path):
                untracked.append(path)
        else:
            modified.append(path)
    info['modified_files'] = sorted(modified)
    info['untracked_code_files'] = sorted(untracked)
    info['dirty'] = bool(modified or untracked)
    return info


def venv_packages(venv):
    """Installed distributions of a virtual environment, from its dist-info names.

    Reads directory names only (fast, no interpreter start), so it is cheap enough
    to run before every trial. Returns (sorted list, sha256 of that list).
    """
    venv = expand(venv)
    names = []
    for lib in Path(venv).glob('lib/python*/site-packages'):
        for entry in lib.iterdir():
            if entry.name.endswith('.dist-info'):
                names.append(entry.name[:-len('.dist-info')])
    names.sort()
    return names, sha256_bytes('\n'.join(names).encode())


def provenance(config):
    """Everything that identifies the code and environment behind a trial."""
    prov = config.raw.get('provenance', {})
    repos = [repo_provenance(r) for r in prov.get('repos', [])]
    venvs = {}
    for v in prov.get('venvs', []):
        names, digest = venv_packages(v)
        venvs[expand(v)] = {'n_packages': len(names), 'sha256': digest}
    key = {}
    for mod in ('numpy', 'scipy', 'yaml', 'cv2', 'rclpy'):
        m = sys.modules.get(mod)
        key[mod] = getattr(m, '__version__', None) if m else None
    dirty = any(r['dirty'] or r['error'] for r in repos)
    return {
        'repos': repos,
        'git_sha': {r['repo']: r['git_sha'] for r in repos},
        'dirty': dirty,
        'modified_files': {r['repo']: r['modified_files'] for r in repos if r['modified_files']},
        'untracked_code_files': {r['repo']: r['untracked_code_files'] for r in repos
                                 if r['untracked_code_files']},
        'venvs': venvs,
        'package_versions': key,
        'python': sys.version.split()[0],
        'platform': platform.platform(),
    }


# ---------------------------------------------------------------------- io helpers
def json_default(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path, data):
    """Atomic JSON write, so a crash never leaves a half-written record."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=json_default, allow_nan=True)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------- sessions
def current_session(config):
    """The open session record, or None."""
    path = os.path.join(config.sessions_dir(), 'current.json')
    if not os.path.isfile(path):
        return None
    cur = read_json(path)
    rec = os.path.join(config.sessions_dir(), cur['session_id'], 'calibration.json')
    if not os.path.isfile(rec):
        return None
    data = read_json(rec)
    return data if data.get('open') else None


def cli_parameter_names(argv):
    """Names given with `-p name:=value` (or `--param`) in a ROS argument list."""
    names = []
    argv = list(argv or [])
    for i, a in enumerate(argv):
        if a in ('-p', '--param') and i + 1 < len(argv):
            names.append(argv[i + 1].split(':=')[0].strip())
        elif a.startswith('-p') and ':=' in a and a != '-p':
            names.append(a[2:].split(':=')[0].strip())
    return names


def cli_parameter_value(argv, name):
    argv = list(argv or [])
    for i, a in enumerate(argv):
        if a in ('-p', '--param') and i + 1 < len(argv) and argv[i + 1].split(':=')[0] == name:
            return argv[i + 1].split(':=', 1)[1]
    return None


def experiment_requested(argv):
    value = cli_parameter_value(argv, 'experiment')
    return value is not None and value.strip().lower() in ('true', '1', 'yes')
