"""Stage table of aircraft-rl-next-study-v1 (signal-limit v1): P0-I, Stage 0, Stage A, Stage B.

Pure data, no model. docs/aircraft-rl-next-study-v1.json is authoritative (it wins over the
.md); ``verify_against_study`` cross-checks this table against it field by field and returns
every disagreement, so a study amendment that this table does not follow refuses the launch
instead of running the old design. The launcher (signal_limit_launch.py) and the UWaterloo
Stage A runner (uw_stage_a.py) read the stage order, the init endpoint role of every stage and
every update count from here, never from literals of their own.

Init endpoint roles (study.json stages[*].init, implementation_work_list launch.py
"init_endpoint, pinned per stage"):
  I   controlled400 sft-1e5-time, pinned by the plan (P0-I)
  W   the freshpool W adapter, pinned by the plan (Stage 0, Stage A)
  I*  the H-select-best Stage-A endpoint among A1*, A2*, A3*: pinned in the launcher state the
      moment Gate A passes, by run ID and adapter SHA256 (Stage B, also the KL reference)
"""

STUDY_PATH = "docs/aircraft-rl-next-study-v1.json"
STUDY_ID = "aircraft-rl-next-study-v1"
SEED = 42
GRID = (1e-5, 5e-5, 1e-4)
# study.json stages: Stage A data "S_a, 600 updates"; Stage B rl.updates "all three LRs to 100
# accepted; top two ... continue without reset to N = 300"; N_accepted 300.
UPDATES = {"stage_a": 600, "rl_screen": 100, "final": 300}
GROUPS = (4, 8, 16)  # every G the study names (P0-I predicts all three)
STAGE_B_GROUPS = (8, 16)  # GB0: "G = smallest in {8,16} with predicted acceptance >= a_floor"
ASSAY_GROUP = 4  # fixed_profile.assay: "assay G stays 4 regardless of training G"
P0_GROUP = 4  # P0-I: "existing p0_screen at G4 with init I"
CAP_NUMERATOR = 1.25  # refusal arithmetic a_floor = N x 1.25 / |S_b|

# (id, objective mode, LR rule). "grid": every LR of GRID; "a1_star": A1*'s LR only.
STAGE_A_ARMS = (
    ("A1", "sft", "grid"),
    ("A2", "egsft", "grid"),
    ("A3", "hn_sft", "grid"),
    ("A4", "hls_sft", "a1_star"),
    ("A5", "cce_sft", "grid"),
)
STAGE_A_ARM_COUNT = 13
# Gate A candidates (A2 x3) and the comparator family (C* = max(A1*, A3*, A5*)). Design-review
# amendment 1 (study.json design_review_v1, finding 3): A3 (HN-SFT) is not the limit of any
# RL reward, so it is a supervised comparator, not a Gate A candidate.
LIMIT_ARMS = ("A2",)
COMPARATOR_ARMS = ("A1", "A3", "A5")
I_STAR_FROM = ("A1", "A2", "A3")  # Stage B init: "H-select-best endpoint among A1*, A2*, A3*"
# LIM-T continues the Gate-A limit objective (study.json Stage B controls LIM-T).
LIM_T_MODES = {"A2": "egsft"}
# Stage B rl.reward: "exact-EOS 0/1 ...", the reward whose Dr-GRPO limit A2 is.
RL_REWARDS = {"A2": "exact"}

STAGES = (
    {"id": "P0-I", "init": "I", "kind": "p0", "group": P0_GROUP, "predict_groups": GROUPS,
     "gate": "PI"},
    {"id": "Stage 0", "init": "W", "kind": "stage0", "probe_groups": STAGE_B_GROUPS,
     "gate": "E0"},
    {"id": "Stage A", "init": "W", "kind": "stage_a", "updates": UPDATES["stage_a"],
     "arms": STAGE_A_ARMS, "gate": "GA"},
    {"id": "Stage B", "init": "I*", "kind": "stage_b", "choose_groups": STAGE_B_GROUPS,
     "rl_screen_updates": UPDATES["rl_screen"], "final_updates": UPDATES["final"],
     "gates": ("GB0", "G2", "G3a", "G3b")},
)
INIT_ROLES = {stage["id"]: stage["init"] for stage in STAGES}


def lr_tag(lr):
    return f"{lr:.0e}"


def stage_a_run_id(arm, lr):
    return f"stage-a-{arm.lower()}-{lr_tag(lr)}-s{SEED}"


def stage_a_runs(a1_star_lr=None):
    """[(arm, mode, lr, run_id)] in launch order: every grid arm at every LR, then A4 at A1*'s
    LR when it is known (A4 is a reported control and cannot pass Gate A, so it runs after
    A1's selection and before Gate A)."""
    runs = []
    for arm, mode, rule in STAGE_A_ARMS:
        if rule == "grid":
            runs += [(arm, mode, lr, stage_a_run_id(arm, lr)) for lr in GRID]
    if a1_star_lr is not None:
        if a1_star_lr not in GRID:
            raise ValueError(f"A1* LR {a1_star_lr!r} is not on the grid {GRID}")
        runs.append(("A4", "hls_sft", a1_star_lr, stage_a_run_id("A4", a1_star_lr)))
    return runs


def a_floor(final_updates, s_b_size):
    """study.json lineage 'explicit refusal arithmetic a_floor = N x 1.25 / |S_b|'."""
    if type(final_updates) is not int or type(s_b_size) is not int or s_b_size <= 0:
        raise ValueError("a_floor needs integer N and a positive |S_b|")
    return final_updates * CAP_NUMERATOR / s_b_size


def verify_against_study(study):
    """Every disagreement between this table and the study JSON, as strings (empty = agree)."""
    problems = []
    if study.get("study_id") != STUDY_ID:
        problems.append(f"study_id {study.get('study_id')!r} != {STUDY_ID!r}")
    stages = {stage.get("id"): stage for stage in study.get("stages") or []}
    if list(stages) != [stage["id"] for stage in STAGES]:
        problems.append(f"stage order {list(stages)} != {[s['id'] for s in STAGES]}")
        return problems
    if not str(stages["P0-I"].get("init", "")).startswith("I = "):
        problems.append("P0-I init is not I")
    if not str(stages["Stage A"].get("init", "")).startswith("W"):
        problems.append("Stage A init is not W")
    if not str(stages["Stage B"].get("init", "")).startswith("I* ="):
        problems.append("Stage B init is not I*")
    if f"{UPDATES['stage_a']} updates" not in str(stages["Stage A"].get("data")):
        problems.append("Stage A update count differs")
    arms = stages["Stage A"].get("arms") or []
    if [a.get("id") for a in arms] != [a[0] for a in STAGE_A_ARMS]:
        problems.append("Stage A arm ids differ")
    for (arm, mode, rule), entry in zip(STAGE_A_ARMS, arms):
        lrs = entry.get("lrs")
        if rule == "grid" and lrs != list(GRID):
            problems.append(f"{arm} LRs {lrs} != {list(GRID)}")
        if rule == "a1_star" and lrs != "A1* LR only":
            problems.append(f"{arm} LR rule {lrs!r} is not 'A1* LR only'")
        if not str(entry.get("mode", "")).startswith(mode):
            problems.append(f"{arm} mode {entry.get('mode')!r} is not {mode!r}")
    if stages["Stage A"].get("arm_count") != STAGE_A_ARM_COUNT or len(
            stage_a_runs(GRID[0])) != STAGE_A_ARM_COUNT:
        problems.append("Stage A arm count differs from 13")
    rl = stages["Stage B"].get("rl") or {}
    if rl.get("N_accepted") != UPDATES["final"] or rl.get("lrs") != list(GRID):
        problems.append("Stage B RL N or LRs differ")
    if (f"to {UPDATES['rl_screen']} accepted" not in str(rl.get("updates"))
            or f"N = {UPDATES['final']}" not in str(rl.get("updates"))):
        problems.append("Stage B RL update schedule differs")
    reward = str(rl.get("reward"))
    if not reward.startswith("exact-EOS 0/1") or "if A3 won" in reward:
        problems.append("Stage B RL reward rule differs")
    controls = [c.get("id") for c in stages["Stage B"].get("controls") or []]
    if controls != ["SFT-E(lr)", "SFT-T(lr)", "LIM-T"]:
        problems.append(f"Stage B controls {controls}")
    assay = str((study.get("fixed_profile") or {}).get("assay"))
    if f"assay G stays {ASSAY_GROUP}" not in assay:
        problems.append("fixed_profile assay G is not 4")
    return problems
