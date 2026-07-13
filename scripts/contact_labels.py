"""Shared contact-detection machinery (phasic-v2, M1/T1.1).

Extracted verbatim from scripts/solve_fbx_canonical_alex_contactfirst.py so that
Stage 2.5 (scripts/ground_canonical_human.py) and Stage 3 (contact-first IK)
detect contact identically from the same canonical human data — one detector,
not two independently-tuned copies. See wiki/concepts/phasic-architecture.md
(once written) and plan.md M1.

CONTACT_EFFECTORS here carries every key the two consumers need between them:
Stage 2.5 uses only `markers` + `flat_ori_role`; Stage 3 additionally uses
`body`/`axis_local`/`world_dir`/`ori_role` for the solver's contact-alignment
terms (those solver-only concerns — CONTACT_ALIGN_WEIGHT, CONTACT_POS,
FOOT_POS_ROLE, etc. — stay in Stage 3, not moved here).
"""
from __future__ import annotations

import numpy as np

CONTACT_EFFECTORS = {
    "left_foot": dict(
        markers=["left_ankle", "left_toe"],
        body="LEFT_FOOT",
        axis_local=np.array([0.0, 0.0, 1.0]),   # foot up-axis
        world_dir=np.array([0.0, 0.0, 1.0]),    # -> world +Z (flat)
        ori_role="left_foot",
        # Only treat as a flat support when the *human* foot is itself near-flat:
        # the canonical foot frame's local-Z (sole normal) within tilt of world +Z.
        flat_ori_role="left_foot",
    ),
    "right_foot": dict(
        markers=["right_ankle", "right_toe"],
        body="RIGHT_FOOT",
        axis_local=np.array([0.0, 0.0, 1.0]),
        world_dir=np.array([0.0, 0.0, 1.0]),
        ori_role="right_foot",
        flat_ori_role="right_foot",
    ),
    "left_hand": dict(
        markers=["left_wrist", "left_hand_middle"],
        body="LEFT_GRIPPER_Z_LINK",
        axis_local=np.array([1.0, 0.0, 0.0]),   # gripper +X = palm/finger-front normal
        world_dir=np.array([0.0, 0.0, -1.0]),   # -> world -Z (press down)
        ori_role="left_hand",
    ),
    "right_hand": dict(
        markers=["right_wrist", "right_hand_middle"],
        body="RIGHT_GRIPPER_Z_LINK",
        axis_local=np.array([1.0, 0.0, 0.0]),
        world_dir=np.array([0.0, 0.0, -1.0]),
        ori_role="right_hand",
    ),
}


def detect_contacts_from_human(positions, role_to_idx, fps, *,
                               orientation_mats=None, ori_to_idx=None,
                               foot_height=0.07, hand_height=0.08,
                               speed_thresh=0.4, foot_flat_tilt=40.0, floor_pct=1.0,
                               foot_flat_margin=6.0, foot_flat_min_base_frames=20,
                               on_height_frac=1.0, on_speed_frac=1.0,
                               onset_max_delay=0.15):
    """Per-frame ground-contact flags for each effector, from human mocap.

    An effector is "in contact" at frame t when the lowest of its markers is
    within `*_height` metres of the clip floor AND moving slower than
    `speed_thresh` (m/s). The floor is the low percentile of the feet markers'
    height across the whole clip.

    Onset hysteresis (`on_height_frac`/`on_speed_frac` < 1): the START of each
    contact interval is delayed until the effector passes STRICTER thresholds
    (height*frac, speed*frac) — the loose thresholds fire while it is still
    descending into a pose. The delay is capped at `onset_max_delay` seconds so
    a crouched plant that hovers under the loose gate without ever passing the
    strict one is trimmed, never dropped (uncapped hysteresis deleted whole
    genuine plant intervals on get-up clips). Release unchanged. 1.0/1.0 = off.

    For effectors with a `flat_ori_role`, contact additionally requires the
    *human* segment to be near-flat. This distinguishes a flat plantar support
    from a foot that is merely near the floor while folded (toes/side down during
    a get-up), where forcing the robot foot flat would just fight tracking.

    IMPORTANT — the flatness gate is RELATIVE to a per-foot self-calibrated
    baseline, not absolute. The canonical foot frame's local-Z is NOT the true
    sole normal: `x = toe−ankle` is the foot's bone axis, declined ~18° below
    horizontal (ankle sits above the grounded toe), so a perfectly FLAT foot reads
    ~18° tilt (+ a ~4° L/R skew) — pure frame geometry, not motion (see
    wiki/concepts/orientation-frames.md FOOTGUN). Gating raw `tilt < 40°` is
    therefore nearly inert. Instead we estimate each foot's own flat baseline as
    the p15 of its tilt over height+speed candidate-contact frames (robust to the
    tilted phantoms in the tail), and require `tilt − baseline < foot_flat_margin`.
    Falls back to the absolute `tilt < foot_flat_tilt` cap if a foot has fewer
    than `foot_flat_min_base_frames` candidate frames (baseline unreliable).

    Returns: (dict effector -> bool array (N,), floor_z).
    """
    N = positions.shape[0]
    dt = 1.0 / float(fps)

    def marker_z(role):
        return positions[:, role_to_idx[role], 2]

    def marker_speed(role):
        p = positions[:, role_to_idx[role], :]
        v = np.zeros(N)
        v[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
        v[0] = v[1] if N > 1 else 0.0
        return v

    # Floor estimate from the feet markers (lowest they reach).
    foot_roles = [r for eff in ("left_foot", "right_foot")
                  for r in CONTACT_EFFECTORS[eff]["markers"] if r in role_to_idx]
    foot_min_z = np.min([marker_z(r) for r in foot_roles], axis=0)
    floor_z = float(np.percentile(foot_min_z, floor_pct))

    contacts = {}
    for eff, cfg in CONTACT_EFFECTORS.items():
        markers = [r for r in cfg["markers"] if r in role_to_idx]
        if not markers:
            contacts[eff] = np.zeros(N, dtype=bool)
            continue
        h = np.min([marker_z(r) for r in markers], axis=0) - floor_z
        spd = np.min([marker_speed(r) for r in markers], axis=0)
        hthr = foot_height if "foot" in eff else hand_height
        flag = (h < hthr) & (spd < speed_thresh)

        flat_role = cfg.get("flat_ori_role")
        if flat_role is not None and orientation_mats is not None and ori_to_idx is not None \
                and flat_role in ori_to_idx:
            up = orientation_mats[:, ori_to_idx[flat_role], :, 2]  # frame local-Z (NOT true sole normal)
            tilt = np.degrees(np.arccos(np.clip(np.abs(up @ np.array([0.0, 0.0, 1.0])), -1, 1)))
            # Per-foot self-calibrated flat baseline from height+speed candidate frames
            # (p15 = the foot's flattest ≈ its anatomical ~18° declination; robust to
            # tilted phantoms which live in the upper tail). Gate on tilt-above-baseline.
            cand = tilt[flag]
            if cand.size >= foot_flat_min_base_frames:
                base = float(np.percentile(cand, 15))
                flag = flag & ((tilt - base) < foot_flat_margin)
            else:
                flag = flag & (tilt < foot_flat_tilt)   # too few candidates → absolute cap

        if on_height_frac < 1.0 or on_speed_frac < 1.0:
            strict = flag & (h < hthr * on_height_frac) & (spd < speed_thresh * on_speed_frac)
            cap = max(0, int(round(onset_max_delay * fps)))
            out = np.zeros(N, dtype=bool)
            t = 0
            while t < N:
                if not flag[t]:
                    t += 1
                    continue
                a = t
                while t < N and flag[t]:
                    t += 1
                b = t                                   # loose interval [a, b)
                s = np.where(strict[a:b])[0]
                onset = a + (min(int(s[0]), cap) if len(s) else cap)
                out[min(onset, b - 1):b] = True         # trim the start, never drop
            flag = out

        contacts[eff] = flag

    return contacts, floor_z


def debounce_flags(flag, min_run):
    """Remove ON/OFF runs shorter than min_run (fill gaps, drop specks).

    Kills marginal-threshold flicker without touching genuine long contacts."""
    if min_run <= 1:
        return flag.copy()
    out = flag.copy().astype(bool)
    n = len(out)
    # drop short ON specks, then fill short OFF gaps
    for target in (True, False):
        i = 0
        while i < n:
            j = i
            while j < n and out[j] == out[i]:
                j += 1
            if out[i] == target and (j - i) < min_run and not (i == 0 and j == n):
                out[i:j] = not target
            i = j
    return out


def ramp_envelope(flag, ramp, preroll):
    """Per-frame contact weight in [0,1] from a boolean contact timeline.

    `preroll` extends each contact earlier (anticipation: begin easing the foot/
    hand toward the support face before touchdown). `ramp` applies a cosine rise
    into each leading edge and fall out of each trailing edge, so the contact
    constraints cross-fade in/out instead of snapping at full weight."""
    n = len(flag)
    base = flag.copy().astype(bool)
    if preroll > 0:
        pr = base.copy()
        idx = np.where(base)[0]
        for i in idx:
            pr[max(0, i - preroll):i] = True
        base = pr
    env = base.astype(np.float64)
    if ramp > 0:
        def cosramp(k):   # k=1..ramp -> rises toward 1
            return 0.5 * (1.0 - np.cos(np.pi * k / (ramp + 1)))
        for i in range(n):
            if not base[i]:
                continue
            if i == 0 or not base[i - 1]:            # leading edge -> ramp preceding frames
                for k in range(1, ramp + 1):
                    p = i - k
                    if p >= 0 and not base[p]:
                        env[p] = max(env[p], cosramp(ramp - k + 1))
            if i == n - 1 or not base[i + 1]:        # trailing edge -> ramp following frames
                for k in range(1, ramp + 1):
                    p = i + k
                    if p < n and not base[p]:
                        env[p] = max(env[p], cosramp(ramp - k + 1))
    return env
