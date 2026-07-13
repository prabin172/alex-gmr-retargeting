#!/usr/bin/env python3
"""Stage 2.5 -- canonical-human grounding + persisted contact labels (phasic-v2 M1/T1.2).

Input: outputs/canonical_human/fbx_fresh/<clip>_with_orient.npz (Stage 2 output).
Output: outputs/canonical_human/fbx_fresh/<clip>_grounded.npz.

Establishes the floor as an INVARIANT (z=0) upstream of everything else, instead
of letting each downstream stage estimate it independently (the pattern behind
the per-clip floor hacks this redesign replaces -- see plan.md, repo root).

  1. Detect contact exactly as Stage 3 does (scripts/contact_labels.py, shared
     code, not a re-derivation -- see T1.1 in planLog.md).
  2. Per contacting effector, the support point is the LOWEST of its markers
     (toe not ankle for feet, already how detect_contacts_from_human works --
     see planLog.md T1.1 for the grep confirming this).
  3. Register the floor on STILL plants only (support-point speed < --plant-speed,
     debounced by --plant-min-run), never on the moving approach/transition
     frames -- mirrors Stage 4's `_compute_anchors` plant definition and Stage
     4.5's `_planted_foot_sole_samples` fallback pattern (see
     wiki/concepts/grounding.md "constant-contact" lesson: registering on raw
     contact_flags drags the floor toward a descending/transitioning contact).
  4. Floor = --contact-percentile (default 50 = median) of the pooled still-plant
     support-point heights across ALL contacting effectors (feet AND hands --
     a prone/get-up clip's true support may be either). One rigid vertical shift
     -> floor = 0. Same shift applied to every frame/role (rigid, not a delta --
     morphology scaling's rest-relative-delta invariant is untouched, this runs
     before Stage 3 ever computes a delta).

Persists `contact_flags`, `contact_effector_names`, `contact_support_z`,
`floor_shift` in the output NPZ alongside all of Stage 2's original fields
(positions Z-shifted; everything else passed through unchanged) so Stage 3 can
consume the labels directly instead of recomputing them (T1.3).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from contact_labels import CONTACT_EFFECTORS, detect_contacts_from_human


def _contact_intervals(flag_col):
    """Contiguous True runs -> list of (start, end) inclusive.

    Duplicated from solve_global_trajectory_opt_contactfirst.py's
    `_contact_intervals` (independent CLI scripts, no shared imports for
    solver-internal helpers -- same convention as `_load_model_with_floor` /
    `floor_phase_weight` elsewhere in this codebase)."""
    intervals = []
    t = 0
    n = len(flag_col)
    while t < n:
        if flag_col[t]:
            s = t
            while t < n and flag_col[t]:
                t += 1
            intervals.append((s, t - 1))
        else:
            t += 1
    return intervals


def still_plant_support_samples(positions, role_to_idx, contacts, fps, *,
                                 plant_speed=0.05, plant_min_run=8):
    """Per contacting effector, the support-point (lowest-marker) height on
    STILL plant sub-segments only.

    Mirrors Stage 4's `_compute_anchors` stillness split: within each contact
    interval, find sub-runs where the support point's own speed stays below
    `plant_speed`; only sub-runs of length >= `plant_min_run` count as a real
    plant (shorter dips are momentary velocity zero-crossings, not a stance --
    same debounce rationale as Stage 4/4.5). Falls back to all contact-labelled
    frames for an effector if it never has a still sub-run long enough (keeps a
    sample rather than dropping the effector's registration entirely -- mirrors
    `_planted_foot_sole_samples`'s per-foot fallback).

    Returns: (samples: flat np.ndarray of support-point Z, per_effector dict of
    {name: (T,) float array of support-point Z, NaN where not in contact}).
    """
    dt = 1.0 / float(fps)
    per_effector_z = {}
    samples = []

    for eff, cfg in CONTACT_EFFECTORS.items():
        markers = [r for r in cfg["markers"] if r in role_to_idx]
        flag = contacts.get(eff)
        z_track = np.full(positions.shape[0], np.nan)
        if not markers or flag is None or not flag.any():
            per_effector_z[eff] = z_track
            continue

        support_z = np.min([positions[:, role_to_idx[r], 2] for r in markers], axis=0)
        support_xy = np.stack(
            [np.mean([positions[:, role_to_idx[r], d] for r in markers], axis=0)
             for d in (0, 1)], axis=-1)
        speed = np.zeros(len(flag))
        speed[1:] = np.linalg.norm(np.diff(support_xy, axis=0), axis=1) / dt
        speed[0] = speed[1] if len(flag) > 1 else 0.0

        eff_samples = []
        for (s, e) in _contact_intervals(flag):
            still = speed[s:e + 1] < plant_speed
            k = s
            while k <= e:
                if still[k - s]:
                    j = k
                    while j <= e and still[j - s]:
                        j += 1
                    if j - k >= plant_min_run:
                        eff_samples.extend(support_z[k:j].tolist())
                        z_track[k:j] = support_z[k:j]
                    k = j
                else:
                    k += 1
        if not eff_samples:
            # Fallback: no still sub-run long enough -- keep all contact-labelled
            # frames rather than lose this effector's registration entirely.
            idx = np.where(flag)[0]
            eff_samples = support_z[idx].tolist()
            z_track[idx] = support_z[idx]

        per_effector_z[eff] = z_track
        samples.extend(eff_samples)

    return np.asarray(samples), per_effector_z


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-npz", required=True, type=Path)
    ap.add_argument("--out-npz", required=True, type=Path)
    # Contact-detection flags -- SAME names as Stage 3's, so a clip's detection
    # tuning (e.g. the Luigi onset-hysteresis overrides) threads through
    # identically once Stage 3 stops recomputing labels itself (T1.3).
    ap.add_argument("--foot-contact-height", type=float, default=0.07)
    ap.add_argument("--hand-contact-height", type=float, default=0.08)
    ap.add_argument("--contact-speed", type=float, default=0.4)
    ap.add_argument("--contact-on-height-frac", type=float, default=0.7)
    ap.add_argument("--contact-on-speed-frac", type=float, default=0.5)
    ap.add_argument("--contact-onset-max-delay", type=float, default=0.15)
    ap.add_argument("--foot-flat-tilt", type=float, default=40.0)
    ap.add_argument("--foot-flat-margin", type=float, default=6.0)
    # Floor-registration flags (new -- Stage 2.5 only).
    ap.add_argument("--plant-speed", type=float, default=0.05,
                    help="Support-point speed (m/s) below which a contact frame counts as a "
                         "STILL plant for floor registration (default: 0.05, matches Stage-4 "
                         "_compute_anchors' plant_speed and Stage-4.5's --still-speed).")
    ap.add_argument("--plant-min-run", type=int, default=8,
                    help="Minimum length (frames) of a stillness sub-run to count as a real "
                         "plant, not a momentary speed dip (default: 8, matches Stage-4's "
                         "PLANT_MIN_RUN at native 120 Hz).")
    ap.add_argument("--contact-percentile", type=float, default=50.0,
                    help="Percentile of pooled still-plant support-point heights that "
                         "registers as the floor (default: 50 = median, matches Stage-4.5's "
                         "constant-contact default).")
    args = ap.parse_args()

    z = np.load(args.in_npz, allow_pickle=True)
    roles = [str(x) for x in z["roles"]]
    role_to_idx = {r: i for i, r in enumerate(roles)}
    positions = np.asarray(z["positions"], dtype=np.float64).copy()
    fps = float(z["fps"])
    orientation_role_names = [str(x) for x in z["orientation_role_names"]]
    ori_to_idx = {r: i for i, r in enumerate(orientation_role_names)}
    orientation_mats = np.asarray(z["orientation_mats"], dtype=np.float64)

    contacts, _loose_floor_z = detect_contacts_from_human(
        positions, role_to_idx, fps,
        orientation_mats=orientation_mats, ori_to_idx=ori_to_idx,
        foot_height=args.foot_contact_height,
        hand_height=args.hand_contact_height,
        speed_thresh=args.contact_speed,
        foot_flat_tilt=args.foot_flat_tilt,
        foot_flat_margin=args.foot_flat_margin,
        on_height_frac=args.contact_on_height_frac,
        on_speed_frac=args.contact_on_speed_frac,
        onset_max_delay=args.contact_onset_max_delay,
    )

    samples, per_effector_z = still_plant_support_samples(
        positions, role_to_idx, contacts, fps,
        plant_speed=args.plant_speed, plant_min_run=args.plant_min_run,
    )

    eff_names = list(CONTACT_EFFECTORS.keys())
    contact_flags = np.stack([contacts[e] for e in eff_names], axis=1)
    contact_support_z = np.stack([per_effector_z[e] for e in eff_names], axis=1)

    if samples.size == 0:
        floor_z = 0.0
        shift = 0.0
        print(f"  [WARN] {args.in_npz.name}: no still-plant contact samples -- no shift applied")
    else:
        floor_z = float(np.percentile(samples, args.contact_percentile))
        shift = -floor_z

    positions[:, :, 2] += shift

    med_after = float(np.nanmedian(contact_support_z + shift)) if samples.size else float("nan")
    print(f"[ground_canonical_human] {args.in_npz.name}  N={positions.shape[0]}  "
          f"still-plant samples={samples.size}  floor(p{args.contact_percentile:g})={floor_z:.4f}  "
          f"shift={shift:+.4f} m  support-z-after(median)={med_after:.4f}")

    out = {k: z[k] for k in z.files}
    out["positions"] = positions
    out["contact_flags"] = contact_flags
    out["contact_effector_names"] = np.asarray(eff_names, dtype=object)
    out["contact_support_z"] = contact_support_z + shift
    out["floor_shift"] = np.float64(shift)
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **out)
    print(f"  -> {args.out_npz}")


if __name__ == "__main__":
    main()
