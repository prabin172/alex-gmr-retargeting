"""
Compare orientation tracking errors between the baseline IK run (outputs/ik/)
and the high-ori-weight run (outputs/ik_highori/).

Prints a per-role and aggregate table showing mean / p95 orientation error
in degrees, and the reduction achieved by the higher weights.

Usage:
    conda run -n gmr python scripts/compare_ori_weights.py
"""

import os
import numpy as np

BASELINE_DIR  = "outputs/ik"
HIGHORI_DIR   = "outputs/ik_highori"

DISTAL_ROLES = {"left_foot", "right_foot", "left_hand", "right_hand"}


def load_ori_errors(npz_path):
    """Return orientation_errors_deg (T, 7) and role names from an IK NPZ."""
    d = np.load(npz_path, allow_pickle=True)
    errors = d["orientation_errors_deg"]          # (T, 7)
    roles  = list(d["orientation_role_names"])    # 7 role strings
    return errors, roles


def stats(arr):
    return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95))}


def main():
    baseline_files = {f.replace("_ik.npz", ""): os.path.join(BASELINE_DIR, f)
                      for f in os.listdir(BASELINE_DIR) if f.endswith("_ik.npz")}
    highori_files  = {f.replace("_ik_highori.npz", ""): os.path.join(HIGHORI_DIR, f)
                      for f in os.listdir(HIGHORI_DIR) if f.endswith("_ik_highori.npz")}

    common = sorted(set(baseline_files) & set(highori_files))
    if not common:
        print("No matching clips found. Run run_highori_weights.sh first.")
        return

    # Accumulate per-role arrays across all clips
    all_base  = {}   # role → list of per-frame errors
    all_high  = {}

    print(f"\nFound {len(common)} matching clip(s): {', '.join(common)}\n")

    for stem in common:
        base_err, roles = load_ori_errors(baseline_files[stem])
        high_err, _     = load_ori_errors(highori_files[stem])

        for ri, role in enumerate(roles):
            all_base.setdefault(role, []).append(base_err[:, ri])
            all_high.setdefault(role, []).append(high_err[:, ri])

    roles_ordered = list(all_base.keys())

    # -----------------------------------------------------------------------
    # Per-role table
    # -----------------------------------------------------------------------
    hdr = (f"  {'role':16s}  {'base_mean°':>10}  {'high_mean°':>10}  "
           f"{'Δmean°':>8}  {'base_p95°':>10}  {'high_p95°':>10}  {'Δp95°':>8}")
    sep = "  " + "-"*16 + "  " + ("  ".join(["-"*10]*4 + ["-"*8]*2))
    print("PER-ROLE ORIENTATION ERROR (degrees, averaged across all clips)")
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))

    role_deltas_mean = []
    role_deltas_p95  = []

    for role in roles_ordered:
        b = np.concatenate(all_base[role])
        h = np.concatenate(all_high[role])
        bs, hs = stats(b), stats(h)
        dm = hs["mean"] - bs["mean"]
        dp = hs["p95"]  - bs["p95"]
        role_deltas_mean.append((role, dm))
        role_deltas_p95.append((role, dp))

        marker = " ◄" if role in DISTAL_ROLES else ""
        print(f"  {role:16s}  {bs['mean']:10.2f}  {hs['mean']:10.2f}  "
              f"{dm:+8.2f}  {bs['p95']:10.2f}  {hs['p95']:10.2f}  {dp:+8.2f}{marker}")

    print("=" * len(hdr))
    print("  ◄ distal roles with doubled weight\n")

    # -----------------------------------------------------------------------
    # Distal vs proximal summary
    # -----------------------------------------------------------------------
    def group_stats(roles_subset):
        b = np.concatenate([np.concatenate(all_base[r]) for r in roles_subset])
        h = np.concatenate([np.concatenate(all_high[r]) for r in roles_subset])
        return stats(b), stats(h)

    distal_roles  = [r for r in roles_ordered if r in DISTAL_ROLES]
    proximal_roles = [r for r in roles_ordered if r not in DISTAL_ROLES]

    bs_d, hs_d = group_stats(distal_roles)
    bs_p, hs_p = group_stats(proximal_roles)

    print("SUMMARY")
    print("-" * 60)
    print(f"  {'group':18s}  {'base_mean°':>10}  {'high_mean°':>10}  {'Δmean°':>8}")
    print(f"  {'distal (2× weight)':18s}  {bs_d['mean']:10.2f}  {hs_d['mean']:10.2f}  "
          f"{hs_d['mean']-bs_d['mean']:+8.2f}")
    print(f"  {'proximal':18s}  {bs_p['mean']:10.2f}  {hs_p['mean']:10.2f}  "
          f"{hs_p['mean']-bs_p['mean']:+8.2f}")
    print()

    # -----------------------------------------------------------------------
    # Per-clip breakdown for distal roles
    # -----------------------------------------------------------------------
    print("PER-CLIP DISTAL ORIENTATION ERROR (degrees, mean across distal roles)")
    print("-" * 60)
    print(f"  {'clip':42s}  {'base°':>7}  {'high°':>7}  {'Δ°':>7}")
    print(f"  {'-'*42}  {'-'*7}  {'-'*7}  {'-'*7}")

    for stem in common:
        base_err, roles = load_ori_errors(baseline_files[stem])
        high_err, _     = load_ori_errors(highori_files[stem])
        distal_idx = [roles.index(r) for r in roles if r in DISTAL_ROLES]
        b_mean = float(base_err[:, distal_idx].mean())
        h_mean = float(high_err[:, distal_idx].mean())
        print(f"  {stem:42s}  {b_mean:7.2f}  {h_mean:7.2f}  {h_mean-b_mean:+7.2f}")

    print()


if __name__ == "__main__":
    main()
