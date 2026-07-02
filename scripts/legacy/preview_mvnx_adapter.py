from pathlib import Path
import argparse
import json
import csv

import matplotlib.pyplot as plt
import numpy as np

from general_motion_retargeting.source_adapters.mvnx import (
    MVNX_TO_CANONICAL_SEGMENT,
    read_mvnx_canonical_frames,
)
from general_motion_retargeting.source_adapters.canonical_human import (
    CANONICAL_BODY_NAMES,
)
from general_motion_retargeting.retargeting.rest_pose_scaling import (
    CANONICAL_TREE_SEGMENTS,
    segment_length,
)

repo_root = Path(__file__).resolve().parents[1]

edges = [
    ("pelvis", "torso"),
    ("torso", "head"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_foot"),
    ("torso", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),
    ("torso", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
]

def pos(frame, role):
    return np.asarray(frame[role]["pos"], dtype=float)

def draw_frame(frame, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, dim0, dim1, xlabel, ylabel, subtitle in [
        (axes[0], 1, 2, "Y left", "Z up", "Front view"),
        (axes[1], 0, 2, "X forward", "Z up", "Side view"),
    ]:
        for a, b in edges:
            pa = pos(frame, a)
            pb = pos(frame, b)
            ax.plot([pa[dim0], pb[dim0]], [pa[dim1], pb[dim1]], linewidth=2)

        xs = [pos(frame, r)[dim0] for r in CANONICAL_BODY_NAMES]
        ys = [pos(frame, r)[dim1] for r in CANONICAL_BODY_NAMES]
        ax.scatter(xs, ys, s=25)

        for role in ["pelvis", "head", "left_foot", "right_foot", "left_hand", "right_hand"]:
            p = pos(frame, role)
            ax.text(p[dim0], p[dim1], role, fontsize=8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Preview MVNX to canonical-human adapter.")
    parser.add_argument("mvnx_path", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=240)
    parser.add_argument("--max-frames", type=int, default=20)
    args = parser.parse_args()

    frames, meta = read_mvnx_canonical_frames(
        mvnx_path=args.mvnx_path,
        frame_type="normal",
        start_frame=args.start_frame,
        stride=args.stride,
        max_frames=args.max_frames,
    )

    if not frames:
        raise RuntimeError("No canonical frames loaded.")

    out_dir = repo_root / "outputs/debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.mvnx_path.stem
    summary_path = out_dir / f"{stem}_canonical_preview_summary.json"
    csv_path = out_dir / f"{stem}_canonical_preview_segment_lengths.csv"
    first_plot = out_dir / f"{stem}_canonical_first_frame.png"
    last_plot = out_dir / f"{stem}_canonical_last_frame.png"

    print("MVNX path:", args.mvnx_path)
    print("Loaded canonical frames:", len(frames))
    print("Frame rate:", meta.get("frame_rate"))
    print("Stride:", args.stride)
    print("Matched normal frame range:", meta["matched_frame_indices"][0], "to", meta["matched_frame_indices"][-1])

    print()
    print("MVNX -> canonical mapping:")
    for k, v in MVNX_TO_CANONICAL_SEGMENT.items():
        print(f"  {k:16s} <- {v}")

    first = frames[0]
    last = frames[-1]

    print()
    print("First loaded canonical frame positions:")
    for role in CANONICAL_BODY_NAMES:
        p = pos(first, role)
        q = np.asarray(first[role]["quat_wxyz"], dtype=float)
        print(
            f"  {role:16s} pos=[{p[0]: .3f},{p[1]: .3f},{p[2]: .3f}] "
            f"quat_wxyz=[{q[0]: .3f},{q[1]: .3f},{q[2]: .3f},{q[3]: .3f}]"
        )

    rows = []
    for parent, child in CANONICAL_TREE_SEGMENTS:
        lengths = [segment_length(frame, parent, child) for frame in frames]
        rows.append({
            "segment": f"{parent}->{child}",
            "parent": parent,
            "child": child,
            "mean_m": float(np.mean(lengths)),
            "std_m": float(np.std(lengths)),
            "min_m": float(np.min(lengths)),
            "max_m": float(np.max(lengths)),
        })

    print()
    print("Segment length stability over loaded frames:")
    print("segment                    mean_m   std_m    min_m    max_m")
    for row in rows:
        print(
            f"{row['segment']:26s} "
            f"{row['mean_m']: .4f}  {row['std_m']: .4f}  {row['min_m']: .4f}  {row['max_m']: .4f}"
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path.write_text(json.dumps({
        "metadata": meta,
        "mapping": MVNX_TO_CANONICAL_SEGMENT,
        "segment_length_summary": rows,
    }, indent=2))

    draw_frame(first, first_plot, f"{stem}: first loaded canonical frame")
    draw_frame(last, last_plot, f"{stem}: last loaded canonical frame")

    print()
    print("Wrote:")
    print(" ", summary_path)
    print(" ", csv_path)
    print(" ", first_plot)
    print(" ", last_plot)

if __name__ == "__main__":
    main()
