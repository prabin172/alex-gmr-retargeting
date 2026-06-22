from pathlib import Path
import json

from general_motion_retargeting.source_adapters.canonical_human import (
    CANONICAL_BODY_NAMES,
    frame_to_jsonable,
    make_neutral_standing_frame,
    make_t_pose_frame,
    validate_canonical_human_frame,
)

repo_root = Path(__file__).resolve().parents[1]
ik_path = repo_root / "general_motion_retargeting/ik_configs/smplx_to_alex.json"

ik_cfg = json.loads(ik_path.read_text())
expected_by_ik = set(ik_cfg["human_body_names_expected"])
canonical = set(CANONICAL_BODY_NAMES)

print("IK config:", ik_path)
print("Canonical body count:", len(CANONICAL_BODY_NAMES))
print("IK expected human body count:", len(expected_by_ik))
print()

missing_from_canonical = sorted(expected_by_ik - canonical)
unused_by_ik = sorted(canonical - expected_by_ik)

if missing_from_canonical:
    raise ValueError(f"IK expects names missing from canonical adapter: {missing_from_canonical}")

if unused_by_ik:
    print("Canonical names not currently used by IK:")
    for name in unused_by_ik:
        print(" ", name)
    print()

neutral = make_neutral_standing_frame()
t_pose = make_t_pose_frame()

validate_canonical_human_frame(neutral)
validate_canonical_human_frame(t_pose)

print("Neutral frame validation: OK")
print("T-pose frame validation: OK")
print()

print("Neutral standing frame landmarks:")
for name in CANONICAL_BODY_NAMES:
    pose = neutral[name]
    pos = pose["pos"]
    quat = pose["quat_wxyz"]
    print(f"{name:15s} pos=[{pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f}] quat_wxyz={quat}")

out_dir = repo_root / "outputs/debug"
out_dir.mkdir(parents=True, exist_ok=True)

neutral_out = out_dir / "canonical_neutral_frame.json"
t_pose_out = out_dir / "canonical_t_pose_frame.json"

neutral_out.write_text(json.dumps(frame_to_jsonable(neutral), indent=2))
t_pose_out.write_text(json.dumps(frame_to_jsonable(t_pose), indent=2))

print()
print("Wrote debug examples:")
print(" ", neutral_out)
print(" ", t_pose_out)
print()
print("These output JSON files are debug outputs and should be ignored by git.")
