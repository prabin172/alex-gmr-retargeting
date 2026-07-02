from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

def strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def floats_from_text(text):
    if text is None:
        return []
    out = []
    for x in text.replace(",", " ").split():
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out

def short_attrs(elem):
    return {k: v for k, v in elem.attrib.items() if len(str(v)) < 120}

def find_all_by_local_name(root, local_name):
    return [e for e in root.iter() if strip_ns(e.tag) == local_name]

def main():
    parser = argparse.ArgumentParser(description="Inspect an Xsens MVNX file structure.")
    parser.add_argument("mvnx_path", type=Path, help="Path to .mvnx file")
    parser.add_argument("--max-frames", type=int, default=3, help="Number of initial frames to inspect")
    args = parser.parse_args()

    path = args.mvnx_path
    if not path.exists():
        raise FileNotFoundError(path)

    print("MVNX file:", path)
    print("Size MB:", f"{path.stat().st_size / (1024 * 1024):.2f}")

    tree = ET.parse(path)
    root = tree.getroot()

    print()
    print("Root tag:", strip_ns(root.tag))
    print("Root attrs:", short_attrs(root))

    tag_counts = Counter(strip_ns(e.tag) for e in root.iter())
    print()
    print("Top XML tags:")
    for tag, count in tag_counts.most_common(30):
        print(f"  {tag:25s} {count}")

    print()
    print("Likely metadata elements:")
    for name in ["subject", "comment", "securityCode", "frameRate", "sampleRate", "recDate", "originalFilename"]:
        elems = find_all_by_local_name(root, name)
        if elems:
            print(f"  {name}:")
            for e in elems[:5]:
                text = (e.text or "").strip()
                if len(text) > 160:
                    text = text[:160] + "..."
                print(f"    attrs={short_attrs(e)} text={text!r}")

    segments_parent = None
    for e in root.iter():
        if strip_ns(e.tag) == "segments":
            segments_parent = e
            break

    segments = []
    if segments_parent is not None:
        for child in list(segments_parent):
            if strip_ns(child.tag) == "segment":
                segments.append({
                    "id": child.attrib.get("id"),
                    "label": child.attrib.get("label"),
                    "name": child.attrib.get("name"),
                    "attrs": short_attrs(child),
                })

    print()
    print("Segments found:", len(segments))
    for i, s in enumerate(segments):
        display = s.get("label") or s.get("name") or s.get("id")
        print(f"  {i:02d}: id={s.get('id')} label/name={display} attrs={s['attrs']}")

    joints_parent = None
    for e in root.iter():
        if strip_ns(e.tag) == "joints":
            joints_parent = e
            break

    joints = []
    if joints_parent is not None:
        for child in list(joints_parent):
            if strip_ns(child.tag) == "joint":
                joints.append(short_attrs(child))

    print()
    print("Joints found:", len(joints))
    for i, j in enumerate(joints[:40]):
        print(f"  {i:02d}: {j}")
    if len(joints) > 40:
        print(f"  ... {len(joints) - 40} more")

    frames = find_all_by_local_name(root, "frame")
    print()
    print("Frames found:", len(frames))

    if not frames:
        print("No <frame> elements found. This may not be a normal MVNX export.")
    else:
        frame_type_counts = Counter(f.attrib.get("type", "NO_TYPE") for f in frames)
        print("Frame type counts:")
        for k, v in frame_type_counts.items():
            print(f"  {k}: {v}")

        print()
        print(f"First {min(args.max_frames, len(frames))} frame summaries:")

        for idx, frame in enumerate(frames[:args.max_frames]):
            print()
            print(f"Frame {idx}: attrs={short_attrs(frame)}")

            child_summaries = []
            for child in list(frame):
                tag = strip_ns(child.tag)
                vals = floats_from_text(child.text)
                child_summaries.append({
                    "tag": tag,
                    "attrs": short_attrs(child),
                    "n_floats": len(vals),
                    "first_values": vals[:12],
                })

            for c in child_summaries:
                print(
                    f"  {c['tag']:25s} n_floats={c['n_floats']:5d} "
                    f"attrs={c['attrs']} first={c['first_values']}"
                )

        print()
        print("Frame child float-count patterns:")
        pattern_counts = defaultdict(Counter)
        for frame in frames[: min(200, len(frames))]:
            ftype = frame.attrib.get("type", "NO_TYPE")
            for child in list(frame):
                tag = strip_ns(child.tag)
                n = len(floats_from_text(child.text))
                pattern_counts[tag][n] += 1

        for tag, counts in sorted(pattern_counts.items()):
            print(f"  {tag}: {dict(counts)}")

    n_segments = len(segments)
    print()
    print("Useful inference:")
    if n_segments:
        print(f"  segment_count = {n_segments}")
        print(f"  expected orientation floats if quaternion per segment = {4 * n_segments}")
        print(f"  expected position floats if xyz per segment = {3 * n_segments}")
    print("  We need global segment positions and orientations for canonical-human conversion.")

    out_dir = Path("outputs/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_mvnx_structure_summary.json"

    summary = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "root_tag": strip_ns(root.tag),
        "root_attrs": short_attrs(root),
        "tag_counts_top": tag_counts.most_common(100),
        "segments": segments,
        "num_segments": len(segments),
        "joints_preview": joints[:80],
        "num_joints": len(joints),
        "num_frames": len(frames),
        "frame_type_counts": dict(Counter(f.attrib.get("type", "NO_TYPE") for f in frames)),
    }

    out_path.write_text(json.dumps(summary, indent=2))
    print()
    print("Wrote summary:")
    print(" ", out_path)

if __name__ == "__main__":
    main()
