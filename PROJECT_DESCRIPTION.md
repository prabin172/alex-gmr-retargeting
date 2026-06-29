# Project Description: alex-gmr-retargeting

## What This Project Does

This repository implements a **motion retargeting pipeline** that takes human motion capture data and transfers it onto the **IHMC Alex humanoid robot**. The core problem is morphology mismatch: a human body and a robot body have different proportions, degrees of freedom, and coordinate conventions, so raw human poses cannot be applied directly. The pipeline bridges this gap via a canonical intermediate representation and MuJoCo-based inverse kinematics (IK).

---

## High-Level Pipeline

```
Raw MoCap Data (MVNX / FBX)
        │
        ▼
[Source Adapter] ── maps proprietary segment names to canonical roles
        │
        ▼
Canonical Human Frame (15 body roles + 4 auxiliary contact sites)
  pelvis, torso, head, left/right hip/knee/foot,
  left/right shoulder/elbow/hand,
  left/right palm, left/right sole
        │
        ▼
[Morphology Delta Scaling] ── scales motion deltas by human-to-robot limb ratios
        │
        ▼
[IK Solver — MuJoCo QP] ── solves joint angles for Alex given weighted position/orientation targets
        │
        ▼
Output: robot joint trajectories (.npz) or IHMC-ready JSON
        │
        ▼
[Visualization] ── GIF / MP4 renders of the retargeted motion on Alex
```

---

## Repository Structure

```
alex-gmr-retargeting/
├── general_motion_retargeting/      # Main Python package
│   ├── source_adapters/
│   │   ├── canonical_human.py       # Canonical skeleton definition (15 roles + 4 aux)
│   │   └── mvnx.py                  # Xsens MVNX → canonical adapter
│   ├── retargeting/
│   │   ├── morphology_delta.py      # Measures and scales limb-length ratios
│   │   └── rest_pose_scaling.py     # Rest-pose-based scaling utilities
│   ├── robot_configs/
│   │   ├── alex.json                # Alex joint/body name mapping
│   │   ├── alex_retarget_sites.json # Named retargeting end-effector sites on Alex
│   │   └── alex_with_sites.json     # Full config with palm/sole contact sites
│   └── ik_configs/
│       └── smplx_to_alex.json       # IK match table: human role → Alex body, weights
│
├── assets/alex/                     # Alex robot model files
│   ├── alex_floating_base.urdf      # URDF model
│   ├── alex_floating_base_with_sites.xml  # MuJoCo XML with IK target sites
│   └── meshes/                      # Visual/collision mesh files
│
├── data/
│   ├── raw/                         # Raw MoCap files (MVNX, FBX)
│   │   ├── inhouse/
│   │   └── vtech_nmp/
│   ├── processed/                   # Intermediate canonical NPZ files
│   └── source_motions/
│
├── scripts/                         # One-off pipeline scripts (not importable library code)
│   ├── solve_fbx_canonical_alex_posori_qp_fresh.py   # Main IK solver (position + orientation)
│   ├── solve_mvnx_alex_motion.py                     # MVNX input IK solve
│   ├── solve_canonical_alex_motion.py                # Canonical NPZ input IK solve
│   ├── build_canonical_orientation_frames_fresh.py   # Pre-process orientation frames
│   ├── build_fbx_canonical_human.py                  # FBX → canonical human
│   ├── export_alex_retarget_npz_to_ihmc_json.py      # Export results for IHMC controller
│   ├── validate_alex_ik_config.py                    # Config validation utilities
│   ├── diagnostics/                                  # Debug and analysis scripts
│   └── visualization/
│       ├── render_alex_qp_direct_mp4_fresh.py        # MP4 render of IK output
│       └── render_fbx_kinematic_v2_alex_visual_mesh_gif.py  # GIF render
│
├── experiments/                     # Experiment logs / scratch runs
├── outputs/                         # IK solver output NPZ files
└── pyproject.toml
```

---

## Key Concepts

### Canonical Human Representation
All source motion formats (MVNX from Xsens suits, FBX from motion capture software) are first converted to a **canonical human frame** — a dictionary of 15 semantic body roles plus 4 auxiliary contact sites (palms, soles). Each entry has a `pos: [x,y,z]` and `quat_wxyz: [w,x,y,z]`. This decouples the source format from the robot retargeting logic.

### Morphology Delta Scaling
Before feeding positions to IK, limb-length differences between the human performer and Alex are measured at a rest/T-pose frame. Per-limb scale ratios are computed and applied to motion *deltas* (relative displacements from rest), so the robot moves proportionally rather than trying to reach physically impossible human-scale positions.

### MuJoCo QP IK Solver
The IK solver runs inside a MuJoCo simulation of Alex. It uses a **Quadratic Programming (QP)** formulation to find joint angles that minimize weighted errors between:
- Target body positions (from scaled canonical human frame)
- Target body orientations (quaternion)
- Optional contact site positions (palm/sole targets for contact-aware motions)

The IK config (`smplx_to_alex.json`) defines the human-role → Alex-body mapping, per-target position/orientation weights, and offsets.

### Alex Robot
Alex is an IHMC-developed bipedal humanoid robot. The MuJoCo model (`alex_floating_base_with_sites.xml`) is a floating-base version (pelvis is a free joint) with named **sites** added at palm and sole locations for end-effector targeting. Robot configs are JSON files listing body names, joint names, and site locations.

### Output Formats
- **NPZ**: NumPy archive of joint angle trajectories + metadata, used internally
- **IHMC JSON**: Exported format for the IHMC robot controller to execute the motion

---

## Data Sources
- **MVNX**: Xsens MVN motion capture suit output (XML-based), segment orientations in global frame
- **FBX**: Standard 3D animation format from motion capture post-processing
- Data sources include in-house recordings and VTech NMP dataset

---

## Tech Stack
- **Python 3.10+**
- **MuJoCo** — physics simulation and IK solving
- **NumPy** — numerical computation throughout
- **URDF / MuJoCo XML** — robot model formats
- Custom XML parsing for MVNX

---

## Current Branch: `feature/alex-compatible-canonical-ik`
Active work adding canonical IK scripts that are fully compatible with Alex's kinematic structure, including fresh position+orientation QP solvers and world-delta formulations for floating-base motion.
