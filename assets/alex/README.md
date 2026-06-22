# Alex assets

This folder is intentionally local-only.

Do not commit IHMC/Alex URDF, mesh, or generated MuJoCo-ready asset files to a public repository unless you have explicit permission.

Expected local generated files include:

- alex_mujoco_ready.urdf
- alex_mujoco_model_summary.json
- meshes/
- source/heh_original.urdf

Regenerate them locally using:

    python scripts/prepare_alex_mujoco_assets.py
