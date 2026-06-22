# Alex assets

This folder is intentionally local-only.

Do not commit IHMC/Alex URDF, mesh, or generated MuJoCo-ready asset files to a public repository unless you have explicit permission.

Expected local generated files include:

- alex_mujoco_ready.urdf
- alex_floating_base.urdf
- alex_mujoco_model_summary.json
- meshes/
- source/heh_original.urdf

Regenerate the cleaned MuJoCo asset locally using:

    python scripts/prepare_alex_mujoco_assets.py

Regenerate the floating-base model locally using:

    python scripts/prepare_alex_floating_base_model.py
