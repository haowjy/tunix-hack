"""Checkpoint discovery and restoration utilities.

This module is used by both inference (loading trained weights) and
training (resuming from checkpoints).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
from flax import nnx
import orbax.checkpoint as ocp


def list_checkpoints(ckpt_root: Path) -> list[dict[str, str]]:
    """Scan checkpoint directory for all available checkpoints.

    Expects checkpoint structure:
        ckpt_root/
            {run_name}/
                actor/
                    {step}/
                        model_params/

    Args:
        ckpt_root: Root directory containing checkpoint runs.

    Returns:
        List of checkpoint info dicts with keys:
        - "run": Run name
        - "step": Step number (as string)
        - "path": Full path to model_params directory
    """
    checkpoints = []

    if not ckpt_root.exists():
        return checkpoints

    for run_dir in sorted(ckpt_root.iterdir()):
        if not run_dir.is_dir():
            continue

        actor_dir = run_dir / "actor"
        if not actor_dir.exists():
            continue

        for step_dir in sorted(
            actor_dir.iterdir(),
            key=lambda x: int(x.name) if x.name.isdigit() else 0,
        ):
            if not step_dir.is_dir():
                continue

            model_params = step_dir / "model_params"
            if model_params.exists():
                checkpoints.append({
                    "run": run_dir.name,
                    "step": step_dir.name,
                    "path": str(model_params),
                })

    return checkpoints


def find_checkpoint(
    ckpt_root: Path,
    run_name: str,
    step: int,
) -> Path | None:
    """Find a specific checkpoint by run name and step.

    Args:
        ckpt_root: Root directory containing checkpoint runs.
        run_name: Name of the training run.
        step: Training step number.

    Returns:
        Path to the checkpoint's model_params directory, or None if not found.
    """
    checkpoints = list_checkpoints(ckpt_root)
    for ckpt in checkpoints:
        if ckpt["run"] == run_name and ckpt["step"] == str(step):
            return Path(ckpt["path"])
    return None


def restore_checkpoint(model: Any, checkpoint_path: Path) -> None:
    """Restore LoRA parameters from a checkpoint into the model.

    This modifies the model in-place by loading the saved LoRA parameters.

    Args:
        model: A model with LoRA adapters applied (from qwix.apply_lora_to_model).
        checkpoint_path: Path to the model_params directory.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Get abstract state structure for LoRA parameters
    # (checkpoint only contains LoRA params, not base model weights)
    abs_lora_state = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        nnx.state(model, nnx.LoRAParam),
    )

    # Restore checkpoint
    checkpointer = ocp.StandardCheckpointer()
    restored_lora_params = checkpointer.restore(
        str(checkpoint_path),
        target=abs_lora_state,
    )

    # Update model with restored LoRA parameters
    nnx.update(
        model,
        jax.tree.map(
            lambda a, b: b,
            nnx.state(model, nnx.LoRAParam),
            restored_lora_params,
        ),
    )
