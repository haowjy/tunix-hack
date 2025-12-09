"""Model loading utilities for Gemma models with optional LoRA."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import jax
from huggingface_hub import snapshot_download
import qwix

from tunix.models.gemma3 import model as gemma3_model
from tunix.models.gemma3 import params_safetensors as gemma3_safetensors
from tunix.generate import tokenizer_adapter as tokenizer_lib

if TYPE_CHECKING:
    from tunix.generate.tokenizer_adapter import Tokenizer

# Default LoRA target modules for Gemma models
DEFAULT_LORA_MODULES = (
    ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
    ".*attn_vec_einsum"
)


def load_tokenizer(model_path: str) -> "Tokenizer":
    """Load tokenizer from a model path.

    Args:
        model_path: Path to the model directory (from snapshot_download or local).

    Returns:
        Tokenizer instance.
    """
    return tokenizer_lib.Tokenizer(
        tokenizer_path=os.path.join(model_path, "tokenizer.model")
    )


def load_model(
    model_id: str,
    mesh: jax.sharding.Mesh,
    *,
    use_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_modules: str = DEFAULT_LORA_MODULES,
) -> tuple:
    """Load a Gemma model from HuggingFace, optionally with LoRA.

    Args:
        model_id: HuggingFace model ID (e.g., "google/gemma-3-1b-it").
        mesh: JAX mesh for sharding.
        use_lora: Whether to apply LoRA adapters.
        lora_rank: LoRA rank (only used if use_lora=True).
        lora_alpha: LoRA alpha scaling factor (only used if use_lora=True).
        lora_modules: Regex pattern for modules to apply LoRA to.

    Returns:
        Tuple of (model, model_config, model_path) where:
        - model: The loaded model (with or without LoRA)
        - model_config: The model configuration
        - model_path: Path to the downloaded model (for tokenizer loading)
    """
    # Download model from HuggingFace
    model_path = snapshot_download(model_id)

    # Get model config based on model ID
    # TODO: support other model sizes when needed
    model_config = gemma3_model.ModelConfig.gemma3_1b()

    # Load base model from safetensors
    base_model = gemma3_safetensors.create_model_from_safe_tensors(
        model_path, model_config, mesh
    )

    if not use_lora:
        return base_model, model_config, model_path

    # Apply LoRA adapters
    lora_provider = qwix.LoraProvider(
        module_path=lora_modules,
        rank=lora_rank,
        alpha=lora_alpha,
    )
    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(base_model, lora_provider, **model_input)

    return lora_model, model_config, model_path
