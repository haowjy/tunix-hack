"""Text generation utilities using Tunix sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tunix.generate import sampler as sampler_lib

if TYPE_CHECKING:
    import jax
    from tunix.generate.tokenizer_adapter import Tokenizer

# Default system prompt for math problems
DEFAULT_SYSTEM_PROMPT = """You are a math tutor. Solve the problem step by step.
Format your response EXACTLY as:
<reasoning>your step-by-step solution</reasoning><answer>final numerical answer only</answer>"""

# Prompt template for Gemma chat format
PROMPT_TEMPLATE = "<start_of_turn>user\n{system_prompt}\n\n{question}<end_of_turn>\n<start_of_turn>model\n"


def create_sampler(
    model: Any,
    tokenizer: "Tokenizer",
    model_config: Any,
    *,
    max_cache_size: int = 640,
) -> sampler_lib.Sampler:
    """Create a sampler for text generation.

    Args:
        model: The loaded model (base or with LoRA).
        tokenizer: Tokenizer instance.
        model_config: Model configuration with num_layers, num_kv_heads, head_dim.
        max_cache_size: Maximum size of the KV cache.

    Returns:
        Configured Sampler instance.
    """
    return sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=max_cache_size,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )


def generate(
    sampler: sampler_lib.Sampler,
    mesh: "jax.sharding.Mesh",
    question: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    max_tokens: int = 256,
) -> str:
    """Generate a response for a question.

    Args:
        sampler: Sampler instance from create_sampler().
        mesh: JAX mesh for sharding.
        question: The question to answer.
        system_prompt: System prompt to prepend.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        max_tokens: Maximum tokens to generate.

    Returns:
        Generated response text.
    """
    prompt = PROMPT_TEMPLATE.format(
        system_prompt=system_prompt,
        question=question,
    )

    with mesh:
        out = sampler(
            input_strings=[prompt],
            max_generation_steps=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            echo=False,
            eos_tokens=[1, 106],  # EOS tokens for Gemma
        )

    return out.text[0]
