"""Data preprocessing utilities."""

from typing import Any


def preprocess_for_sft(examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Preprocess examples for supervised fine-tuning.

    Converts examples to input/output pairs with <reasoning>/<answer> format.

    Args:
        examples: Raw examples with domain-specific fields.

    Returns:
        List of {"input": ..., "output": ...} dicts.
    """
    processed = []
    for ex in examples:
        # Format depends on domain
        domain = ex.get("domain", "")

        if domain == "math":
            input_text = ex["question"]
            output_text = f"<reasoning>{ex.get('reasoning', '')}</reasoning><answer>{ex['answer']}</answer>"
        elif domain == "creative":
            input_text = ex["prompt"]
            output_text = f"<reasoning>{ex.get('reasoning', '')}</reasoning><answer>{ex.get('story', '')}</answer>"
        else:
            # Generic format
            input_text = ex.get("input", ex.get("question", ex.get("prompt", "")))
            output_text = f"<reasoning>{ex.get('reasoning', '')}</reasoning><answer>{ex.get('answer', '')}</answer>"

        processed.append({"input": input_text, "output": output_text})

    return processed


def preprocess_for_grpo(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preprocess examples for GRPO training.

    Args:
        examples: Raw examples with domain-specific fields.

    Returns:
        List of examples with "prompt" field for generation.
    """
    processed = []
    for ex in examples:
        domain = ex.get("domain", "")

        if domain == "math":
            prompt = ex["question"]
        elif domain == "creative":
            prompt = ex["prompt"]
        else:
            prompt = ex.get("input", ex.get("question", ex.get("prompt", "")))

        processed.append({**ex, "prompt": prompt})

    return processed
