"""Multi-domain reward function for mixed GRPO training."""

from typing import Any

from tunix_hack.rewards.math_reward import math_reward
from tunix_hack.rewards.code_reward import code_reward
from tunix_hack.rewards.creative_reward import creative_reward


def science_reward(output: str, reference_answer: str) -> float:
    """Compute reward for science problems.

    Uses simple overlap with reference answer.

    Args:
        output: Model output with <reasoning> and <answer> tags.
        reference_answer: The reference answer to compare against.

    Returns:
        Reward score between 0 and 1.
    """
    from tunix_hack.utils.xml_parsing import extract_tag

    answer = extract_tag(output, "answer")
    reasoning = extract_tag(output, "reasoning")

    # Simple word overlap
    answer_words = set(answer.lower().split())
    reference_words = set(reference_answer.lower().split())

    if not reference_words:
        return 0.0

    overlap = len(answer_words & reference_words) / len(reference_words)

    # Structure bonus
    has_reasoning = len(reasoning) > 20
    structure_bonus = float(has_reasoning)

    return 0.7 * overlap + 0.3 * structure_bonus


def multi_domain_reward(output: str, example: dict[str, Any]) -> float:
    """Route to appropriate domain-specific reward function.

    Args:
        output: Model output with <reasoning> and <answer> tags.
        example: Example dict with "domain" key and domain-specific fields:
            - math: requires "answer" field
            - code: optionally uses "tests" field
            - science: requires "reference_answer" field
            - creative: requires "prompt" field

    Returns:
        Reward score between 0 and 1.
    """
    domain = example.get("domain", "")

    if domain == "math":
        return math_reward(output, example["answer"])
    elif domain == "code":
        return code_reward(output, example.get("tests"))
    elif domain == "science":
        return science_reward(output, example["reference_answer"])
    elif domain == "creative":
        return creative_reward(output, example["prompt"])
    else:
        return 0.0
