"""Math reward function for GRPO training."""

import re

from tunix_hack.utils.xml_parsing import extract_tag


def normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison.

    Args:
        answer: The answer string to normalize.

    Returns:
        Normalized answer string.
    """
    # Remove whitespace
    answer = answer.strip()
    # Remove common formatting
    answer = answer.replace(",", "").replace("$", "").replace("%", "")
    # Try to extract just the number
    match = re.search(r"-?\d+\.?\d*", answer)
    if match:
        return match.group()
    return answer.lower()


def math_reward(output: str, ground_truth_answer: str) -> float:
    """Compute reward for math problems.

    Args:
        output: Model output with <reasoning> and <answer> tags.
        ground_truth_answer: The correct answer.

    Returns:
        Reward score between 0 and 1.
    """
    # Extract tags
    answer = extract_tag(output, "answer")
    reasoning = extract_tag(output, "reasoning")

    # Correctness check
    correct = float(normalize_math_answer(answer) == normalize_math_answer(ground_truth_answer))

    # Structural sanity (no deep semantics yet)
    has_reasoning = len(reasoning) > 20
    not_too_long = len(reasoning) < 800
    structure_bonus = float(has_reasoning and not_too_long)

    # Weight correctness higher
    return 0.8 * correct + 0.2 * structure_bonus
