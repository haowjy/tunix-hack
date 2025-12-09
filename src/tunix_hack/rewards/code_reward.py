"""Code reward function for GRPO training."""

from tunix_hack.utils.xml_parsing import extract_tag


def code_reward(output: str, tests: list[str] | None = None) -> float:
    """Compute reward for code problems.

    Args:
        output: Model output with <reasoning> and <answer> tags.
        tests: Optional list of test cases to run (not implemented yet).

    Returns:
        Reward score between 0 and 1.
    """
    # Extract tags
    answer = extract_tag(output, "answer")
    reasoning = extract_tag(output, "reasoning")

    # Basic structural checks
    has_code = "def " in answer or "class " in answer or "return" in answer
    has_reasoning = len(reasoning) > 20
    not_too_long = len(reasoning) < 800

    # For now, just check structure (test execution not implemented)
    structure_score = float(has_code and has_reasoning and not_too_long)

    # TODO: Add actual test execution
    # if tests:
    #     pass_count = run_tests(answer, tests)
    #     test_score = pass_count / len(tests)
    #     return 0.7 * test_score + 0.3 * structure_score

    return structure_score
