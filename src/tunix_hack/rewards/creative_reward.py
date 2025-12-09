"""Creative writing reward function for GRPO training."""

from tunix_hack.utils.xml_parsing import extract_tag


def creative_reward(output: str, prompt: str) -> float:
    """Compute reward for creative writing.

    This does not prove the reasoning is philosophically "correct", but it pushes
    the model to explicitly talk about characters, plot, and style in <reasoning>,
    and produce non-degenerate stories in <answer>.

    Args:
        output: Model output with <reasoning> and <answer> tags.
        prompt: The original writing prompt.

    Returns:
        Reward score between 0 and 1.
    """
    reasoning = extract_tag(output, "reasoning")
    story = extract_tag(output, "answer")

    # Non-empty, not insane
    if len(reasoning) < 50 or len(story) < 150:
        return 0.0

    # Simple signals that it's actually "reasoning about writing"
    lower = reasoning.lower()
    mentions_characters = any(w in lower for w in ["character", "protagonist", "motivation"])
    mentions_plot = any(w in lower for w in ["because", "therefore", "conflict", "arc"])
    mentions_style = any(w in lower for w in ["tone", "style", "pacing"])

    reasoning_score = (
        0.4 * float(mentions_characters) + 0.4 * float(mentions_plot) + 0.2 * float(mentions_style)
    )

    # Rough coherence: avoid wild repetition
    sentences = [s.strip() for s in story.split(".") if s.strip()]
    unique_sentences = len(set(sentences))
    total_sentences = max(1, len(sentences))
    diversity = unique_sentences / total_sentences

    coherence_score = float(0.6 <= diversity <= 1.0)

    return 0.6 * reasoning_score + 0.4 * coherence_score
