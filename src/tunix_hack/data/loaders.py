"""Dataset loading utilities."""

from typing import Any


def load_dataset(name: str, split: str = "train") -> list[dict[str, Any]]:
    """Load a dataset by name.

    Args:
        name: Dataset name (e.g., "gsm8k", "mbpp", "creative").
        split: Dataset split to load.

    Returns:
        List of examples.
    """
    # TODO: Implement actual dataset loading
    # For now, return empty list as placeholder
    raise NotImplementedError(f"Dataset loading for '{name}' not implemented yet")
