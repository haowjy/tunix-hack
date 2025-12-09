# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Training a Gemma model with GRPO (Group Relative Policy Optimization) using Tunix to produce reasoning traces and answers in `<reasoning>/<answer>` format. This is a JAX/Flax project (not PyTorch) targeting a Kaggle competition.

## Commands

```bash
# Setup
uv venv && uv sync

# Run notebooks
jupyter lab
```

## Architecture

### Source Code (`src/tunix_hack/`)

- **`rewards/`** - Domain-specific reward functions for GRPO training
  - `math_reward.py` - Binary correctness (80%) + structure bonus (20%)
  - `code_reward.py` - Structure checks for code output
  - `creative_reward.py` - Heuristic scoring for character/plot/style mentions
  - `multi_domain.py` - Routes to domain-specific rewards based on `example["domain"]`

- **`training/`** - SFT and GRPO training loops (to be implemented with Tunix)
  - `sft.py` - Supervised fine-tuning
  - `grpo.py` - GRPO training

- **`data/`** - Dataset loading and preprocessing
  - `loaders.py` - Dataset loading by name
  - `preprocessing.py` - Converts to `<reasoning>/<answer>` format

- **`utils/`** - Utilities
  - `xml_parsing.py` - Tag extraction with `extract_tag()` and `has_valid_format()`

## Guiding Principles for Development

ALWAYS FOLLOW SOLID PRINCIPLES.

Then, these principles can also help you make architectural decisions and other development tasks:

1. **Start Simple, Stay Simple**
   - Write the simplest thing that could work
   - Add complexity only when necessary
   - Regularly refactor to remove unnecessary complexity

2. **Make Correctness Obvious**
   - Code should make bugs impossible or obvious
   - Use types to prevent invalid states
   - Fail fast and loudly (don't swallow errors)

3. **One Thing At A Time**
   - Don't optimize and add features simultaneously
   - Test each change before moving on
   - Small, incremental changes are easier to debug

4. **Comment the "Weird" and the "WHY"**
   - anything that is not obvious, comment why.
   - If it needs a guard, comment why
   - If it prevents a race, explain the race
   - If you had to debug it, future you will too
   - etc.

5. **Extensible** - Design for extensibility.

6. **Keep Documentation Up-to-Date** - Update documentation AFTER finalizing changes. See "Feature Documentation Sync Rule" for feature documentation workflow.

7. **Keep the code clean** - keep the code clean and readable, as the code grows, it will become more difficult to understand, its easier to refactor now than later (make sure to delete dead code as well).

### Data Format

All outputs use XML-style tags:
```
<reasoning>step-by-step explanation</reasoning><answer>final answer</answer>
```

### Key Dependencies

- **JAX/Flax** for model training (not PyTorch)
- **Tunix** (`google-tunix`) for GRPO implementation
- **Transformers** for loading Gemma models in Flax format

## Resources

- **Local**: RTX 3090 for exploration
- **Kaggle**: 9-hour TPU session for final runs (JAX is TPU-optimized)
