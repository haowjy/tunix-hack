# Tunix Integration Notes

## About Tunix

[Tunix](https://github.com/google/tunix) is a JAX-native LLM post-training library from Google that provides:

- **Supervised Fine-Tuning** (SFT)
- **Reinforcement Learning**: PPO, GRPO, GSPO-token
- **Preference Fine-Tuning**: DPO
- **Knowledge Distillation**

## Key Differences from PyTorch

Since Tunix is JAX-based, this project uses:

- **JAX/Flax** instead of PyTorch for model training
- **Flax NNX** for model architecture (not PyTorch nn.Module)
- **JAX transformations** (jax.jit, jax.pmap, etc.) for optimization
- **TPU-optimized** training (great for Kaggle TPU sessions)

## Installation

Tunix is installed via:

```bash
uv add "google-tunix[prod]"
```

Or from GitHub (latest):
```bash
uv add "git+https://github.com/google/tunix"
```

## Model Loading

Gemma models can be loaded in Flax format:

```python
from transformers import FlaxGemmaForCausalLM, AutoTokenizer

model = FlaxGemmaForCausalLM.from_pretrained("google/gemma-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
```

## Training with Tunix

Tunix provides GRPO (Group Relative Policy Optimization) which is what we need for this project. The training loop will use:

- JAX-based model parameters
- Tunix's GRPO implementation
- Reward functions (defined in `src/rewards/`)

## Resources

- [Tunix GitHub](https://github.com/google/tunix)
- [Tunix Examples](https://github.com/google/tunix/tree/main/examples)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)

## Examples from Tunix Repo

Tunix provides examples for:
- PEFT Gemma with QLoRA
- Training Gemma on grade school Math problems using GRPO
- Logit Distillation using Gemma models
- Training Llama3 or Qwen2 using GRPO with SGLang-Jax rollout

These examples will be helpful references when implementing our training scripts.

