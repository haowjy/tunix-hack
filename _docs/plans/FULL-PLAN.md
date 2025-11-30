Here’s a concrete, end‑to‑end plan you can follow, tailored to:

- Your goal: **“correct, useful reasoning traces + good answers”**  
- Your resources: **RTX 3090 locally + 9‑hour Kaggle TPU session**  
- The competition constraints: **Gemma + Tunix + GRPO + `<reasoning>/<answer>` format**

I’ll structure this as:

1. Overall strategy
2. What to do first (local experiments)
3. How to “train” the model step‑by‑step
4. How to handle creative‑writing data specifically
5. How to turn this into the Kaggle submission

You can treat this as a checklist.

---

## 1. Overall strategy

High level:

- Use your **3090 for exploration and model selection**.
- Use **Kaggle only for the final, reproducible run** (and maybe one small dry run).
- Start with **simple, verifiable rewards** (math/code) to get GRPO + Tunix working.
- Then bring in **creative writing**, where “correct reasoning” is more about *coherent, grounded explanation*, not absolute truth.

Architecturally:

- **Phase 1**: Train a **single mixed‑domain LoRA** (math + code + science + summarization + creative).
- **Phase 2 (optional)**: If you still have time/energy, experiment locally with:
  - A math‑focused LoRA and
  - A creative‑writing‑focused LoRA  
  …and see if merging helps. Only bring this into Kaggle if it’s clearly better.

---

## 2. What to do first (local experiments)

Before touching Kaggle, get three things working on your 3090:

### 2.1. Baseline SFT with `<reasoning>/<answer>` (no RL yet)

Goal: make sure the model can obey the format and produce *some* reasoning and answers.

Steps:

1. Pick a base model:
   - Gemma 2B (or Gemma 3 1B if that’s what Tunix expects).
2. Collect a **small supervised dataset** (~200–400 examples total):
   - Math: GSM8K style:
     - Input: problem
     - Output: `<reasoning>step-by-step solution</reasoning><answer>final number</answer>`
   - Code: simple MBPP‑like problems with brief explanations.
   - Creative: a handful of prompts where you manually write:
     - A short “why this story works / what I’m going to do” reasoning
     - Then the story as `<answer>…</answer>`.
3. Fine‑tune with standard LoRA SFT (no RL, no GRPO) on this tiny set.

You want to confirm:

- Model always outputs both tags.
- Reasoning is not empty.
- It doesn’t hallucinate random XML.

If this fails, you don’t want to be debugging GRPO on top of it.

---

### 2.2. Get GRPO working on a **pure math task** (local)

Because math correctness is binary and easy to score, it’s the cleanest sandbox.

Dataset (local copy of):

- A few hundred GSM8K problems with known correct answers.
- You *don’t* need their written solutions for RL; you’ll just check the final answer.

Training target:

- Prompt: the problem.
- Desired output:  
  `<reasoning>…</reasoning><answer>…</answer>`

Reward function (keep it simple first):

```python
def extract_tag(text, tag):
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1:
        return ""
    return text[start + len(tag) + 2 : end].strip()

def math_reward(output, ground_truth_answer):
    # 1) Extract
    answer = extract_tag(output, "answer")
    reasoning = extract_tag(output, "reasoning")

    # 2) Correctness
    correct = float(normalize_math_answer(answer) == normalize_math_answer(ground_truth_answer))

    # 3) Structural sanity (no deep semantics yet)
    has_reasoning = len(reasoning) > 20
    not_too_long = len(reasoning) < 800

    structure_bonus = float(has_reasoning and not_too_long)

    # Weight correctness higher
    return 0.8 * correct + 0.2 * structure_bonus
```

Your experiment:

- Run **short GRPO training** (~1–2 hours) on your 3090.
- Inspect:
  - Does accuracy on a small held‑out math set go up?
  - Does reasoning become more step‑by‑step and less random?
  - Does the model reliably produce the XML format?

Once this pipeline works for math, you know:

- LoRA + GRPO + reward function plumbing are sound.
- You’re not going to waste Kaggle hours on basic bugs.

---

### 2.3. Prototype creative‑writing data & a *basic* reward

Now, answer your question: **“what dataset and how to train for creative writing reasoning?”**

You have a few realistic options:

#### Option A: Use an existing reasoning‑heavy writing dataset

There’s a small but very relevant HF dataset:

- `rekrek/reasoning-engaging-story` on Hugging Face  
  It has fields like:
  - `story_*_reasoning`
  - `story_*_solution`
  - Multi‑step story planning, symbolism, characters, world‑building, etc. \[source: HF dataset listing\]

Pros:
- Already has **explicit reasoning + story** pairs.
- You can map:
  - `*_reasoning` → `<reasoning>…</reasoning>`
  - `*_solution` → `<answer>…</answer>`

Cons:
- Only ~110 rows → must be combined with your own or other sources.

You can combine that with:

- Reddit **WritingPrompts** (for prompts + human stories) \[Fan et al. 2018\]
- Or generate your own reasoning using Gemini / GPT‑4.

#### Option B: Synthetic creative data with your own style

Process:

1. Collect prompts:
   - From WritingPrompts, or your own list (maybe 150–300).
2. For each prompt, use a strong API model (Gemini 1.5 Flash is fine) with a prompt like:

   > Given this writing prompt:  
   > [PROMPT]  
   > First, think step by step about:  
   > – What tone and style to use  
   > – Who the main character is and what they want  
   > – How the story should begin, develop, and end  
   > – What themes or symbolism you want to highlight  
   > Then output in this format:  
   > `<reasoning>...</reasoning><answer>...</answer>`

3. Manually skim a subset and discard garbage.

Now you have a supervised dataset of:

```text
input:   PROMPT
output:  <reasoning>narrative planning, motivations, consistency checks...</reasoning>
         <answer>the story itself</answer>
```

You can:

- First do SFT on this creative subset (to get style & format).
- Then optionally add RL on top (if you define a reward).

#### Creative-writing reward (keep it practical)

You won’t perfectly measure “correct reasoning”, but you can push toward *coherent, grounded* reasoning:

```python
def creative_reward(output, prompt):
    reasoning = extract_tag(output, "reasoning")
    story = extract_tag(output, "answer")

    # 1) Non-empty, not insane
    if len(reasoning) < 50 or len(story) < 150:
        return 0.0

    # 2) Simple signals that it's actually "reasoning about writing"
    lower = reasoning.lower()
    mentions_characters = any(w in lower for w in ["character", "protagonist", "motivation"])
    mentions_plot = any(w in lower for w in ["because", "therefore", "conflict", "arc"])
    mentions_style = any(w in lower for w in ["tone", "style", "pacing"])

    reasoning_score = (
        0.4 * float(mentions_characters)
        + 0.4 * float(mentions_plot)
        + 0.2 * float(mentions_style)
    )

    # 3) Rough coherence: avoid wild repetition
    unique_sentences = len(set(s.strip() for s in story.split('.') if s.strip()))
    total_sentences = max(1, story.count('.'))
    diversity = unique_sentences / total_sentences

    coherence_score = float(0.6 <= diversity <= 1.0)

    return 0.6 * reasoning_score + 0.4 * coherence_score
```

This does *not* prove the reasoning is philosophically “correct”, but it pushes the model to:

- Explicitly talk about characters, plot, and style in `<reasoning>`.
- Produce non‑degenerate stories in `<answer>`.

You can later refine by training a small **reward model** (e.g. using LitBench or LiteraryTaste preferences) if you want, but I’d start with this.

---

## 3. How to “train it” step‑by‑step

Here’s an order of experiments that builds up complexity safely.

### Step 0 – Environment sanity (local)

- Load Gemma + a minimal LoRA.
- Generate text with `<reasoning>/<answer>` template prompt.
- Confirm tokenization + max sequence lengths are okay.

### Step 1 – Supervised finetuning for format & basic reasoning

On your 3090:

1. Build a **combined SFT dataset** (~400–800 examples total):
   - Math + code + science + summarization + creative.
   - All outputs in the required XML‑ish format.

2. SFT with LoRA:
   - `r=16`, `alpha=32`, `target_modules=['q_proj', 'v_proj']`.
   - 1–2 epochs, small LR (e.g. 1e‑5).

3. Evaluate qualitatively:
   - Does the model always produce both tags?
   - Are the reasoning parts halfway sensible across domains?

### Step 2 – GRPO on math only (local)

- Use the math reward from §2.2.
- Run a small GRPO training (~5–10k steps).
- Track:
  - Math accuracy vs. baseline SFT model.
  - Reasoning length and “step‑iness”.

If this doesn’t show improvement, fix this *before* moving on.

### Step 3 – GRPO on mixed domains (local)

Now extend to a small **mixed RL dataset**:

- Math: reward = correctness + structure.
- Code: reward = test pass (if you can run simple unit tests locally) + explanation length.
- Science: reward = overlap with reference answer (string or embedding similarity) + explanation structure.
- Creative: heuristic reward from §2.3.

Implement:

```python
def multi_domain_reward(output, example):
    domain = example["domain"]
    if domain == "math":
        return math_reward(output, example["answer"])
    elif domain == "code":
        return code_reward(output, example["tests"])
    elif domain == "science":
        return science_reward(output, example["reference_answer"])
    elif domain == "creative":
        return creative_reward(output, example["prompt"])
    else:
        return 0.0
```

Run GRPO with:

- Fewer steps (you just want to see if it moves the needle).
- Small KL penalty (beta ~ 0.02–0.05) to avoid drift from SFT.

Inspect:

- Does math performance stay good?
- Does creative reasoning become more explicit (more mentions of character/plot/etc.)?
- Any collapse (model starts ignoring some domains)? If so, rebalance sampling.

---

## 4. Where to start experimenting (in practice)

Given your time and GPU:

**Phase A – This week**

1. Implement SFT dataset + training loop (Step 1).
2. Implement math GRPO sandbox (Step 2).
3. Run both; log everything; keep checkpoints.

**Phase B – After SFT + math GRPO feel stable**

4. Build a small creative‑writing dataset:
   - 100–200 prompts, reasoning+story from Gemini / GPT‑4, plus some of your own.
5. Integrate creative into SFT.
6. Try a tiny mixed‑domain GRPO run (just to check nothing explodes).

If, at any point, RL makes things worse across the board, you can still submit an **SFT‑only reasoning model**; the competition doesn’t say RL is mandatory, just encouraged.

---

## 5. How to map this to Kaggle (final run)

Once you’ve found:

- A base SFT checkpoint that works well.
- GRPO hyperparameters that seem stable.
- A training set small enough to run in ~6–7 hours on TPU.

Then your Kaggle notebook should:

1. **Download / load data**
   - All your datasets must be either:
     - Public (HF, etc.), or
     - Included as Kaggle datasets you upload.

2. **Load base Gemma + LoRA config**

3. **Optionally**:
   - Load your best local SFT checkpoint as the **starting point** (if rules allow shipping weights by uploading them as a Kaggle dataset).
   - If not allowed, re‑run a light SFT in‑notebook (with documented hyperparams).

4. **Run a single, well‑tuned GRPO training session**:
   - Mixed domains.
   - 1–2 epochs over a **modest** dataset (few hundred examples total, multiple rollouts).
   - Log rewards, KL, and maybe a small eval every N steps.

5. **Evaluate**:
   - On held‑out test splits per domain.
   - Print a few example `<reasoning>/<answer>` pairs in the notebook.

6. **Save final LoRA / model**:
   - To `/kaggle/working`.
   - Show in notebook how to reload it and run inference with a single code cell.

---

If you want, next step I can:

- Propose concrete hyperparameters (dataset sizes, rollout counts) sized to **definitely** fit in 9 hours.
- Or sketch out a toy training script (pseudo‑code or real Python) for SFT + GRPO that you can adapt into your own repo.