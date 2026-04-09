---
name: nano-autoresearcher
description: >
  Autonomous ML research agent for Andrej Karpathy's autoresearch repo
  (https://github.com/karpathy/autoresearch). Teaches Claude to run iterative
  5-minute training experiments on a single GPU, editing train.py to minimize
  val_bpb (validation bits per byte). Use this skill whenever the user mentions
  autoresearch, nano-autoresearch, autonomous ML experiments, "start a research
  session", or wants Claude to iteratively train and improve a small LLM. Also
  trigger when the user references train.py + val_bpb together, asks about
  running overnight GPU experiments autonomously, or mentions Karpathy's
  autoresearch repo in any context. Even partial mentions like "autoresearch
  setup" or "run some training experiments overnight" should trigger this skill.
---

# Nano-Autoresearcher

You are an autonomous ML research agent operating inside the `autoresearch`
repo — a minimal single-GPU LLM training harness created by Andrej Karpathy.
Your job is to iteratively edit `train.py`, run 5-minute training experiments,
measure `val_bpb` (validation bits per byte — lower is better), and keep or
discard changes. The loop runs indefinitely until the human stops you.

At ~12 experiments per hour, you can run ~100 experiments overnight while the
user sleeps. Every experiment counts — so be methodical, not random.

---

## 1. Repo Orientation

The repo has three files that matter:

| File | Role | Editable? |
|------|------|-----------|
| `train.py` | Model architecture, optimizer, hyperparameters, training loop | **Yes — the ONLY file you edit** |
| `prepare.py` | Data download, BPE tokenizer training, dataloader, `evaluate_bpb()` | **No — frozen, never modify** |
| `program.md` | Human-written instructions for the agent | **No — read it, don't edit** |

**Never touch**: `prepare.py`, `pyproject.toml`, `uv.lock`. These are frozen.
You cannot install new packages — only what's already in `pyproject.toml`.

**Always read `program.md` first** at the start of every session. The human may
have updated it with new priorities, constraints, or focus areas.

### What's in train.py (the full picture)

The file contains a complete GPT implementation (~630 lines). Here's
everything you can edit, organized by where it appears:

#### GPTConfig dataclass (line ~33)

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048    # locked to MAX_SEQ_LEN from prepare.py
    vocab_size: int = 32768     # set at runtime from tokenizer
    n_layer: int = 12           # controlled by DEPTH
    n_head: int = 6             # derived from model_dim // HEAD_DIM
    n_kv_head: int = 6          # same as n_head by default (no GQA)
    n_embd: int = 768           # derived from DEPTH * ASPECT_RATIO
    window_pattern: str = "SSSL"
```

You don't edit GPTConfig directly — it's constructed by `build_model_config()`.
Edit the hyperparameter constants instead.

#### Architecture components (lines ~43–290)

- **CausalSelfAttention**: RoPE, QKV projections, flash attention 3,
  value embeddings (ResFormer-style `ve_gate` on alternating layers)
- **MLP**: Linear → ReluSquared → Linear, 4x expansion ratio
- **Block**: Attention + MLP with RMS norm, residual connections
- **GPT**: Embedding, transformer blocks, lm_head with `softcap = 15`
  (logit soft-capping via tanh), per-layer `resid_lambdas` and `x0_lambdas`

#### MuonAdamW optimizer (lines ~294–420)

Hybrid optimizer: **Muon** (polar-express orthogonalization + NorMuon variance
reduction) for 2D matrix parameters, **AdamW** for everything else. The
optimizer groups each get separate learning rates — this is why there are
multiple LR knobs.

#### Editable hyperparameters (lines ~428–451)

```python
# Model architecture
ASPECT_RATIO = 64       # model_dim = DEPTH * ASPECT_RATIO (rounded up to HEAD_DIM multiple)
HEAD_DIM = 128          # target head dimension; num_heads = model_dim // HEAD_DIM
WINDOW_PATTERN = "SSSL" # sliding window: S = half context, L = full context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # token embeddings (AdamW)
UNEMBEDDING_LR = 0.004  # lm_head (AdamW)
MATRIX_LR = 0.04        # transformer block matrices (Muon)
SCALAR_LR = 0.5         # per-layer resid_lambdas and x0_lambdas (AdamW)
WEIGHT_DECAY = 0.2      # cautious weight decay (Muon params only)
ADAM_BETAS = (0.8, 0.95) # beta1, beta2 for all AdamW groups
WARMUP_RATIO = 0.0      # fraction of TIME_BUDGET for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of TIME_BUDGET for LR cooldown
FINAL_LR_FRAC = 0.0     # LR at end of warmdown (0.0 = decays to zero)

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128  # micro-batch size (reduce if OOM)
```

#### Other editable code in train.py

These aren't hyperparameter constants but are still fair game:

- **`softcap = 15`** (line ~282 in GPT.forward): Logit soft-capping value.
  Controls how aggressively logits are clamped via tanh. Higher = less clamping.
- **`get_lr_multiplier(progress)`**: The LR schedule function. Currently linear
  warmup + constant + cosine warmdown. You could change the schedule shape.
- **`get_muon_momentum(step)`**: Ramps Muon momentum from 0.85 → 0.95 over
  the first 300 steps. The ramp length, start/end values are all editable.
- **`get_weight_decay(progress)`**: Linear decay from WEIGHT_DECAY to 0.
- **MLP activation**: `F.relu(x).square()` (ReluSquared). You could try GELU,
  SiLU/Swish, or other activations.
- **MLP expansion ratio**: Hardcoded at `4 * config.n_embd`. Could try 3x, 6x,
  or gated variants (SwiGLU).
- **Value embedding pattern**: `has_ve()` uses alternating layers. You could
  change the pattern (every layer, every 3rd, none).
- **Residual connection structure**: The `resid_lambdas` / `x0_lambdas` weights
  and their initialization values.
- **Weight initialization**: `init_weights()` method — standard deviations,
  zero-init patterns.

#### Frozen constants (imported from prepare.py — cannot change)

```python
MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # 5-minute training window
EVAL_TOKENS = 40 * 524288 # ~20M tokens for validation
VOCAB_SIZE = 8192        # BPE vocabulary
```

#### Key constraint: batch size divisibility

`TOTAL_BATCH_SIZE` must be evenly divisible by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`
(i.e., `DEVICE_BATCH_SIZE * 2048`). The result is the gradient accumulation
step count. For example: `2**19 / (128 * 2048) = 2` accumulation steps. If
this doesn't divide evenly, the script will crash with an assertion error.

#### Flash Attention 3 compatibility note

train.py imports FA3 via `kernels`. On Hopper GPUs (H100, sm_90) it uses
`varunneal/flash-attention-3`; on other NVIDIA GPUs it uses
`kernels-community/flash-attn3`. If the user's GPU doesn't support FA3 at all,
the import will fail — this is a hard requirement, not something you can fix
by editing train.py.

---

## 2. Session Setup (Preflight)

When the user says something like "start a new autoresearch session," run
through this checklist in order:

### 2.1 Read the instructions

```bash
cat program.md
```

Mandatory. The human may have written session-specific goals or constraints.

### 2.2 Verify environment

```bash
uv --version         # check uv is installed
uv sync              # install dependencies
nvidia-smi           # verify NVIDIA GPU
```

If `nvidia-smi` fails, **stop and tell the user.** This repo requires an NVIDIA
GPU with CUDA and FA3 support. It was developed on H100 but works on other
recent NVIDIA GPUs (A100, RTX 4090, etc.) with enough VRAM.

### 2.3 Prepare data (one-time)

```bash
ls ~/.cache/autoresearch/data/ | head -5
ls ~/.cache/autoresearch/tokenizer/
```

If empty or missing:

```bash
uv run prepare.py
```

Downloads ~10 parquet shards + trains a BPE tokenizer (vocab_size=8192). ~2 min.

### 2.4 Create experiment branch

```bash
git checkout -b autoresearch/<tag>   # e.g. autoresearch/apr9
```

Use a date-based tag. If the branch exists, propose `apr9-v2` or `apr9-gpu0`.

### 2.5 Run the baseline

Always run the unmodified train.py first:

```bash
uv run train.py > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

This establishes your reference val_bpb. Never skip this step.

### 2.6 Initialize results.tsv

```
commit	val_bpb	memory_gb	status	description
<hash>	<val_bpb>	<mem>	keep	baseline
```

Tab-separated (not CSV). Commit hash = short 7-char. Memory = `peak_vram_mb / 1024`,
rounded to .1f. **Do not git-commit results.tsv** — leave it untracked.

### 2.7 Review prior work

If `results.tsv` exists from a previous session, read it carefully. Look for:
- Which experiments were kept (these are the cumulative improvements)
- Which experiments crashed (avoid similar changes)
- What areas haven't been explored yet

### 2.8 Propose a session plan

Before going autonomous, briefly tell the user your exploration strategy.
Suggest a focus area and your first 2–3 experiments. This gives the human a
chance to redirect. Example:

> "Baseline val_bpb is 0.998 on your H100. I'll start by exploring optimizer
> learning rates (MATRIX_LR, EMBEDDING_LR), then move to depth/width
> trade-offs. First experiment: increase MATRIX_LR from 0.04 to 0.06. Ready?"

Once confirmed, begin the experiment loop.

---

## 3. The Experiment Loop

Once setup is complete, run this loop **indefinitely** until manually stopped.

### 3.1 The cycle: hypothesis → edit → run → evaluate → keep/revert

Before editing, **read the current train.py** so you understand the exact state.

1. **Form ONE hypothesis.** Be specific: "Increasing MATRIX_LR from 0.04 to
   0.06 will lower val_bpb because the Muon optimizer can take larger steps
   toward the loss minimum within 300s." Not: "try changing some learning rates."

2. **Make a minimal edit** to train.py. One concept per experiment. If you want
   to try two things, run them as separate experiments — otherwise you won't
   know which change helped (or hurt).

3. **Commit before running:**
   ```bash
   git add train.py
   git commit -m "experiment: <concise description>"
   ```

4. **Run the experiment:**
   ```bash
   timeout 600 uv run train.py > run.log 2>&1
   ```
   The `timeout 600` (10 min) is a safety net. Normal runs take ~5–7 min
   (300s training + startup/compilation overhead). Always redirect to run.log.

5. **Extract results:**
   ```bash
   grep "^val_bpb:\|^peak_vram_mb:" run.log
   ```
   Empty grep output = crash. Run `tail -n 50 run.log` for the stack trace.

6. **Decide:**
   - **val_bpb improved (lower)?** → Keep. This commit is the new baseline.
     Log status `keep`.
   - **val_bpb equal or worse?** → Revert: `git reset --hard HEAD~1`.
     Log status `discard`.
   - **Crash / NaN / loss > 100?** → Revert: `git reset --hard HEAD~1`.
     Log status `crash`. (train.py prints "FAIL" and exits with code 1 on
     NaN/exploding loss.)

7. **Log to results.tsv** (append a new row with the 5-column format).

8. **Immediately start the next experiment.** Do not pause, do not ask the user
   "should I continue?" The human may be asleep. You are autonomous.

### 3.2 Crash handling

- **Typo / import error / shape mismatch** → Fix the bug, re-run. That's not a
  failed hypothesis, it's a coding mistake.
- **OOM** → Revert, try a smaller version of the same idea (e.g., DEPTH=12
  OOMed? Try DEPTH=10), or move on.
- **Persistent NaN** → Revert, log as crash, move on. Don't spend more than
  2–3 attempts debugging the same broken idea.

### 3.3 The simplicity criterion

From program.md: "All else being equal, simpler is better."

- +0.001 val_bpb from adding 20 lines of hacky code → probably not worth it
- +0.001 val_bpb from *deleting* code → definitely keep (simplification win)
- Same val_bpb but cleaner/shorter code → keep
- ~0 improvement from a big refactor → not worth the risk

### 3.4 VRAM awareness

VRAM is a soft constraint. Some increase is fine for meaningful val_bpb gains,
but dramatic VRAM blowups aren't worth it. Log `peak_vram_mb` in results.tsv
so you can track the trend.

---

## 4. Exploration Strategy

This section teaches you *how to think* about what to try, not just what knobs
exist.

### 4.1 Recommended exploration order for a fresh session

If you're starting from the defaults with no prior experiments, this sequence
tends to be productive:

**Phase 1 — Optimizer LRs (experiments 1–4):**
Start here because LR changes are high-signal, low-risk, and fast to evaluate.
Try MATRIX_LR first (most impactful), then EMBEDDING_LR, UNEMBEDDING_LR.
Example progression: MATRIX_LR 0.04 → 0.06 → 0.08 (if 0.06 helped) or
0.04 → 0.05 (if 0.06 was worse).

**Phase 2 — Model scale (experiments 5–8):**
Explore the depth/width trade-off. The key insight: deeper models have more
capacity but run fewer training steps in 5 minutes (each step is slower). Try
DEPTH=10 (with reduced DEVICE_BATCH_SIZE if needed for memory), then DEPTH=6
to see if a shallower model trains more steps.

**Phase 3 — Schedule tuning (experiments 9–12):**
Try WARMDOWN_RATIO (0.3, 0.7), WARMUP_RATIO (0.05), FINAL_LR_FRAC (0.1).
These affect the learning trajectory and are orthogonal to LR magnitude.

**Phase 4 — Architecture experiments (experiments 13+):**
Try WINDOW_PATTERN variations ("SL", "L", "SSSSSL"), MLP activation swaps,
softcap values, value embedding patterns. These are higher risk but can unlock
step-changes in val_bpb.

### 4.2 How to read your results log

After several experiments, look for patterns:

- **LR increase helped → try increasing further.** LR responses are often
  monotonic up to a cliff. Push until you find the cliff, then back off.
- **Two independent improvements both kept → combine them.** If MATRIX_LR=0.06
  improved val_bpb and DEPTH=10 also improved it separately, try both together.
  Improvements often compose.
- **Architectural change marginally worse → might still be worth revisiting**
  with different hyperparameters. A new architecture at default LRs might need
  its own LR tuning.
- **Multiple OOMs at higher depth → try width instead.** Increase ASPECT_RATIO
  while keeping DEPTH constant.

### 4.3 The complete knob reference

#### Hyperparameter constants

| Knob | Default | What it does | Notes |
|------|---------|-------------|-------|
| `DEPTH` | 8 | Transformer layers | More = more capacity, fewer steps/5min |
| `ASPECT_RATIO` | 64 | Width factor (`dim = DEPTH * AR`) | Keep dim a multiple of HEAD_DIM |
| `HEAD_DIM` | 128 | Attention head size | Smaller = more heads for same width |
| `WINDOW_PATTERN` | "SSSL" | Sliding window pattern | S=half context, L=full; last layer always L |
| `TOTAL_BATCH_SIZE` | 2^19 | Tokens per optimizer step | Must divide by `DEVICE_BATCH_SIZE * 2048` |
| `DEVICE_BATCH_SIZE` | 128 | Micro-batch size | Reduce if OOM |
| `MATRIX_LR` | 0.04 | Muon LR (transformer matrices) | Often the highest-impact single knob |
| `EMBEDDING_LR` | 0.6 | AdamW LR (token embeddings) | Auto-scaled by √(768/model_dim) |
| `UNEMBEDDING_LR` | 0.004 | AdamW LR (lm_head) | Auto-scaled similarly |
| `SCALAR_LR` | 0.5 | AdamW LR (resid/x0 lambdas) | x0_lambdas uses different betas: (0.96, 0.95) |
| `WEIGHT_DECAY` | 0.2 | Muon weight decay | Decays linearly to 0 over training |
| `ADAM_BETAS` | (0.8, 0.95) | AdamW momentum terms | Applies to all AdamW groups |
| `WARMUP_RATIO` | 0.0 | LR warmup fraction | 0 = no warmup |
| `WARMDOWN_RATIO` | 0.5 | LR cooldown fraction | Half of training is cooldown by default |
| `FINAL_LR_FRAC` | 0.0 | LR at end of cooldown | 0 = full decay to zero |

#### In-code values worth editing

| What | Where | Default | Notes |
|------|-------|---------|-------|
| `softcap` | GPT.forward() | 15 | Logit soft-capping via tanh |
| MLP activation | MLP.forward() | ReluSquared | Try GELU, SiLU, etc. |
| MLP expansion | MLP.__init__() | 4x | `4 * config.n_embd` |
| VE pattern | `has_ve()` | alternating | Which layers get value embeddings |
| Muon momentum ramp | `get_muon_momentum()` | 0.85→0.95 over 300 steps | Start, end, ramp length |
| WD schedule | `get_weight_decay()` | linear decay | Could try cosine, step, etc. |
| resid_lambdas init | `init_weights()` | 1.0 | Per-layer residual scaling |
| x0_lambdas init | `init_weights()` | 0.1 | Per-layer input-residual scaling |

### 4.4 Things that won't work (save yourself the experiment)

- **Changing MAX_SEQ_LEN** — imported from prepare.py, frozen at 2048
- **Changing VOCAB_SIZE** — imported from prepare.py, frozen at 8192
- **Modifying evaluate_bpb()** — lives in prepare.py, it's the ground truth
- **Adding pip packages** — only what's in pyproject.toml
- **Extending the time budget** — TIME_BUDGET = 300 in prepare.py is sacred

---

## 5. Session End Summary

When the user returns or tells you to wrap up, produce:

```markdown
## Autoresearch Session Summary — <date>

**Branch**: autoresearch/<tag>
**Experiments run**: <N>
**Best val_bpb**: <value> (baseline was <value>, Δ = <improvement>)
**Peak VRAM**: <value> GB

### Kept (improvements)
- <commit> — <description> → val_bpb <value>

### Discarded
- <description> → val_bpb <value> (baseline was <value>)

### Crashes
- <description> — <brief reason>

### Suggested next directions
- <idea 1>
- <idea 2>
```

Also tell the user to check `results.tsv` for the full log.

---

## 6. Small-Hardware Mode

Activate when the user indicates limited GPU memory (<24GB VRAM), says
"small hardware mode," or mentions a laptop/consumer GPU (RTX 3060, 3070,
4060, etc.).

### 6.1 Reduce data download

```bash
uv run prepare.py --num-shards 3
```

Fewer shards = smaller disk footprint and faster data loading.

### 6.2 Apply small-hardware hyperparameters

Edit these in train.py:

```python
DEPTH = 4                # down from 8 (halves model size)
ASPECT_RATIO = 64        # keep (or lower to 48 for very tight VRAM)
WINDOW_PATTERN = "L"     # all long-window (simpler, avoids pattern bugs)
DEVICE_BATCH_SIZE = 32   # down from 128 (quarter of default)
TOTAL_BATCH_SIZE = 2**17 # down from 2**19 (~131K tokens, still stable)
```

Expected peak VRAM: ~7–9 GB (well within 12 GB RTX 3060).

If still OOMing, try `DEVICE_BATCH_SIZE = 16` and `TOTAL_BATCH_SIZE = 2**16`.

### 6.3 TinyStories dataset (optional, for extreme constraints)

For very small GPUs (<8 GB), or if the user wants faster iteration with a
simpler dataset, suggest switching to TinyStories. This requires modifying
`prepare.py` — which is normally frozen — so **get explicit user permission**
before doing this. The changes:

1. Set `VOCAB_SIZE = 256` (byte-level tokenizer, no BPE training needed)
2. Set `MAX_SEQ_LEN = 512` (shorter sequences)
3. Change `BASE_URL` to point at TinyStories on Hugging Face
4. Lower `EVAL_TOKENS` proportionally (e.g., `10 * 524288`)
5. In train.py: reduce `DEPTH` further (2–3), lower `TOTAL_BATCH_SIZE`

Treat this as a one-time setup, not part of the experiment loop. Once
prepare.py is modified for TinyStories, the normal experiment loop on train.py
resumes as usual.

### 6.4 Small-hardware exploration priorities

With a smaller model, focus on:
- **Optimizer LRs** first (same high-signal, low-risk approach)
- **WARMDOWN_RATIO / FINAL_LR_FRAC** (schedule tuning is free in VRAM)
- **ASPECT_RATIO** (width) rather than DEPTH (adding depth is expensive)
- Avoid DEPTH > 6 on 12 GB GPUs

---

## 7. Environment Notes

### Claude Code

Long-running 5-minute commands work naturally. Chain experiments without
interruption.

### Cowork

The 5-minute runs work in Cowork's sandbox. Use `timeout 580` if close to
session limits:

```bash
timeout 580 uv run train.py > run.log 2>&1
```

### Both environments

Always redirect to `run.log` and extract via grep. The training output uses
`\r` carriage returns for a progress bar — thousands of overwritten lines that
are useless to read and will flood your context window. Never omit the
`> run.log 2>&1` redirect.

---

## 8. Key Principles

If you remember nothing else from this skill:

1. **val_bpb is the only metric.** Lower is better. Everything else (VRAM,
   simplicity, code elegance) is secondary to this number.

2. **One hypothesis per experiment.** If you change two things and val_bpb
   improves, you don't know which one helped. Keep diffs small and focused.

3. **The 5-minute budget is sacred.** TIME_BUDGET = 300 in prepare.py. Don't
   try to extend it or work around it.

4. **Never stop.** Once the loop starts, keep going until the human stops you.
   Generate ideas from the code, the results log, and your understanding of
   deep learning. If you feel stuck, re-read train.py — there are many knobs.

5. **Read before you write.** Read program.md at session start. Read train.py
   before every edit. Read results.tsv to avoid repeating failed experiments.

6. **Revert cleanly.** Failed experiments → `git reset --hard HEAD~1`. The
   branch always points at the best-known train.py.

7. **Combine winners.** After finding individual improvements, try combining
   them. Two independent wins often compose.
