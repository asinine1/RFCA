# RFCA Phase 1: Frozen-Theorizer / Causal-Solidifier Feasibility

**Project:** RFCA — Rolling Future Canvas Architecture  
**Repository:** `https://github.com/asinine1/RFCA`  
**Phase:** 1 — architecture, controls, feasibility, and first frozen-theorizer GPU run  
**Status:** End-to-end pipeline operational; meaningful general language-quality or slot-specific canvas benefit not yet established.  
**Last consolidated:** 2026-09-22

## 1. Executive summary

RFCA investigates a two-part language architecture:

1. The **theorizer** is a bidirectional diffusion model that develops a
   provisional, uncertain representation of future content in a fixed-length
   canvas.
2. The **solidifier** is a causal autoregressive model that reads the committed
   prefix and the theorized canvas, then commits exactly one next token.

After commitment, the canvas is rolled forward by one slot. The committed
position is discarded, the remaining provisional state is shifted, and only
the newly exposed tail is initialized. The solidifier therefore receives
future-oriented information without being given the future as committed
cleartext tokens.

The central research question is not whether diffusion and autoregression can
be placed in the same model. Existing work has already established that broad
architecture family. The narrower Phase 1 question is:

> Under matched parameter, compute, memory, and latency budgets, does an
> imperfect, position-aware, model-generated rolling future state improve a
> separate causal one-token decision on dependency-sensitive long-horizon
> tasks?

The most reliable Phase 1 result is that **training on imperfect retained
canvas states can greatly improve rolling-state stability on a synthetic task**.
The project has not yet shown that this produces a reliable general-language
quality advantage over a matched causal or pooled-summary control.

## 2. Terminology and intended mechanism

### Theorizer

The diffusion side. It refines a continuous 256-slot canvas conditioned on the
already committed prefix. It is intended to carry a vague, high-entropy future
representation rather than readable future tokens.

The desired stopping behavior is much less certain than ordinary DiffusionGemma
generation. The project initially discussed ordinary entropy stopping around
`0.005`; RFCA instead targets a much higher boundary, approximately `0.5`
nats as a first test point. The intention is to stop when the theorizer has a
useful directional hypothesis without allowing it to converge into cleartext.

### Solidifier

The causal side. It reads:

- the committed prefix;
- the frozen theorizer's causal prefix hidden state; and
- the theorizer's provisional canvas hidden states with explicit destination
  slot identity.

It produces one next-token distribution and commits one token. It does not
receive the future token sequence as input during inference.

### Rolling transition

The intended state transition is:

```text
committed prefix
        |
        v
theorizer refines the retained canvas
        |
        v
solidifier reads prefix + positioned canvas
        |
        v
commit exactly one token
        |
        v
shift canvas left by one slot; initialize only the fresh tail
        |
        +---- repeat
```

This is a shift-and-refresh operation, not appending a token to the end of the
canvas. The project explicitly tested and preserved that behavior.

## 3. Decisions made before the frozen-theorizer implementation

### Dropped API distillation from DeepSeek

The idea of using DeepSeek R1/V4.1 Flash through an API as an autoregressive
teacher was considered and rejected. A much stronger external teacher could
teach the solidifier to ignore the canvas or hide a collapsed/global shortcut.
The first research instrument therefore keeps the causal solidifier
self-contained and uses a frozen diffusion model as the theorizer.

### Positional information is required

The user requirement was that shuffled language should degrade and the
solidifier should know which provisional information belongs to which canvas
slot. A first cross-attention implementation failed this requirement: plain
canvas cross-attention was nearly permutation-invariant, with a measured
maximum shuffle-logit delta of approximately `7.6e-6`.

The solidifier was changed to re-inject explicit destination-slot embeddings at
the canvas read boundary. Diagnostics were also corrected so that shuffling
changes provisional content while preserving destination slots, rather than
accidentally shuffling content and its attached position together.

### The solidifier can be larger than the theorizer interface

The project intentionally tested a dense solidifier that is much larger than
the mechanism that produces the canvas. This reflects the intended division of
labor: the theorizer proposes an uncertain future representation, while the
solidifier performs the precise causal decision.

### Frozen theorizer before on-policy rolling

Jointly training the theorizer and solidifier from scratch produced a canvas
that could collapse or broadcast a global summary. The primary experiment was
therefore moved to a genuinely pretrained, frozen diffusion theorizer while
training a separate solidifier. The from-scratch implementation remains a
control and a source of diagnostic tasks.

## 4. Repository and implementation work

### Main files

- `training/train_rfca_student.py` — from-scratch Gemma-style RFCA prototype,
  synthetic tasks, canvas ablations, rolling diagnostics, and curricula.
- `training/frozen_diffusion_solidifier.py` — frozen DiffusionGemma harness,
  full-canvas RFCA path, matched no-canvas causal control, teacher-forced
  rolling training, and held-out per-commit evaluation.
- `training/validate_frozen_diffusion_solidifier.py` — pre-GPU interface and
  gradient/invariance gate.
- `training/README.md` — commands, environments, checkpoints, and experiment
  notes.
- `project context/08_rfca_quality_benchmark.md` — internal quality benchmark
  and interpretation.
- `project context/09_external_viability_review_2026-09-19.md` — external
  research-frontier, competition, media, and novelty review.
- `DNR_ACA_MECHANISMS_RESEARCH_REPORT.md` — the real-text source used for the
  first frozen-theorizer Runpod trial.

### From-scratch model features

The initial PyTorch prototype includes:

- causal prefix encoding;
- a bidirectional mask-only canvas denoiser;
- full-canvas next-token solidification;
- approximate 416M, 877M, and 1.05B presets using a Gemma-sized
  262,144-token vocabulary;
- rolling canvas inference;
- entropy, normalized entropy, and mean maximum-probability traces;
- maximum-pass fallback when an entropy boundary is not reached;
- full/no-canvas/first-position/pooled/shuffled/random diagnostics;
- KL divergence and top-1 agreement measurements;
- position probes and ordered-sequence probes;
- teacher-forced imperfect-retained-canvas curriculum training;
- capacity-matched causal controls.

### Frozen DiffusionGemma harness

The frozen harness:

- loads `DiffusionGemmaForBlockDiffusion`;
- carries both canvas token IDs and self-conditioning logits through rolling
  updates;
- returns continuous prefix and canvas hidden states;
- uses a separate causal solidifier with tied token embeddings and output head;
- injects explicit canvas slot embeddings;
- supports `--canvas-mode full|none`;
- adds one causal prefix layer to the no-canvas control through
  `--causal-extra-layers 1`;
- defaults to multinomial theorizer sampling to match the generation path;
- supports deterministic `--denoiser-sampling argmax` for controlled ablations;
- saves only trainable solidifier weights, not a duplicate frozen language
  model;
- seeds PyTorch and CUDA after the reproducibility fix.

The 400M frozen student configuration is:

```text
d_model=768
n_heads=12
layers=12
ff_dim=3072
canvas_length=256
theorizer_hidden_size=2816
vocab_size=262144
```

Its exact full-canvas trainable count is `293,270,017` parameters.

## 5. Local experiments and what they established

### Basic smoke and rolling plumbing

The initial 20-step MPS smoke test completed training, validation, and
checkpoint writing. Training loss fell from approximately `211.5` to `101.4`;
validation loss fell from approximately `139.9` to `55.7`. This established
that the basic PyTorch pipeline, checkpointing, and optimizer path worked.

A rolling smoke checkpoint completed four commits and serialized a full trace.
The `0.5`-nat entropy threshold was not reached within four passes, correctly
exercising the maximum-pass fallback. This was plumbing evidence, not a
quality result.

### First canvas-use smoke task

On a 200-step local smoke checkpoint evaluated over 128 examples:

| Condition | Loss | Accuracy |
| --- | ---: | ---: |
| Full canvas | 2.2608 | 0.4062 |
| No canvas | 39.4793 | 0.2109 |
| First-position only | 2.2748 | 0.4609 |
| Pooled canvas | 2.3139 | 0.4062 |
| Shuffled canvas | 2.2608 | 0.4062 |
| Random canvas | 28.7590 | 0.0469 |

The solidifier was clearly using canvas information, but the simple task did
not require position-specific access. Pooled and shuffled canvases performed
like the full canvas, so this did not justify the full RFCA claim.

### Position probe

A corrected position probe placed distractors in early slots and a decision
symbol at a designated final slot. On 256 examples:

| Condition | Accuracy |
| --- | ---: |
| Full position-aware canvas | 0.5859 |
| No canvas | 0.1055 |
| First position only | 0.3047 |
| Pooled canvas | 0.3320 |
| Slot-preserving shuffle | 0.4688 |
| Random canvas | 0.0703 |

Two additional shuffle seeds gave full/shuffled results of `0.5391/0.4414`
and `0.5742/0.4609`. This was initial evidence of position-specific canvas
use, but not yet a strong language-quality result.

### Ordered and indexed sequence probes

The corrected ordered-sequence probe asked the theorizer to sort four keyed
words into one of 24 permutations and the solidifier to emit a class byte for
the exact order. Results after 2,000 steps:

| Condition | Accuracy |
| --- | ---: |
| Full canvas | 0.9141 |
| No canvas | 0.2881 |
| Random | 0.2217 |
| Pooled | 0.8818 |
| Slot-preserving shuffle | 0.8945 |

This was strong canvas dependence but weak evidence of slot-specific extraction:
the theorizer appeared to broadcast an order summary into many slots.

The indexed final-query version reached:

| Condition | Accuracy |
| --- | ---: |
| Full canvas | 0.9805 |
| No canvas | 0.7266 |
| Random | 0.7002 |
| Pooled | 0.9766 |
| Slot-preserving shuffle | 0.9756 |

A second shuffle seed gave `0.9795` full versus `0.9785` shuffled. This again
confirmed canvas contribution, but it did not establish that information was
stored in the intended slots.

### Rolling collapse and imperfect-canvas curriculum

The first rolling diagnostic showed severe retained-state collapse. On the
standard smoke checkpoint, retained accuracy fell from `0.5469` at commit 1 to
`0.0000` by commit 8, while reset accuracy stayed around `0.37--0.44` after
the first commit.

A larger-solidifier clean-only model also failed to retain its canvas. Its
retained accuracy over commits 1--8 was:

```text
0.9766, 0.4922, 0.4297, 0.2656,
0.1016, 0.2031, 0.0469, 0.1875
```

Teacher-forced imperfect-canvas curriculum training substantially improved
retention:

```text
0.9375, 0.9141, 0.9453, 0.8750,
0.9062, 0.8516, 0.8516, 0.8984
```

This is the strongest positive Phase 1 result. It supports the narrower claim
that training on imperfect retained states teaches the solidifier to remain
functional during rolling updates.

### Matched causal control

A capacity-matched causal control used no canvas and comparable dense capacity.
Its clean diagnostic loss/accuracy were `0.1862/0.9370`, versus
`0.2224/0.9312` for curriculum RFCA. Its retained and reset rolling accuracies
were identical across commits:

```text
0.9531, 0.5391, 0.5547, 0.6406,
0.4922, 0.8203, 0.5391, 0.6641
```

Therefore, the curriculum result is a retention result, not yet a one-shot
accuracy win over similar causal capacity.

## 6. Frozen DiffusionGemma work

### Checkpoint discovery

The installed local DiffusionGemma `.dgq` pack is a Rust/Metal inference
format. It is approximately 18.84 GiB and contains about 25.82B indexed
elements, but it is not directly loadable by the PyTorch Transformers harness.

The official Transformers checkpoint was downloaded and verified:

```text
google/diffusiongemma-26B-A4B-it
```

It contains 11 safetensor shards, occupies approximately 48 GiB, and exposes
the required 256-slot continuous canvas hidden states. The Mac has 48 GiB of
RAM, so the checkpoint was staged for an A100 rather than loaded locally.

### Interface validation

The tiny public DiffusionGemma checkpoint was used only as an interface test.
It successfully returned canvas hidden states and logits, and the frozen
solidifier trained for one-step/two-commit smoke runs. It achieved zero
held-out language accuracy, as expected for a random/test-scale checkpoint.
Those numbers must not be used as a quality result.

The pre-GPU validator caught and fixed two important issues:

1. The first target-selection test accidentally let the tied output head copy
   the current prompt token, producing a false zero-loss result.
2. Plain cross-attention did not preserve enough slot identity, leading to the
   explicit destination-slot embedding fix.

The corrected interface gate passed with:

```text
canvas removal max-logit delta: 0.3316
canvas shuffle delta:            0.2151
frozen-theorizer delta:          0.0
fixed-feature loss:              27.8936 -> 0.0
retained rolling commits:        4
trainable gradient tensors:      39
```

## 7. Runpod experiment

### Environment

The first real run used one A100 SXM 80GB. The project did not use two GPUs
because the current harness is single-device and has no sharding or distributed
training path.

The Runpod template initially contained PyTorch `2.4.1+cu124`; the project
requires at least 2.5. The environment was upgraded to the official PyTorch
2.5.1 CUDA 12.4 wheel family. The repository requirement was changed from
`torch>=2.8` to `torch>=2.5`.

The persistent Runpod layout is:

```text
/workspace/isef2027       repository
/workspace/models          DiffusionGemma checkpoint
/workspace/runs            logs, metrics, and student checkpoints
```

### Operational fixes

The first launch exposed two non-research issues:

- `/workspace/runs` did not exist, so `tee` could not create the log.
- The checkout path is lowercase `/workspace/isef2027`, while the first data
  argument used uppercase `/workspace/ISEF2027`.

The next launch loaded all 1,047 checkpoint shards and instantiated the
293,270,017-parameter student. It then exposed a real dtype mismatch: frozen
DiffusionGemma hidden states were bf16 while newly initialized bridge layers
were float32. The bridge inputs are now cast at the boundary, preserving the
frozen model's bf16 operation and the student's float32 trainable weights.

The local synthetic bf16 smoke test passed for both full RFCA and adapter
paths. The Runpod training process subsequently reached step 53 and then
completed the 100-step run without NaNs, OOMs, or runtime failures.

### First real RFCA result

The full-canvas 400M run used:

```text
train examples:       4059
validation examples:  1337 available, 32 evaluated per commit
commits:               8
diffusion steps:       2
entropy bound:         0.5
training steps:        100
```

Held-out RFCA metrics:

```text
loss range:     61.92--75.76
accuracy:       1/32 on commits 1--5
                2/32 on commits 6--8
```

The no-canvas causal control used one extra causal prefix layer and produced:

```text
loss range:     69.69--83.58
accuracy:       1/32 on every commit
```

Across the eight commits:

| Metric | Full RFCA | No-canvas causal |
| --- | ---: | ---: |
| Mean held-out loss | 66.60 | 75.90 |
| Loss wins | 7/8 commits | 1/8 commits |
| Exact accuracy | 11/256 = 4.3% | 8/256 = 3.1% |

This is encouraging directional evidence that the canvas can help, but it is
not a conclusive quality result. The evaluation was tiny, the model trained
for only 100 steps, losses remained high, and the default multinomial canvas
sampling introduced noise. Furthermore, these two first runs happened before
the PyTorch/CUDA seeding fix, so they were not perfectly paired.

## 8. External viability and novelty review

The external review is recorded in
`project context/09_external_viability_review_2026-09-19.md`.

### Research frontier

Broad novelty claims are unsafe. DiffusionGemma, Block Diffusion, TiDAR,
speculative diffusion decoding, Deferred Commitment Decoding, DiffCoT, PLANNER,
LLaDA, Mercury, and related work already cover large parts of the
diffusion/autoregressive, draft/verify, revisable-future, or deferred-commitment
space.

The defensible contribution is an empirical gap study:

> Does an imperfect, position-aware, repeatedly retained full canvas improve a
> causal one-token decision beyond a matched causal or pooled summary under
> matched budgets?

Unsafe claims include “first diffusion-autoregressive language model,” “first
future canvas,” “first deferred commitment system,” or general claims that the
model plans, reasons, or improves language quality.

### Competition viability

The project is potentially viable as a narrow DMRSEF mechanism/retention study
if the benchmark becomes clean and the visual explanation is strong. CSEF/state
recognition is conditional on decisive held-out results. An ISEF finalist is
plausible only after stronger evidence. ISEF category-award or Grand-Award
evidence is not present yet.

The project should be presented as a causal state-management study, not as a
general AI system. The media should visibly show:

- the provisional canvas;
- the hard commitment boundary;
- the one-token commit;
- the roll/refresh operation;
- matched controls;
- retained-versus-reset results;
- compute and latency costs; and
- student-owned implementation and analysis.

AI assistance, model terms, dataset licenses, hashes, preprocessing, cloud/GPU
support, and continuation boundaries must be documented for fair submission.

## 9. Current claims and non-claims

### Claims supported by Phase 1

- The RFCA interface can run end to end.
- A frozen DiffusionGemma theorizer can provide continuous prefix and canvas
  hidden states to a separate causal solidifier.
- Canvas removal can substantially change predictions.
- Explicit slot identity is necessary for the intended positional interface.
- Clean-only training can collapse under retained rolling state.
- Imperfect retained-canvas curriculum training can substantially improve
  rolling-state stability on a synthetic task.
- The first short frozen-DiffusionGemma run shows a promising lower-loss signal
  for full RFCA than for the no-canvas control.

### Claims not yet supported

- RFCA improves general language quality.
- RFCA improves reasoning.
- RFCA is faster or cheaper than a causal model.
- The canvas stores genuinely slot-specific information on natural language.
- RFCA beats a matched causal model after parameter, FLOP, latency, and memory
  normalization.
- The system is the first diffusion/autoregressive or deferred-commitment
  architecture.
- The system plans or sees the future in a human-like sense.

## 10. Decisive next experiment

The next primary benchmark should be a **randomized-index future-canvas
retrieval task** designed to defeat global broadcast summaries.

Each example should contain multiple independently generated facts or symbols
in distinct latent slots. A random query should select one slot, while the
mapping and slot permutation vary by example. The target must not be inferable
from a fixed position or a single global order summary.

Required conditions:

1. Full position-aware RFCA canvas.
2. No-canvas matched causal control.
3. Pooled-canvas control.
4. Slot-preserving shuffled canvas.
5. First-position-only canvas.
6. A one-shot draft/refine or block-diffusion baseline if feasible.

Minimum evidence standard:

- at least 1,000 held-out examples;
- at least three independent seeds;
- deterministic argmax as the primary result and stochastic sampling as an
  ablation;
- held-out slot permutations and templates;
- exact-match accuracy and retained/reset accuracy across eight commits;
- parameter count, denoiser calls, wall-clock latency, peak memory, and
  training compute for each condition;
- full RFCA must beat pooled and shuffled controls, not merely no-canvas.

Before that benchmark, run the frozen DiffusionGemma comparison again with the
latest reproducibility fix:

```text
--seed 2027
--denoiser-sampling argmax
--eval-examples 128 or 256
--steps 500--1000
```

Stay on one A100 until the paired result is clean. Do not scale to two GPUs
before the single-device experiment establishes a real signal.

A strong null result is still useful:

> The imperfect-canvas curriculum stabilizes retained state, but the full
> canvas does not add slot-specific information or final accuracy beyond a
> matched causal or pooled summary.

That would support a boundary-condition/state-retention study if replicated
and honestly framed.

## 11. Repository and reproducibility state

The repository is initialized on `main` with the configured GitHub remote. The
latest pushed commit at the time of this Phase 1 consolidation is:

```text
4a5c8f2 Update project handoff notes
```

Important preceding commits:

```text
ebc68aa Seed torch for reproducible solidifier runs
4c07890 Fix frozen student dtype boundaries
3db77b8 Relax frozen training PyTorch floor
```

The `.gitignore` excludes virtual environments, `.env` files, SSH/private-key
files, model weights, native model packs, generated runs, temporary files,
logs, and Python caches. Large model checkpoints are intentionally not stored
in Git.

The Runpod pod was stopped after the first GPU smoke comparisons. On restart,
verify that `/workspace` is still the persistent network volume, pull the
latest `main`, and rerun the deterministic paired experiment before spending
on a second GPU.

