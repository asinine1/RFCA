# RFCA experiment plan

**Status:** active pre-code experiment plan, checked 2026-09-16. All
comparisons should use the same tokenizer, data splits, parameter budget
where practical, and documented compute accounting.

## 1. Core comparison set

| System | Purpose |
|---|---|
| Standard causal autoregressive model | Lower-complexity quality and cost baseline |
| Block diffusion model | Comparison to diffusion inside fixed-size autoregressive blocks |
| Diffusion canvas with first-position-only solidifier | Tests whether full-canvas access matters |
| Diffusion canvas with pooled solidifier | Tests a cheap summary interface |
| RFCA with full-canvas solidifier | Main proposed architecture |
| RFCA with variable-length commitment | Efficiency extension, not the base claim |

The exact baselines must be implemented or reproduced closely enough that the
comparison is fair. A baseline that receives a different data budget or
unreported sampling budget is not a meaningful control.

## 2. Ablations

At minimum, vary:

- canvas horizon H;
- number of denoising passes K;
- canvas initialization;
- first-position, pooled, and full-canvas interfaces;
- frozen diffusion, adapter tuning, and joint tuning;
- shift-only, partial-refresh, and full-refresh update policies;
- one-token versus confidence-based variable commitment;
- on-policy versus teacher-forced prefixes;
- shared versus separate positional encodings;
- diffusion loss weight and rolling loss weight.

The highest-priority ablations are the ones that establish mechanism:
full-canvas versus summary, imperfect versus clean canvases, and rolling
updates versus one-shot block commitment.

## 3. Evaluation stages

### Stage A: language modeling sanity checks

Use a small public or synthetic corpus to test next-token loss, perplexity
where applicable, exact-match on short structured sequences, and generation
validity. The aim is to determine whether the model learns the interface at
all.

### Stage B: long-range and structured dependencies

Use synthetic tasks where the useful future structure is controlled, such as
bracket completion, delayed copying, multi-step symbolic transformations, and
format-constrained generation. These tasks can reveal whether the canvas
changes current commitments rather than serving as unused decoration.

### Stage C: reasoning-relevant stress tests

Use carefully scoped public benchmarks or generated reasoning tasks only
after the architecture passes the earlier stages. Separate reasoning quality
from memorization and report the full sampling budget. Do not claim general
reasoning improvement from a single benchmark.

## 4. Metrics

Report both quality and resource use:

- next-token loss or perplexity where appropriate;
- exact match and validity on structured tasks;
- task accuracy on held-out reasoning-style tasks;
- calibration or confidence-quality relationship;
- tokens per second;
- denoiser calls per committed token;
- wall-clock time per generated token;
- peak memory;
- total training compute;
- canvas-use diagnostics.

The primary comparison should be matched by either generated-token budget,
wall-clock budget, or denoiser-call budget. If the result changes depending
on the budget definition, report all views.

## 5. Canvas-use diagnostics

To test whether the solidifier uses the provisional future:

- replace the canvas with noise at evaluation time;
- shuffle canvas positions;
- replace the canvas with a pooled summary;
- mask selected canvas positions;
- compare solidifier logits with and without canvas access;
- measure sensitivity to denoising level;
- inspect whether attention or cross-attention concentrates on future
  positions relevant to the next commitment.

A model that performs identically when the canvas is removed has not
demonstrated the intended mechanism, even if its headline score is strong.

## 6. Success criteria

Before running the final test, write down a primary hypothesis and a
resource budget. A credible result would show that, under at least one
pre-specified matched budget, RFCA improves a clearly defined quality or
structured-dependency metric without an unreported or impractical compute
increase.

A null result is still scientifically useful if the ablations show whether
the failure came from the canvas representation, rolling update, training
curriculum, or compute cost.

## 7. Claim language

Safe claim:

“We developed and evaluated a rolling hybrid architecture in which a
limited-pass diffusion canvas conditions an autoregressive next-token
solidifier.”

Unsafe until proven:

- “the first diffusion-autoregressive language model”;
- “human-like visualization”;
- “the model plans its answer”;
- “better reasoning” without task-specific evidence;
- “faster generation” without matched wall-clock measurements.
