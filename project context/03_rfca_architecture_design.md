# RFCA architecture design

**Status:** active architecture design, checked 2026-09-16. This document
defines the mechanism to implement and test; it does not claim novelty or
state-of-the-art performance.

## 1. Literature anchors

These papers and technical materials define the comparison boundary:

- [DiffusionGemma model overview](https://ai.google.dev/gemma/docs/diffusiongemma)
- [DiffusionGemma mechanics](https://ai.google.dev/gemma/docs/diffusiongemma/explained)
- [DiffusionGemma technical report](https://arxiv.org/abs/2608.00146)
- [Block Diffusion](https://arxiv.org/abs/2503.09573)
- [TiDAR: Think in Diffusion, Talk in Autoregression](https://arxiv.org/abs/2511.08923)
- [TiDAR project page](https://tidarlm.github.io/)
- [DiffCoT](https://aclanthology.org/2026.findings-acl.1939/)
- [Deferred Commitment Decoding](https://arxiv.org/abs/2601.02076)
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392)

The active novelty question is whether the particular full-canvas
solidifier plus rolling one-token commitment and canvas-update rule is
meaningfully distinct from these systems. This must be resolved by reading
the methods, not by relying on names or abstracts.

## 2. Components

### 2.1 Committed causal prefix

The prefix x_<t contains only tokens that have been permanently committed.
The solidifier can use ordinary causal self-attention over this prefix.

### 2.2 Provisional future canvas

The canvas C_t contains H future positions. It is initialized with noise,
masked embeddings, a shifted previous canvas, or a mixture of these. Each
position is provisional. A canvas token is not part of the committed
sequence until the solidifier commits it.

The first implementation should support H in a small sweep such as 16, 32,
64, and 128. H = 256 remains the motivating design point, not a requirement
for the first end-to-end run.

### 2.3 Bidirectional denoiser

The diffusion component may attend across the canvas positions. It may also
condition on the committed prefix through a causal or prefix-to-canvas
cross-attention path. The exact parameter sharing with the solidifier is an
experiment variable.

The denoiser must expose the noise level or denoising step to the model. A
limited-pass canvas is expected to be imperfect; that imperfection is part of
the training distribution rather than an implementation bug.

### 2.4 Full-canvas solidifier

The solidifier receives:

- the committed prefix representation;
- a representation of every position in the current canvas;
- position and noise-level information;
- an explicit boundary marker separating committed and provisional state.

The first version should compare at least three interfaces:

1. first-position-only read;
2. pooled canvas read;
3. full-canvas read with learned query/cross-attention.

The third is the RFCA candidate. The first two are necessary controls because
they test whether any benefit comes from seeing the entire future rather than
from a simpler summary.

### 2.5 Rolling update

After x_t is committed, the canvas update U should be explicit. Candidate
policies are:

- shift and append a newly initialized position;
- shift and re-noise only the newly exposed tail;
- warm-start the remaining canvas and run a partial refresh;
- refresh the whole canvas at a lower denoising budget;
- commit a variable-length prefix when confidence is high.

The one-token policy is the clearest scientific starting point. Variable
commitment is a later efficiency variant, not a hidden behavior in the base
model.

## 3. Reference inference algorithm

~~~text
prefix = BOS
canvas = initialize_canvas(prefix, horizon=H)

while prefix does not contain EOS:
    canvas = diffuse(prefix, canvas, steps=K)
    next_token = solidify(prefix, canvas)
    prefix = prefix + next_token
    canvas = roll_update(prefix, canvas, next_token)
~~~

The implementation must log K, H, update policy, wall-clock time, denoiser
calls, solidifier calls, and the number of committed tokens. This is needed
for matched-compute comparisons.

## 4. Parameter-sharing variants

Implement in this order:

1. **Frozen diffusion plus small solidifier:** the lowest-risk interface
   prototype.
2. **Frozen diffusion plus learned adapter:** the diffusion model remains
   fixed while a projection or cross-attention adapter teaches the solidifier
   how to read the canvas.
3. **Partially shared backbone:** selected layers or embeddings are shared,
   with separate diffusion and commitment heads.
4. **Jointly tuned hybrid:** only after the earlier variants are stable.

The project should not assume that maximum parameter sharing is automatically
better. Sharing can reduce memory but can also create optimization
interference between denoising and next-token commitment.

## 5. Position and state handling

The architecture must distinguish three positions:

- absolute position in the generated sequence;
- relative position within the current canvas;
- denoising/noise step.

The rolled canvas must not accidentally reuse a position encoding that tells
the solidifier a provisional token is already committed. At minimum, test
explicit canvas-relative positions plus a sequence-position offset and an
is-canvas indicator.

## 6. Main technical risks

- **Compute:** re-denoising H positions after every committed token can be
  far more expensive than block commitment.
- **Drift:** a warm-started canvas may accumulate stale predictions after many
  shifts.
- **Interface collapse:** the solidifier may ignore the canvas and behave like
  an ordinary autoregressive model.
- **Leakage:** teacher-forced clean futures can make the design appear useful
  without teaching it to handle its own imperfect canvases.
- **Optimization conflict:** joint training may improve one component while
  damaging the other.

Every risk has a corresponding ablation or diagnostic in
[the experiment plan](05_rfca_experiment_plan.md).
