# Student training and canvas-integrity plan

**Status:** active implementation note, checked 2026-09-18.

## Current decision

The first executable student is initialized from scratch rather than loaded
from the released DiffusionGemma or Gemma 4 weights. The released models are
useful architectural references, but using a much larger pretrained causal
teacher at this stage could make the solidifier look successful while it
ignores a collapsed or uninformative diffusion canvas.

The student therefore uses a Gemma-style parameterization—RMSNorm, SwiGLU,
tied token embeddings, and scaled dot-product attention—with a separate causal
prefix path, bidirectional canvas denoiser, and full-canvas solidifier. The
initial presets target approximately 400M, 750M, and 1B parameters when used
with a Gemma-sized tokenizer. The script prints the exact count for every run.

## Initial objective

For a committed prefix (x_{<t}), the model receives an all-mask canvas of
horizon (H). The denoiser predicts the future canvas, while the solidifier
uses the final prefix state plus all canvas positions to predict the first
future token. The clean target suffix is a label only; it is never inserted
into the canvas input in the baseline.

The initial loss is:

```text
L = lambda_diff * CE(canvas_logits, future_suffix)
  + lambda_solid * CE(next_token_logits, future_suffix[0])
```

This is not yet the complete rolling curriculum. It is the interface sanity
check needed before adding warm-started canvas updates and on-policy unrolls.
The current direction is to use a genuinely trained, frozen diffusion
theorizer for the next solidifier experiment; the from-scratch theorizer is
now a control because it does not retain useful hidden state across rolling
commits.

## Training order

1. Run the byte-tokenizer synthetic smoke test.
2. Run a short real-text experiment with the smaller student preset.
3. Compare the full-canvas solidifier against first-position-only and pooled
   canvas controls.
4. Test canvas removal, position shuffling, and denoising-step changes.
5. The current script now has an inference-time rolling loop: the theorizer
   refines a continuous full canvas until a configurable uncertainty boundary,
   the solidifier commits one token, and the canvas shifts with a fresh tail.
6. The position-dependent probe is now implemented. The solidifier explicitly
   re-injects canvas slot identities at its read boundary, and the shuffle
   ablation swaps provisional canvas content while keeping destination slots
   fixed. On the revised 300-step probe, full accuracy was 0.5859, pooled
   0.3320, and slot-preserving shuffled 0.4688 (two additional shuffle seeds
   gave full 0.5391/0.5742 versus shuffled 0.4414/0.4609). This is initial
   evidence that the solidifier uses position-specific canvas information.
7. The ordinary text smoke corpus is not a language-quality gate yet: full and
   shuffled accuracy were both 0.5430, so that tiny predictable corpus does not
   force order-sensitive future structure.
8. The harder four-word ordered-sequence probe reached 0.9141 full accuracy,
   versus 0.2881 with no canvas and 0.2217 with random canvas. However, pooled
   and slot-preserving shuffled canvas remained 0.8818 and 0.8945. The
   theorizer is using the canvas but broadcasting an order summary rather than
   requiring slot-specific extraction.
9. The indexed final-slot probe reached 0.9805 full accuracy versus 0.7266
   with no canvas, but pooled and shuffled remained 0.9766 and 0.9756 (a
   second shuffle seed gave 0.9785). This confirms canvas dependence but not
   the stronger position-sensitive language claim.
10. Added a teacher-forced `rolling_diagnose` path that performs the intended
    multi-token transition: refine, commit, append the known token, shift the
    hidden canvas, and initialize only the new tail. On the fixed-window smoke
    checkpoint, retained accuracy fell from 0.5469 at commit 1 to 0.0859 at
    commit 2 and 0.0000 by commit 7, while resetting the canvas each commit
    stayed around 0.37-0.44. The rolling mechanics work, but the checkpoint
    is out-of-distribution after its first retained update.
11. Use a trained frozen diffusion theorizer for the next primary experiment.
    It must expose continuous canvas states, accept the committed prefix, and
    support the intended one-slot hidden-state shift. Train a small projection
    or cross-attention adapter plus solidifier on teacher-forced rolling
    trajectories before any sampled unrolls.
12. Do not rent A100s until the frozen-theorizer solidifier passes the rolling
    retention and canvas-order gates.
13. Move the best verified configuration to two A100s and record wall-clock,
   peak memory, tokens/second, and denoiser calls per committed token.
14. The local machine now has the real 26B-A4B checkpoint, but in the Rust /
   Metal-only `.dgq` format. The PyTorch adapter path is implemented against
   the official Transformers interface and validated with the tiny public
   checkpoint; do not call the tiny run a quality result. Before meaningful
   adapter training, either obtain a compatible Transformers checkpoint on the
   GPU host or add a native activation-export bridge for the `.dgq` runtime.
15. The causal solidifier must add explicit destination-slot embeddings when it
    reads the frozen canvas. A plain cross-attention pool is permutation
    invariant over canvas rows and therefore cannot satisfy the position-
    dependent RFCA requirement. The local validator now gates this with both
    canvas-removal and canvas-shuffle sensitivity checks.

The initial rolling stop criterion should be entropy in nats with a starting
threshold around `0.5`, while mean max-token confidence is logged in parallel.
The threshold must be swept and reported rather than treated as a universal
value; vocabulary size and calibration affect entropy. A confidence threshold
around `0.5` is an alternate, more vocabulary-stable control.

An API teacher is intentionally out of scope for this first scaffold. It can
be reconsidered later as a separately registered ablation, not as an unseen
source of extra supervision in the main result.
