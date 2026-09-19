# RFCA student-training mockup

This directory contains the first executable training scaffold for the
Rolling Future-Conditioned Autoregression (RFCA) project.

The student is deliberately not the released DiffusionGemma checkpoint. The
released DiffusionGemma model is a 26B-total/4B-active Gemma 4 MoE model. The
scaffold here is a smaller Gemma-style model built from scratch so that its
parameter count, training data, and canvas interface are controlled by the
experiment.

The first version has three pieces:

1. A causal prefix encoder for committed tokens.
2. A bidirectional canvas denoiser that cross-attends to the prefix.
3. A full-canvas solidifier that predicts the next committed token.

The training canvas is initialized entirely with a learned mask token. The
target future is used only for the diffusion loss and next-token loss; it is
not copied into the canvas input. This is intentionally conservative. A later
experiment can add model-generated or partially denoised canvases, but that
should be labeled separately from this baseline.

## 1. Install

Create an environment appropriate for the local machine, then install a
platform-appropriate PyTorch build followed by:

```bash
python -m pip install -r training/requirements.txt
```

The project-local `.venv` has PyTorch 2.8 for the small MPS smoke tests. The
separate `.venv-diffusion` environment below has the newer Transformers stack
for the frozen-theorizer adapter.

### Frozen DiffusionGemma adapter environment

The frozen-theorizer experiment uses a separate Python 3.10+ environment
because the current DiffusionGemma Transformers implementation requires the
newer model API:

```bash
python3.12 -m venv .venv-diffusion
.venv-diffusion/bin/python -m pip install -r training/requirements-frozen-diffusion.txt
```

The local machine also has the real 26B-A4B checkpoint at
`/Users/shrim1729/Models/DiffusionGemma/model/diffgemma-26b-a4b-it-q4`. It is a
19 GiB `.dgq` quantized mmap pack for the installed Rust/Metal runtime. That
runtime is useful for generation, but it is not a Transformers/PyTorch
checkpoint and therefore cannot be passed directly to the adapter trainer.
The PyTorch student script currently uses a Transformers checkpoint and keeps
the tiny public checkpoint as an interface-only test:

```bash
.venv-diffusion/bin/python training/frozen_diffusion_solidifier.py probe \
  --model-id trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion \
  --device cpu --dtype fp32 --prompt "Hello" --commits 3 \
  --diffusion-steps 2 --entropy-bound 0.5

.venv-diffusion/bin/python training/frozen_diffusion_solidifier.py train \
  --model-id trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion \
  --data synthetic --device cpu --dtype fp32 --solidifier-mode ar \
  --solidifier-preset smoke --solidifier-max-prefix-length 64 \
  --steps 12 --commits 3 --diffusion-steps 2 --entropy-bound 0.5 \
  --output runs/frozen-solidifier/tiny-ar-smoke.pt
```

The tiny checkpoint is not a quality baseline; it verifies that the frozen
DiffusionGemma forward returns prefix hidden states, continuous canvas hidden
states, logits, and rolling self-conditioning state, and that gradients reach
only the causal solidifier. The `smoke`, `student-400m`, `student-750m`, and
`student-1b` presets select the trainable causal stack; the exact parameter
count is printed during training. The full model should be used for the
meaningful experiment once a compatible Transformers checkpoint is available
on the GPU host, or once the local `.dgq` runtime has an activation-export
bridge.

The official Transformers checkpoint is now staged at
`/Users/shrim1729/Models/DiffusionGemma/hf/diffusiongemma-26B-A4B-it`. Its
configuration is `DiffusionGemmaForBlockDiffusion` with a 256-slot canvas,
2,816-dimensional hidden states, and a 262,144-token vocabulary. The model
files occupy roughly 48 GiB before runtime overhead, so this 48-GiB-RAM Mac
should not load the full checkpoint; use an A100 host for the real run.

The frozen comparison harness supports a full-canvas RFCA run and a causal
control with one extra causal layer to keep trainable capacity close:

```bash
.venv-diffusion/bin/python training/frozen_diffusion_solidifier.py train \
  --model-id /path/to/diffusiongemma-26B-A4B-it \
  --device cuda --dtype bf16 --data path/to/held-out-text.txt \
  --solidifier-mode ar --solidifier-preset student-400m \
  --solidifier-max-prefix-length 256 --canvas-mode full \
  --causal-extra-layers 0 --commits 8 --diffusion-steps 2 \
  --entropy-bound 0.5 --steps 1000 \
  --output runs/frozen-solidifier/rfca.pt \
  --metrics-output runs/frozen-solidifier/rfca.metrics.json

# Repeat with --canvas-mode none --causal-extra-layers 1 for the causal control.
```

The training loop teacher-forces the known committed token, shifts the frozen
canvas, and evaluates held-out rolling accuracy separately at each commit.

Before any longer run, execute the architecture gate:

```bash
.venv-diffusion/bin/python training/validate_frozen_diffusion_solidifier.py \
  --model-id trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion \
  --device cpu --dtype fp32 --solidifier-preset smoke \
  --max-prefix-length 64 --diffusion-steps 2 --entropy-bound 0.5 \
  --commits 4 --overfit-steps 8 \
  --output runs/frozen-solidifier/validation.json
```

This gate checks frozen-weight isolation, nonzero student gradients, exact
one-slot canvas shifting, multi-commit rolling execution, finite logits,
canvas removal sensitivity, and canvas-order sensitivity. The causal student
adds explicit destination-slot embeddings at the canvas read boundary; this
is required because ordinary cross-attention over a sequence without slot
embeddings is permutation-invariant. Passing this gate establishes viable
plumbing and trainability, not language quality from the tiny checkpoint.

## 2. CPU/MPS smoke test

This uses a byte tokenizer and synthetic structured text, so it does not
download a model or dataset:

```bash
python training/train_rfca_student.py train \
  --model-preset smoke \
  --tokenizer byte \
  --data synthetic \
  --prefix-length 64 \
  --canvas-length 16 \
  --diffusion-steps 2 \
  --batch-size 2 \
  --steps 20 \
  --eval-every 10 \
  --save-every 20 \
  --output-dir runs/smoke
```

The useful result of this run is not model quality. It verifies that the
dataset, mask-only canvas, denoiser, solidifier, loss, checkpoint, and device
selection all work end to end.

## 3. First real student

For a Gemma-family tokenizer, use a compatible tokenizer identifier and accept
the model's license terms if the hosting service requests it. The model is
still initialized from scratch; only the tokenizer is reused.

```bash
python training/train_rfca_student.py train \
  --model-preset student-400m \
  --tokenizer google/gemma-4-E2B-it \
  --data path/to/train.jsonl \
  --prefix-length 256 \
  --canvas-length 32 \
  --diffusion-steps 2 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --gradient-checkpointing \
  --precision bf16 \
  --output-dir runs/student-400m
```

JSONL accepts either `{"text": "..."}` records or
`{"prompt": "...", "completion": "..."}` records. Plain text files use one
example per line.

The preset names are approximate because the tokenizer embedding table is a
large part of the total parameter count:

| Preset | Intended use | Main width/depth |
| --- | --- | --- |
| `smoke` | local end-to-end check | 256 / 2+2 blocks |
| `student-400m` | smaller research student | 768 / 10+10 blocks |
| `student-750m` | first rented-GPU candidate | 1024 / 16+16 blocks |
| `student-1b` | upper end of the target range | 1280 / 12+12 blocks |

The script prints the exact parameter count at startup. Treat that printed
count, rather than the preset label, as the value to report.

For a causal-only quality control using the same student family, set
`--canvas-layers 0` and `--diffusion-loss-weight 0`. For a matched-capacity
control, increase `--prefix-layers` until the parameter count and throughput
are comparable. To test the proposed dense-vs-theorizer split, add
`--solidifier-layers 4`; this inserts four extra causal blocks after the
solidifier reads the canvas, making the solidifier substantially larger than
the theorizer. To train against imperfect retained canvases, add
`--rolling-curriculum-prob 0.5 --rolling-curriculum-max-commits 4`. Selected
training batches then append the known token, shift the hidden canvas, refresh
only its new tail, and train the next commitment from that retained state.
Evaluation remains clean; use `rolling_diagnose` to measure retained-state
quality. The pre-GPU comparison and its interpretation are recorded in
[the quality benchmark](../project%20context/08_rfca_quality_benchmark.md).

## 4. Two A100s

After the single-device smoke test and a single-device short run succeed, use
PyTorch distributed training:

```bash
torchrun --standalone --nproc_per_node=2 \
  training/train_rfca_student.py train \
  --model-preset student-750m \
  --tokenizer google/gemma-4-E2B-it \
  --data path/to/train.jsonl \
  --prefix-length 256 \
  --canvas-length 32 \
  --diffusion-steps 2 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --gradient-checkpointing \
  --precision bf16 \
  --steps 10000 \
  --output-dir runs/student-750m-2xa100
```

At the stated `$1.59` per GPU-hour, two GPUs cost about `$3.18` per wall-clock
hour before storage, networking, or provider-specific charges. The first cloud
run should be a short throughput and memory measurement, not a long blind
training job.

## 5. Rolling theorizer/solidifier test

The rolling inference path now keeps the theorizer canvas as continuous hidden
states. It refines the entire canvas, stops at a configurable uncertainty
boundary, lets the solidifier commit one token, shifts the canvas, initializes
the newly exposed tail, and repeats.

The first recommended threshold is entropy-based because that matches the
DiffusionGemma-style stopping idea. Entropy is measured in nats and lower
values are more certain. Start with a substantially looser threshold than
`0.005`, for example:

```bash
python training/train_rfca_student.py generate \
  --checkpoint runs/student-400m/checkpoint-0001000.pt \
  --tokenizer google/gemma-4-E2B-it \
  --prompt "Explain a rolling future canvas." \
  --theorizer-stop entropy \
  --theorizer-entropy-threshold 0.5 \
  --max-theorizer-steps 8 \
  --trace-output runs/student-400m/rolling-trace.json
```

The trace records mean confidence, entropy, normalized entropy, number of
theorizer passes, and whether the threshold was reached. If the threshold is
not reached by `--max-theorizer-steps`, the loop commits using the last
available provisional canvas and records `threshold_stop=false`; that fallback
is intentional and must be reported.

Confidence is also available and is easier to compare across vocabularies:

```bash
--theorizer-stop confidence --theorizer-confidence-threshold 0.50
```

Boundary interpolation is enabled by default to reduce a single-pass jump
past the requested boundary. It operates on hidden states and does not insert
argmax canvas tokens. Disable it only as an explicit ablation with
`--no-boundary-interpolation`.

This is an inference-time rolling test. The training loop still needs a
short-on-policy rolling curriculum so the theorizer learns the distribution
of its own shifted hidden canvases.

To measure that gap directly on a fixed-window checkpoint, use the
teacher-forced rolling diagnostic:

```bash
python training/train_rfca_student.py rolling_diagnose \
  --checkpoint runs/smoke-200/checkpoint-0000200.pt \
  --tokenizer byte \
  --samples 128 \
  --commits 8 \
  --theorizer-passes 2 \
  --output runs/smoke-200/rolling-diagnostics.json
```

The retained path refines the current canvas, commits the known target token,
appends that token to the causal prefix, shifts the hidden canvas by one slot,
and initializes only the new tail. The reset control reinitializes the blank
canvas at every commit. On the current fixed-window smoke checkpoint, retained
accuracy fell from 54.7% on commit 1 to 8.6% on commit 2 and 0% by commit 7,
while the reset control stayed between 36.7% and 43.8%. This is evidence that
the rolling state needs its own training curriculum; it is not evidence that
the rolling architecture is fundamentally wrong.

## 6. Canvas-use diagnostics

Before renting GPUs, run the solidifier sensitivity test:

```bash
python training/train_rfca_student.py diagnose \
  --checkpoint runs/student-400m/checkpoint-0001000.pt \
  --tokenizer google/gemma-4-E2B-it \
  --data path/to/validation.jsonl \
  --samples 512 \
  --output runs/student-400m/canvas-diagnostics.json
```

It compares the same prefixes under full, no-canvas, first-position-only,
pooled, slot-preserving shuffled, and random canvas conditions. The shuffled
control swaps provisional canvas content while keeping each destination slot's
learned position embedding fixed; shuffling complete hidden states would not be
a valid position test. It reports next-token loss,
accuracy, KL divergence from the full-canvas logits, and top-1 agreement with
the full-canvas decision.

Interpretation is deliberately strict:

- Full should beat no-canvas on a held-out set if the canvas contributes useful
  information.
- Random canvas should damage the decision if the solidifier is sensitive to
  canvas content rather than merely using a fixed bias.
- Full should beat pooled and first-position-only if the proposed full-canvas
  interface matters.
- Shuffling should matter on a position-dependent task. If it does not, the
  model may be using only a bag-of-future summary; that is a useful result, but
  it is not evidence for the full RFCA mechanism.

For a purpose-built position probe, use the dependency-free byte task:

```bash
python training/train_rfca_student.py train \
  --task position_probe \
  --model-preset smoke \
  --tokenizer byte \
  --prefix-length 64 \
  --canvas-length 16 \
  --probe-examples 2048 \
  --steps 300 \
  --output-dir runs/position-probe

python training/train_rfca_student.py diagnose \
  --checkpoint runs/position-probe/checkpoint-0000300.pt \
  --tokenizer byte \
  --task position_probe \
  --samples 256 \
  --output runs/position-probe/diagnostics.json
```

The probe places distractors in the early canvas positions and the decision
symbol in the final position. It is designed to test interface behavior, not
to represent language quality.

The current revised probe passes the interface gate locally: the full canvas
reached 0.5859 accuracy, versus 0.3320 with pooled canvas and 0.4688 with
slot-preserving shuffled canvas. Two additional shuffle seeds gave full
accuracies of 0.5391 and 0.5742 versus shuffled accuracies of 0.4414 and
0.4609. The ordinary text smoke corpus did not show a shuffled-language drop,
so a harder sequence or language task is still required before treating this
as evidence about natural-language generation.

The harder ordered-sequence probe uses four words with numeric sort keys. The
theorizer is trained to produce the sorted phrase, while the solidifier emits
one class byte for the exact permutation. Run it with:

```bash
python training/train_rfca_student.py train \
  --task ordered_sequence_probe \
  --model-preset smoke \
  --tokenizer byte \
  --prefix-length 80 \
  --canvas-length 32 \
  --probe-examples 16384 \
  --steps 2000 \
  --output-dir runs/ordered-sequence-probe

python training/train_rfca_student.py diagnose \
  --checkpoint runs/ordered-sequence-probe/checkpoint-0002000.pt \
  --tokenizer byte \
  --task ordered_sequence_probe \
  --samples 1024 \
  --output runs/ordered-sequence-probe/diagnostics.json
```

The local run reached 91.4% full-canvas accuracy versus 28.8% with no canvas
and 22.2% with random canvas, showing substantial canvas dependence. However,
pooled and slot-preserving shuffled canvases still reached 88.2% and 89.5%.
The current theorizer is therefore broadcasting an order summary; this is not
yet evidence that shuffled theorized language collapses.

The indexed final-slot variant reached 98.1% full accuracy versus 72.7% with
no canvas, but pooled and shuffled remained 97.7% and 97.6%. It confirms
canvas use but not slot-specific extraction.

## 7. What this does not claim yet

- It does not reproduce DiffusionGemma's released architecture or weights.
- It does not yet establish language quality with the frozen 26B theorizer;
  the local tiny-checkpoint run is interface-only, and the full checkpoint is
  staged for the A100 experiment.
- The probe supports position-specific interface use, but it does not yet show
  that natural-language generations collapse when theorized canvas order is
  destroyed.
- It does not use an API teacher. Keeping the first student self-contained is
  intentional so a strong causal teacher cannot hide a collapsed or ignored
  canvas.

## 8. Research bookkeeping

For every run, preserve the checkpoint, `run_args.json`, `model_config.json`,
tokenizer artifact, data version/hash, seed, horizon, denoising-step count,
and device information. Do not select a final run based only on the training
loss; record next-token loss, diffusion loss, canvas-ablation results, wall
clock, tokens/second, and peak memory together.

Relevant upstream references:

- [DiffusionGemma overview](https://ai.google.dev/gemma/docs/diffusiongemma)
- [DiffusionGemma mechanics and Transformers interface](https://huggingface.co/docs/diffusers/main/en/api/pipelines/diffusion_gemma)
- [Gemma 4 model overview](https://ai.google.dev/gemma/docs/core)
