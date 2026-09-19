# Scratchpad

This file is the persistent handoff note for work in this repository. Read it at the start of every conversation and update it at the end of every turn.

## Current state

- No prior scratchpad existed when this file was created.
- The repository contains a research report at `DNR_ACA_MECHANISMS_RESEARCH_REPORT.md` and supporting material under `project context/`.
- The active implementation direction is a from-scratch, Gemma-style RFCA
  student rather than direct training of the released DiffusionGemma weights.
- The first student keeps the causal solidifier self-contained; the proposed
  DeepSeek API teacher idea was explicitly dropped because it could hide an
  ignored or collapsed diffusion canvas.

## Work completed in this turn

- Created `AGENTS.md` with repository instructions.
- Created this `SCRATCHPAD.md` and established the required conversation-start and turn-end workflow.
- Included the repo guidance to be honest, gently critical, and avoid fake products, features, references, or hollow pages.
- Verified both files exist and contain the intended instructions and initial project state.
- Added `training/train_rfca_student.py`, a configurable PyTorch prototype
  with a causal prefix encoder, bidirectional mask-only canvas denoiser, and
  full-canvas next-token solidifier.
- Added `training/README.md`, dependency notes, local smoke/A100 command
  examples, and configs under `training/configs/`.
- Added `project context/07_student_training_plan.md` and linked it from the
  project context README to record the no-API-teacher decision and canvas
  integrity gates.
- Presets estimate to approximately 416M, 877M, and 1.05B parameters with a
  262,144-token Gemma-sized vocabulary; the script prints the exact count at
  runtime.
- Verified the training script with `py_compile` and AST parsing, and checked
  the mask-only canvas construction in both training and evaluation paths.
- A project-local `.venv` contains PyTorch 2.8.0 and NumPy 2.0.2 on arm64
  macOS for the byte-tokenizer smoke path; a separate `.venv-diffusion` now
  contains the newer Transformers stack for frozen DiffusionGemma.
- Ran the documented 20-step synthetic smoke test on MPS. It completed
  training, validation, and checkpoint writing; training loss fell from about
  211.5 to 101.4 and validation loss from about 139.9 to 55.7.
- Loaded `runs/smoke/checkpoint-0000020.pt` successfully and verified its
  saved step, state tensors, model config, and run arguments. A second 2-step
  post-cleanup MPS run also completed without the earlier NumPy/scaler
  warnings.
- Extended the model with a rolling inference path: the theorizer refines a
  continuous full canvas, the solidifier commits one token, `roll_canvas`
  shifts the hidden canvas and initializes a fresh tail, and the loop repeats.
- Added entropy, normalized-entropy, and mean max-probability traces, a
  configurable max theorizer-pass fallback, and optional hidden-state boundary
  interpolation to reduce one-pass overshoot.
- Fixed a canvas-initialization artifact discovered during testing: the blank
  canvas no longer uses the tied mask-token embedding, which could create
  artificial output-head confidence before any theorizer pass.
- Re-trained a fresh 20-step smoke checkpoint at
  `runs/smoke-rolling/checkpoint-0000020.pt` and ran a four-commit rolling
  trace. The full rolling path and trace serialization passed; the `0.5`-nat
  entropy threshold was not reached within four passes, so the expected
  `threshold_stop=false` fallback was exercised. This is plumbing evidence,
  not a quality result.
- Added a `diagnose` command covering full, no-canvas, first-position-only,
  pooled, shuffled, and random canvas variants. It reports next-token loss,
  accuracy, KL divergence from full-canvas logits, and top-1 agreement.
- Trained a longer 200-step local smoke checkpoint at
  `runs/smoke-200/checkpoint-0000200.pt` and evaluated 128 examples. Results:
  full loss 2.2608 / accuracy 0.4062; no-canvas loss 39.4793 / accuracy
  0.2109; first-position loss 2.2748 / accuracy 0.4609; pooled loss 2.3139 /
  accuracy 0.4062; shuffled loss 2.2608 / accuracy 0.4062; random loss
  28.7590 / accuracy 0.0469.
- Interpretation of that first smoke run: the solidifier is using canvas
  information, but this task does not show a benefit from full
  position-specific access. Pooled and shuffled canvases were effectively
  equivalent to full, so the result was not enough to justify GPU hours for
  the full RFCA claim.
- Added a deterministic `position_probe` task after correcting an initially
  impossible probe generator. The probe puts distractors in early canvas
  positions and a decision symbol at the final position, with the final
  symbol as the separate solidifier target.
- The original position-probe shuffle was invalid: it permuted hidden states
  together with their already-attached position embeddings, preserving the
  content-position pairs. The solidifier now explicitly re-injects canvas slot
  identities at its read boundary, and the diagnostic shuffles provisional
  content while keeping destination slots fixed.
- Trained `runs/position-probe-300-v4/checkpoint-0000300.pt` locally for 300
  steps and diagnosed 256 examples. Results: full accuracy 0.5859, no-canvas
  0.1055, first-position 0.3047, pooled 0.3320, slot-preserving shuffled
  0.4688, random 0.0703. Two additional shuffle seeds gave full/shuffled
  accuracies of 0.5391/0.4414 and 0.5742/0.4609. This is initial evidence
  that the solidifier uses position-specific canvas information.
- Re-ran the ordinary text smoke checkpoint with the corrected diagnostic:
  full accuracy was 0.5430 and slot-preserving shuffled was also 0.5430.
  The tiny predictable smoke corpus is therefore not a meaningful
  order-sensitive language gate yet.
- Updated `training/train_rfca_student.py` and documentation to record the
  corrected shuffle semantics and the distinction between the passing
  interface probe and the still-missing language-quality test.
- Added `ordered_sequence_probe`, which trains the theorizer to sort four
  keyed words into one of 24 permutations and trains the solidifier to emit a
  class byte for the exact order. The initial six-class version was impossible
  because the prefix was identical while the canvas target varied; it was
  discarded as a valid experiment. The corrected four-word run is
  `runs/ordered-sequence-probe-sort4-2000/checkpoint-0002000.pt` with
  diagnostics in `runs/ordered-sequence-probe-sort4-2000/diagnostics.json`.
  It reached full accuracy 0.9141, no-canvas 0.2881, random 0.2217, pooled
  0.8818, and slot-preserving shuffled 0.8945. This is strong canvas
  dependence but only a small positional-order effect; the theorizer appears
  to broadcast an order summary.
- Added `indexed_sequence_probe`, where the same keyed sort asks for the final
  word's first byte. The random-query version remained at chance and was not
  useful. The simplified final-query run
  `runs/indexed-sequence-probe-last-2000/checkpoint-0002000.pt` used
  `--solidifier-loss-weight 2` and reached full accuracy 0.9805, no-canvas
  0.7266, random 0.7002, pooled 0.9766, and slot-preserving shuffled 0.9756.
  A second shuffle seed gave full 0.9795 versus shuffled 0.9785. This confirms
  canvas contribution but still does not establish slot-specific extraction.
- The harder sequence probes therefore produced useful PoC numbers but did
  not meet the desired “language collapses when shuffled” gate. Do not rent
  GPUs yet; the next experiment should either constrain the theorizer against
  global broadcasting or define a genuinely multi-token rolling task whose
  continuation cannot be summarized into one invariant vector.
- Re-ran `py_compile` after adding the sequence tasks; the training/diagnostic
  script passes syntax validation.
- Added `rolling_diagnose`, a teacher-forced multi-token rolling evaluator.
  It follows the intended transition exactly: refine the retained canvas,
  score and append the known next token, shift the hidden canvas one slot with
  `roll_canvas`, initialize only the fresh tail, and repeat. It compares this
  against a reset control that starts a blank canvas at each commit. On
  `runs/smoke-200/checkpoint-0000200.pt`, retained accuracy was 0.5469 at
  commit 1, 0.0859 at commit 2, 0.0781 at commit 3, 0.0469 at commit 4,
  0.0156 at commits 5-6, and 0.0000 at commits 7-8. Reset accuracy stayed
  between 0.3672 and 0.4375 after commit 1. The intended rolling mechanics
  work, but fixed-window training does not prepare the model for retained
  hidden canvases.
- Fixed prefix-window overflow in both rolling generation and rolling
  diagnostics by keeping BOS plus the newest committed context when the
  configured causal prefix limit is reached. Syntax validation passed again.
- Ran an eight-token end-to-end rolling generation on
  `runs/smoke-200/checkpoint-0000200.pt`; all eight commits completed with two
  theorizer passes each and the trace was written to
  `runs/smoke-200/rolling-generation-8.json`. The generated text was
  intentionally low quality because this is only a tiny fixed-window smoke
  checkpoint, but the rolling state path completed successfully.
- Decision: the jointly trained from-scratch theorizer is not yet a reliable
  foundation for solidifier experiments. Its retained hidden canvas collapses
  during rolling evaluation, and its sequence probes often broadcast a global
  order summary. The next primary experiment will use a genuinely trained,
  frozen diffusion theorizer while training the autoregressive solidifier.
  This should prevent the two sides from co-adapting around a collapsed canvas.
- Frozen-theorizer prerequisites: identify a checkpoint whose implementation
  exposes continuous canvas hidden states (not only decoded tokens), verify it
  can condition on the committed prefix and accept a shifted canvas, then
  train the separate causal solidifier first with teacher-forced rolling
  prefixes and later with sampled rolling trajectories. Keep the existing
  from-scratch model as a control, not as the main proof instrument.
- Installed a separate Python 3.12 `.venv-diffusion` with PyTorch 2.14,
  Transformers 5.17, Pillow, torchvision, and safetensors for the official
  DiffusionGemma Transformers interface. Added
  `training/requirements-frozen-diffusion.txt` so this environment is
  reproducible without changing the older byte-tokenizer environment.
- Found the real local DiffusionGemma checkpoint at
  `/Users/shrim1729/Models/DiffusionGemma/model/diffgemma-26b-a4b-it-q4`.
  `diffgemma summary` verified the installed 18.84 GiB `.dgq` mmap pack with
  25.82B indexed elements, 256-token canvas, and 2816-dimensional decoder
  hidden state. This is a Rust/Metal inference format, not a Transformers
  checkpoint; the PyTorch script now fails with an explicit explanation if a
  `.dgq` directory is passed to it. A native layer probe was not allowed to
  start while an existing `diffgemma chat` process held the runtime's memory
  budget, and that user process was left untouched.
- Added `training/frozen_diffusion_solidifier.py`. It loads a frozen
  `DiffusionGemmaForBlockDiffusion`, carries both canvas token IDs and
  self-conditioning logits through one-slot rolling updates, runs a fixed
  number of denoising steps with an entropy-bound sampler, and returns frozen
  prefix/canvas hidden states. The default trainable path is now a separate
  causal solidifier with a causal prefix Transformer, tied token embeddings,
  frozen-prefix bridge, continuous-canvas cross-attention, and tied output
  head. A lightweight adapter-only control remains available with
  `--solidifier-mode adapter`.
- Added `smoke`, `student-400m`, `student-750m`, and `student-1b` solidifier
  presets. The local smoke preset is 34,025,985 trainable parameters with the
  262,144-token test vocabulary. Checkpoints save only trainable solidifier
  tensors, never a duplicate frozen language-model head.
- Verified the new script with `py_compile`, a three-commit tiny DiffusionGemma
  probe returning `(1, 32, 16)` canvas hidden states and `(1, 32, 262144)`
  logits, and a one-step two-commit causal-solidifier train run. The saved
  smoke checkpoint has 38 trainable tensors and no frozen `lm_head` tensors.
  The tiny public checkpoint is an interface test only; its random/test-scale
  weights do not provide a language-quality result.
- Also verified the explicit adapter-only control for one commit: it trained
  545 parameters and saved successfully at
  `runs/frozen-solidifier/tiny-adapter-control.pt`.
- The frozen denoiser now defaults to multinomial canvas sampling to match the
  official DiffusionGemma generation path, with `--denoiser-sampling argmax`
  retained as a deterministic ablation. Re-ran the two-commit probe and
  one-step causal smoke training after this change; both passed.
- Added `training/validate_frozen_diffusion_solidifier.py` as a pre-GPU
  integration gate. Its first run deliberately caught a bad test target: the
  validator used the last prompt token, making the tied output head appear to
  achieve zero loss by copying the current token. The target is now selected
  outside the prefix vocabulary.
- The corrected gate then exposed a real architecture issue: plain canvas
  cross-attention was nearly permutation-invariant (`7.6e-6` max shuffle
  delta). Added explicit destination-slot embeddings to the causal
  solidifier's canvas read boundary, matching the project's position
  requirement. The full gate now passes on the tiny checkpoint: canvas removal
  max-logit delta `0.3316`, canvas shuffle delta `0.2151`, 39 trainable
  gradient tensors, frozen-theorizer delta `0.0`, fixed-feature loss
  `27.8936 -> 0.0`, and four retained rolling commits. Results are saved in
  `runs/frozen-solidifier/validation.json`.
- Added a true causal-only quality control to `train_rfca_student.py` via
  `--canvas-layers 0`, plus `--prefix-layers` for matched-capacity controls.
  Trained RFCA and causal controls for 600 updates on ordered-sequence and
  standard synthetic text-window tasks. On ordered sequences, RFCA reached
  0.7832/0.4556 accuracy/loss versus 0.7437/0.5381 for the smaller causal
  control at seed 2027; seed 2028 was 0.7827/0.4571 versus 0.7603/0.4908.
  The matched-capacity causal control reached 0.7720/0.4624, leaving only a
  1.1-point RFCA advantage. On standard text windows, RFCA reached
  0.8569/0.3843 versus 0.9536/0.1340 for the matched causal control. RFCA
  training was roughly twice as expensive against the smaller control and
  similar throughput to the matched control.
- Ran rolling quality diagnostics on the standard RFCA checkpoint. Retained
  accuracy fell from 0.8750 at commit 1 to 0.2969, 0.0781, 0.0859, 0.0547,
  0.1016, 0.0625, and 0.1250 over commits 2-8, generally below the reset
  control. Added the full benchmark writeup at
  `project context/08_rfca_quality_benchmark.md`. Current decision: RFCA has
  a small structured-task benefit but no demonstrated general quality benefit
  and is not yet worth GPU hours.
- Added the quality-control guidance to `training/README.md` and re-ran syntax
  validation for both training scripts plus the frozen-theorizer validator.
  All eight benchmark JSON artifacts load successfully.
- Tested the revised “larger solidifier, smaller theorizer” design by adding
  `--solidifier-layers 4` to `training/train_rfca_student.py`. The extra dense
  causal blocks run after the canvas read, so the solidifier has more capacity
  to interpret the theorizer than the theorizer has to generate the canvas.
- On the standard synthetic byte-window task, the larger-solidifier RFCA model
  reached loss/accuracy `0.1383/0.9551` versus `0.2107/0.9229` for the matched
  dense no-canvas control. RFCA used 7,960,576 parameters versus 6,386,944 and
  ran about 28.0 versus 34.6 steps/s. This is a meaningful one-shot gain, but
  it costs about 1.25x parameters and 19% throughput.
- On the ordered probe, larger-solidifier RFCA reached `0.4965/0.7583` versus
  `0.4741/0.7695` for the dense control. Removing RFCA's canvas reduced
  accuracy to `0.6255`, pooling to `0.7100`, but slot shuffling stayed at
  `0.7603`; canvas dependence is real while strong positional dependence is
  not yet shown.
- Ran the standard rolling diagnostic for the larger-solidifier checkpoint.
  Retained accuracy was `0.9766, 0.4922, 0.4297, 0.2656, 0.1016, 0.2031,
  0.0469, 0.1875` over commits 1–8, versus reset accuracy
  `0.9766, 0.5547, 0.5547, 0.5703, 0.4922, 0.8672, 0.5547, 0.6641`.
  The stronger solidifier improves one-shot quality but does not fix retained
  rolling-state drift. The dense control's retained and reset predictions were
  identical, as expected.
- Updated `project context/08_rfca_quality_benchmark.md` and
  `training/README.md` with the larger-solidifier controls and revised
  interpretation. Current decision: this design merits more local testing on
  real text and rolling curricula, but it is still not enough evidence for
  rented GPU hours.
- Added a teacher-forced retained-canvas curriculum to
  `training/train_rfca_student.py` with `--rolling-curriculum-prob` and
  `--rolling-curriculum-max-commits`. Selected standard-task batches now
  append the known token, shift the hidden canvas one slot, initialize only the
  new tail, and optimize the next commitment from that imperfect retained
  state. The diffusion loss is scored on each available future slice; the
  solidifier loss is scored at the final rolling commit.
- Verified the curriculum path with a four-step MPS training/checkpoint run.
  A 600-step larger-solidifier run at probability `0.5` and max depth `4`
  reached clean diagnostic loss/accuracy `0.2224/0.9312`, below the
  clean-only baseline `0.1383/0.9551`, and ran at 15.7 versus 28.0 steps/s.
- The rolling result improved sharply: curriculum retained accuracy across
  commits 1–8 was `0.9375, 0.9141, 0.9453, 0.8750, 0.9062, 0.8516,
  0.8516, 0.8984`, while the clean-only larger-solidifier model was
  `0.9766, 0.4922, 0.4297, 0.2656, 0.1016, 0.2031, 0.0469, 0.1875`.
  Retained state no longer collapses, although reset remains better at some
  commits. Artifacts are under
  `runs/quality-standard-rfca-large-solidifier-rolling-curriculum-600/`.
- Updated `training/README.md` and `project context/08_rfca_quality_benchmark.md`
  with the curriculum interface and results. The frozen-theorizer script
  already uses a full teacher-forced rolling loop (`--commits`), so the next
  meaningful gate is to repeat this imperfect-state result with compatible
  frozen DiffusionGemma weights and held-out real text.
- Trained a capacity-matched causal control with `canvas_layers=0`,
  `prefix_layers=3`, and `solidifier_layers=4` (`7,436,032` parameters versus
  RFCA's `7,960,576`). Its clean diagnostic loss/accuracy were `0.1862/0.9370`
  versus curriculum RFCA's `0.2224/0.9312`, so the causal control was slightly
  better one-shot. Its rolling retained/reset accuracies were identical:
  `0.9531, 0.5391, 0.5547, 0.6406, 0.4922, 0.8203, 0.5391, 0.6641`.
- Compared with that causal control, curriculum RFCA retained
  `0.9375, 0.9141, 0.9453, 0.8750, 0.9062, 0.8516, 0.8516, 0.8984` across
  commits 1–8. This supports a narrower claim: imperfect-canvas training is
  useful for retaining provisional future state, even though it is not a
  one-shot accuracy win over similar causal capacity.
- Added the capacity-matched causal comparison to
  `project context/08_rfca_quality_benchmark.md`. New artifacts are under
  `runs/quality-standard-causal-matched-rolling-comparison-600/`.
- Extended `training/frozen_diffusion_solidifier.py` with a held-out rolling
  evaluation harness and `--canvas-mode full|none`. The no-canvas control adds
  one causal prefix layer via `--causal-extra-layers 1` so its trainable count
  stays close to the full-canvas solidifier. Training still teacher-forces
  every requested commit, shifts the frozen state, and now writes per-commit
  held-out metrics with `--metrics-output`.
- Ran both frozen harness paths for 20 CPU steps on project-report prose using
  the tiny public checkpoint. The full path had 34,025,985 trainable
  parameters; the causal control had 34,151,808. Both had zero held-out token
  accuracy because the tiny checkpoint is an interface/test model, not a
  language-quality checkpoint. Artifacts are
  `runs/frozen-solidifier/tiny-realtext-rfca.*` and
  `runs/frozen-solidifier/tiny-realtext-causal.*`; do not use these as quality
  evidence.
- Queried the official Hugging Face manifest and downloaded
  `google/diffusiongemma-26B-A4B-it` in Transformers format to
  `/Users/shrim1729/Models/DiffusionGemma/hf/diffusiongemma-26B-A4B-it`.
  Verified `DiffusionGemmaForBlockDiffusion`, 11 safetensor shards, 256 canvas
  slots, 2,816 hidden size, and 262,144 tokenizer vocabulary. The files occupy
  about 48 GiB while the Mac has 48 GiB RAM, so the full checkpoint is staged
  for an A100 run rather than loaded locally. Updated `training/README.md`
  with the full RFCA and causal-control commands.

## Open work / next steps

- Use the already-installed local `.venv` for the existing synthetic smoke
  test, and `.venv-diffusion` for frozen-theorizer experiments.
- Use the revised full/no-canvas/first-position/pooled/slot-preserving-shuffle
  diagnostics as the interface gate before implementing rolling updates or
  on-policy unrolls.
- Sweep the theorizer stopping boundary on a meaningfully trained checkpoint;
  start with entropy thresholds around 0.5, 1.0, and 1.5 nats plus a
  vocabulary-stable mean-confidence control around 0.5.
- Transfer the staged Transformers checkpoint to the GPU host (or mount the
  shared model path), then run the full RFCA and matched causal solidifier
  comparison teacher-forced before trying sampled rolling trajectories.
- Repeat the new teacher-forced rolling curriculum with compatible frozen
  DiffusionGemma weights and a held-out real-text corpus. Then add a harder
  held-out language or structured-sequence task where the order
  of future canvas content affects a multi-token continuation, or constrain
  the theorizer so it cannot broadcast the entire order summary to every
  slot. The current sequence probes confirm canvas dependence but do not prove
  that natural-language generations collapse under canvas shuffling. Do not
  move to rented GPUs until that stronger gate passes, or the research
  question is narrowed accordingly.
- The official tokenizer/model identifier and local download flow are verified;
  the staged checkpoint remains subject to the Gemma terms on the target host.
- Reviewed the current Runpod setup guidance. The recommended first launch is
  one on-demand A100 SXM 80GB pod with a same-region persistent network volume
  mounted at `/workspace`; the frozen harness is single-device, so a second
  GPU would not be used without adding sharding/distributed support. Verify
  the listing says 80GB rather than 40GB before purchase.
- Initialized a new Git repository on `main` and added `.gitignore` rules for
  `.env`, virtual environments, SSH/private-key files, model weights, native
  model packs, generated `runs/`, temporary files, and Python caches. Created
  initial commit `4e9527620e094edaf630165b3d2b611bfdc1a4a0` as
  `Asinine <shrim1729@gmail.com>`; the working tree was clean afterward.
- Run a short 400M single-device trial with the frozen theorizer before moving
  to the 2xA100 configuration. Do not treat the tiny public checkpoint as the
  quality gate or spend GPU hours until the real theorizer path is available.
