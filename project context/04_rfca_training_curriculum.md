# RFCA training curriculum

**Status:** active training plan, checked 2026-09-16. The sequence is
intended to make the solidifier learn from the same imperfect future canvases
it will receive during generation.

## 1. Training principle

Begin with an already-trained or separately trainable diffusion language
model. First teach a solidifier to interpret limited-pass, model-generated
canvases while the diffusion component is frozen. Only after the interface is
stable should the two components be tuned together.

The diffusion pass budget must be deliberately limited or sampled from a
range. If every canvas is fully cleaned before the solidifier sees it, the
solidifier will not learn the intended provisional-future problem.

## 2. Phase 0: establish the diffusion baseline

Train or obtain a small diffusion language model that can generate
reproducible canvases. Measure canvas quality as a function of horizon,
initialization, and denoising steps. Save the exact checkpoint, tokenizer,
data version, and sampling settings.

Required outputs:

- diffusion-only generation samples;
- per-step loss and validation metrics;
- canvas quality versus denoising budget;
- memory and wall-clock cost;
- a reproducible canvas-generation script.

## 3. Phase 1: frozen diffusion, off-policy solidifier

Freeze the diffusion model. Generate training examples by conditioning on a
real prefix, creating a canvas from the diffusion model, and asking a
separate solidifier to predict the next ground-truth token.

Sample denoising levels rather than selecting one fixed number:

- very noisy canvases;
- intermediate canvases;
- nearly cleaned canvases;
- different horizons and initialization policies.

The target for the solidifier is the next token, not the clean future canvas.
The canvas must be created without inserting the target suffix as readable
content. A clean target may be retained only for an explicitly labeled
diagnostic control.

Compare first-position-only, pooled, and full-canvas interfaces in this phase.
This isolates whether the full-canvas read is doing useful work before
rolling generation introduces additional error.

## 4. Phase 2: short on-policy rolling training

Use the solidifier's own committed tokens to build short rolling trajectories.
At each step, generate or update the canvas from the model's current prefix,
then train on the next commitment.

Start with short unrolls, such as 4 to 16 committed tokens. Increase the
unroll length only when the system remains stable. Keep teacher-forced
prefixes as a control, but report on-policy results separately.

This phase tests exposure bias, canvas drift, and whether the solidifier
actually uses future information after the prefix leaves the training
distribution.

## 5. Phase 3: adapter and interface tuning

Unfreeze only the canvas-to-solidifier adapter, projection, or selected
cross-attention layers. Keep most of the diffusion backbone frozen. This
limits catastrophic forgetting and makes the source of improvements easier
to interpret.

Track both objectives:

- diffusion denoising quality;
- next-token commitment quality.

If the adapter improves commitment only by causing the canvas to become a
hidden teacher-forced channel, reject the run as leakage or interface
failure.

## 6. Phase 4: joint tuning

Jointly tune selected diffusion and solidifier parameters only after the
earlier phases have passed their gates. A starting objective is:

~~~text
L_total = lambda_diff * L_diff
        + lambda_solid * L_solid
        + lambda_roll * L_rolling
~~~

L_diff preserves the diffusion objective, L_solid trains next-token
commitment from the current canvas, and L_rolling is the short on-policy
trajectory loss. Sweep the weights in a small, documented range rather than
choosing a value after seeing the test results.

Use separate validation data for selecting the curriculum and reserve a
held-out test set for the final comparison.

## 7. Leakage and reproducibility controls

- Do not construct the main canvas by copying the clean target suffix.
- Do not let the solidifier access labels, future-token IDs, or target-derived
  embeddings through an unlogged preprocessing path.
- Log the random seed, checkpoint, horizon, denoising steps, canvas
  initialization, update policy, and training mode for every run.
- Keep an explicit clean-canvas diagnostic separate from the main result.
- Evaluate the solidifier with canvases generated at inference time, not only
  with training-time cached canvases.

## 8. Curriculum gates

Advance from Phase 1 to Phase 2 only if the full-canvas solidifier beats or
matches the simpler interfaces at matched training and inference budgets.
Advance to Phase 3 only if short on-policy rollouts do not collapse.
Advance to Phase 4 only if diffusion quality and commitment quality can be
tracked independently.

If joint training is unstable, the frozen-diffusion plus adapter system is a
valid project endpoint. The project should not force joint training merely
because it sounds more complete.
