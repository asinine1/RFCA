# Rolling Future-Conditioned Autoregression: project concept

**Status:** active pre-code project concept, checked 2026-09-16. This is a
working research direction, not a novelty claim, submission-ready Research
Plan, or experimental result.

## 1. Project direction

The project has been rerouted from adaptive context retrieval to a
rolling diffusion–autoregressive generation architecture.

The central idea is to let a diffusion model maintain an imperfect,
revisable representation of what the model may say next, while an
autoregressive solidifier uses that representation to choose and commit the
next concrete token. The model therefore has a provisional future canvas
while generation remains permanently causal at the commitment boundary.

The working name is **Rolling Future-Conditioned Autoregression (RFCA)**.
This name is provisional. The literature review must determine whether a
better established term already exists and whether the proposed interface is
distinct enough to support a fair-project contribution.

## 2. Proposed generation loop

At time t:

1. The committed prefix consists only of tokens already selected by the
   solidifier.
2. A future canvas of horizon H is initialized or warm-started after that
   prefix. H = 256 is the motivating target, but smaller horizons are
   required for the first implementation.
3. A diffusion denoiser makes a limited number of passes over the canvas.
   The canvas is intentionally provisional rather than perfectly cleaned.
4. The solidifier reads the committed prefix and the entire model-generated
   future canvas, or a learned representation of the entire canvas.
5. The solidifier selects the next token x_t.
6. Only x_t becomes permanent. The future canvas is shifted, refreshed, or
   partially re-noised and the loop repeats.

The important architectural distinction is that the solidifier is
future-conditioned: it is not merely verifying the first token proposed by a
diffusion block. It uses the full provisional future to decide the current
token, while the future itself remains revisable.

## 3. Working research question

Can a rolling hybrid architecture that uses a limited-pass bidirectional
diffusion future canvas to condition an autoregressive next-token solidifier
produce better quality or reasoning-relevant behavior than matched-budget
autoregressive, block-diffusion, and simpler future-canvas controls?

The question should be narrowed after pilot measurements. “Better” must be
defined with measurable quality, consistency, compute, and memory metrics. A
claim that the architecture is more human-like or that it visualizes its
future is not an acceptable scientific endpoint by itself.

## 4. Why this is worth investigating

Standard autoregressive generation commits one token at a time and has no
explicit revisable representation of a longer immediate future. Diffusion
language models can revise many positions in a canvas, but their relationship
to causal token-by-token commitment is different. RFCA makes the interface
between those behaviors the object of study:

- the diffusion side supplies non-causal provisional structure;
- the solidifier supplies a causal commitment rule;
- the rolling update determines how provisional structure survives each
  commitment;
- the training curriculum determines whether the solidifier learns to use
  imperfect canvases rather than clean future labels.

This is a narrower and more architecture-focused contribution than the
superseded ACA/DNR project.

## 5. Prior-art boundary

The broad idea is not unclaimed territory. Relevant work includes
DiffusionGemma's causal prefix plus bidirectional canvas, Block Diffusion's
diffusion inside autoregressive blocks, TiDAR's diffusion-style drafting with
autoregressive sampling, DiffCoT's sliding-window diffusion-style reasoning,
Deferred Commitment Decoding, and Diffusion Forcing.

The possible contribution is therefore not “the first model to combine
diffusion and autoregression.” The candidate contribution is a specific
rolling interface in which a full provisional future canvas conditions a
next-token solidifier, followed by a defined canvas update and a matched
training/evaluation protocol. This claim is provisional and must be checked
against the full papers before it is used in an abstract or display.

## 6. Mathematical sketch

Let x_<t be the committed prefix and C_t^(k) be the future canvas after k
denoising passes:

~~~text
C_t^(k+1) = D_theta(x_<t, C_t^(k), k)
p_phi(x_t | x_<t, C_t^(K)) = G_phi(x_<t, R(C_t^(K)))
C_(t+1)^(0) = U(C_t^(K), x_t, noise, position)
~~~

Here D is the diffusion denoiser, R is the interface that exposes the full
canvas to the solidifier, G is the autoregressive solidifier, and U is the
rolling canvas update. The implementation must define all four components
explicitly; otherwise the idea collapses into an underspecified “draft then
decode” system.

## 7. Causality and leakage rules

- The solidifier may read the committed prefix and the model-generated
  canvas.
- It may not read clean future target tokens at inference or in the main
  training path.
- Training canvases must be produced by the diffusion model, corrupted from
  valid model states, or generated by a clearly labeled synthetic control.
- A clean-ground-truth canvas may be used only as a separate diagnostic, never
  as evidence for the main architecture.
- The committed prefix is the only permanent state used to define the next
  causal step.

## 8. Initial boundaries

The first version should use public or synthetic text data, no human
participants, no private data, and no medical or biological application.
The first model should be small enough to run reproducibly on available
hardware or a documented rented-GPU budget. DiffusionGemma-scale training is
not an initial implementation assumption.

The first horizon should be substantially smaller than 256 if necessary. The
project should measure whether the mechanism survives scaling rather than
spend the entire project budget on an impractical horizon.

## 9. Go/no-go gates

Proceed only if:

- a diffusion-only canvas can be generated reproducibly;
- the solidifier can learn from imperfect model-generated canvases;
- the rolling update can run without hidden clean-future information;
- matched-budget baselines are implemented;
- quality and compute can be measured on the same tasks.

If full rolling RFCA is too expensive, the fallback is a smaller horizon,
fewer denoising passes, partial canvas refresh, or a block-commit variant.
The fallback must preserve the research question rather than silently become
ordinary speculative decoding.
