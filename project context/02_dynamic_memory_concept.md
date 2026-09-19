# Archived: Dynamic Memory project concept

**Status:** superseded on 2026-09-16. This document records the prior ACA/DNR
context-retrieval direction and is retained for research history only. The
active plan is [02_rolling_diffusion_ar_concept.md](02_rolling_diffusion_ar_concept.md).

## Project facts currently established

- **Researcher:** one student researcher; no team contribution model is
  planned.
- **Grade:** sophomore, so the intended competition path is the Senior Division
  at the regional/state levels, subject to the fairs' eligibility records.
- **Target route:** DMRSEF, with either the direct ISEF nomination route or the
  DMRSEF-to-CSEF-to-ISEF route depending on the result and the 2027 allocation.
- **Compute:** a Mac with 48 GB unified memory is available now. Rented GPUs
  and possible laboratory access are options, not assumptions; they must be
  recorded as support and site decisions if used.
- **Research order:** Dynamic Attention first; Dynamic Reception only after the
  attention phase has a stable baseline, metrics, and interpretation.
- **Adaptivity requirement:** the proposed mechanism must be non-uniform across
  attention layers and query-dependent. A single shared resolution for all
  layers is a control condition, not the target architecture.
- **Project status:** currently treated as a new project. The fact that the
  idea existed before implementation does not establish scientific novelty or
  continuation status. Novelty requires a verified literature comparison, and
  continuation status depends on prior qualifying work in a similar area.

## The idea in one sentence

Dynamic Memory asks whether a causal language model can use its limited
long-context memory more effectively by adapting both **which information gets
memory capacity** and **the resolution at which local context is represented**.

### Part 1 naming

The first mechanism is **ACA (Adaptive Compressed Attention)**:

- **ACA-Low:** CSA-inspired, lower compression and finer adaptive spans from
  `{1, 2, 4, 8, 16}`, followed by sparse retrieval.
- **ACA-High:** HCA-inspired, higher compression and coarser adaptive spans
  from `{32, 64, 128, 256, 512}`, followed by dense global attention.

Both are single variable-rate compressors. A pattern such as
`256-32-32-64-128` describes one ACA-High segmentation, not multiple stacked
compression layers.

Part 2 is **DNR (Dynamic N-Gram Reception)**. DNR controls the local causal
receptive field that remains outside ACA: each layer and position can receive
an adaptive contiguous token span, initially with `n` in `{1, 2, 4, 8, 16}`.
The first DNR version should replace fixed SWA rather than add another local
memory path.

```text
Dynamic Attention   -> WHAT deserves memory capacity?
Dynamic Reception   -> HOW finely should it be represented?
Fully Dynamic Memory -> the two decisions together
```

The project is interesting because these are different decisions. A model may
want to retain an instruction or a retrieval key while representing a highly
predictable span coarsely. Conversely, it may need fine local resolution in a
region that is not globally salient. The point of the study is to test that
distinction, not to assume it is true.

## Working research question

> Under matched model, data, training-token, context-length, and compute
> budgets, does jointly adapting memory allocation and local receptive-field
> resolution improve the long-context quality–efficiency tradeoff of a causal
> language model compared with either mechanism alone or a standard baseline?

This is a stronger and safer question than “Can we build a model with better
memory?” It gives us a negative-result outcome that is still scientifically
useful: one mechanism may help, both may fail, or the apparent gain may
disappear when latency and memory are measured honestly.

## Four-condition experimental matrix

| Condition | Dynamic Attention | Dynamic Reception | Purpose |
|---|---:|---:|---|
| Baseline | off | off | matched reference model |
| DA-only | on | off | isolate adaptive memory allocation |
| DR-only | off | on | isolate adaptive local resolution |
| FDM | on | on | test the joint effect and interaction |

The variable-rate compression study adds a second factorial split inside this
matrix:

| Segmentation condition | Content-conditioned | Query-conditioned |
|---|---:|---:|
| Fixed baseline | off | off |
| Content-only | on | off |
| Query-only | off | on |
| Content + query | on | on |

Content conditioning chooses the variable segment lengths while memory is
formed. Query conditioning lets the current query request refinement or
re-segmentation of selected regions. The combined condition is the intended
“high precision only when necessary” mechanism. These factors must have
separate memory, recomputation, and latency accounting.

All four conditions must share one baseline implementation. Mechanisms should
be selected by configuration, not copied into four model forks. If a mechanism
cannot be implemented with the same parameter, token, and training protocol,
that mismatch must be reported as a limitation rather than hidden.

The matrix is the final comparison, not necessarily the order in which the
project must be built. We will stage the work as:

```text
Phase 0: rules, literature, question, and measurement definitions
Phase 1: baseline + Dynamic Attention only
Phase 2: attention ablations and multi-seed validation
Phase 3: Dynamic Reception as an independent mechanism
Phase 4: matched four-condition matrix, including FDM
```

This respects the attention-first decision while preserving a valid final
factorial comparison.

## Candidate hypotheses

These are hypotheses to test, not claims:

1. **DA hypothesis:** a learned non-uniform memory allocation policy can retain
   retrieval-relevant information with less effective memory use than uniform
   allocation, but only if the policy is causal and its overhead is included.
2. **DR hypothesis:** a learned choice among local receptive-field sizes can
   preserve useful local context with fewer fine-grained representations than
   a fixed-resolution model.
3. **Interaction hypothesis:** DA and DR may be complementary because salience
   and local resolution are not the same variable. The joint model should be
   judged by an interaction effect, not only by whether it beats the baseline.
4. **Null/overhead hypothesis:** routing overhead, optimization difficulty,
   instability, or poor allocation can erase any quality or efficiency gain.

## Proposed operational definitions

We should not code these definitions until we agree that they match the
scientific question:

- **Dynamic Attention** means a causal policy produces a non-uniform decision
  about what each layer reads for the current query: for example, a
  layer-specific SWA length, CSA block selection, or CSA read budget. A single
  decision shared across all attention layers is not sufficient for the main
  claim.
- **Dynamic Reception** means one causal compressor adaptively segments each
  layer's memory stream into variable-length regions: for example, CSA spans
  in `{1, 2, 4, 8, 16}` or HCA spans in `{32, 64, 128, 256, 512}`. A sequence
  such as `256-32-32-64-128` is one dynamic HCA pass, not five compression
  layers. The segmentation decision must be separated experimentally from the
  decision about which compressed blocks are read.
- **Effective memory use** must be defined separately from wall-clock speed.
  A soft gate that multiplies values is not token pruning and must not be
  reported as a real latency reduction.
- **Long-context memory quality** should include next-token loss/perplexity and
  a retrieval task with known answers. Retrieval accuracy alone is not enough;
  next-token performance alone may miss memory behavior.

## Safe first study design

The first research phase should be a controlled, non-human computational
study:

1. Start with deterministic synthetic sequences that test exact retrieval,
   distractors, repeated patterns, and varying context lengths.
2. Add only a verified public-domain or explicitly licensed corpus after the
   data record is complete.
3. Use one small causal model and one training recipe for all four conditions.
4. Hold tokenizer, initialization policy, optimizer, learning-rate schedule,
   batch/effective batch, context lengths, training tokens, and evaluation
   protocol fixed.
5. Run multiple seeds once the pipeline is stable; do not use one favorable
   seed as the result.
6. Measure quality, latency, peak accelerator memory, parameter count, and
   any claimed memory/compression statistic on the same hardware.
7. Save routing diagnostics only for selected runs so the analysis remains
   inspectable without producing unusable tensor dumps.

The first attention phase should answer a narrower question before reception
is introduced:

> Can a causal, inspectable Dynamic Attention policy change token-level memory
> allocation in a way that preserves or improves long-context quality at a
> measured memory/latency cost relative to the same baseline?

The initial implementation should be small enough to run on the Mac for
debugging and unit tests. GPU rental or lab outreach should be used to scale a
pre-registered comparison, not to rescue an unclear question or selectively
search for a favorable result.

## What would count as evidence?

Evidence should be reported as matched comparisons and uncertainty, not as a
single best number:

- validation loss and perplexity;
- exact retrieval accuracy by context length and retrieval depth;
- training throughput and step latency;
- prefill/decode latency under a stated implementation and batch size;
- peak accelerator memory and estimated/actual KV-memory use;
- parameter and active-parameter counts where relevant;
- allocation/reception distributions, router entropy, and layer-wise behavior;
- seed-to-seed variation, failed runs, and any tradeoff between quality and
  cost.

The eventual headline should be a Pareto comparison or a clearly bounded
negative result, not “the model remembers like a human.”

## Claims we should avoid

- “human-like memory,” “human memory replacement,” or general intelligence
  claims;
- medical, cognitive, educational, or safety claims without a completely
  different approved study;
- claiming compression when only a soft weighting was added;
- claiming faster inference without measuring the implementation end to end;
- claiming novelty or superiority before a verified literature review and
  matched experiments;
- presenting a demo or literature review without an actual testable experiment.

## Decisions still open

1. Which exact reference mechanism makes the DA/DR distinction visible while
   staying simple enough to audit?
2. Is the first result a language-modeling study, a retrieval study, or a
   staged combination of both?
3. What is the smallest model/context scale that can expose the effect without
   making results dominated by hardware noise?
4. Which corpus is legally and scientifically appropriate after the synthetic
   phase?
5. Is this an individual project or a team project, and what work will each
   student personally own?
6. Is the work a continuation of any prior language-model or memory project?
7. What hardware and external support will be used, and does that change the
   forms analysis?

Until these are resolved, “Dynamic Memory” is a useful umbrella name, not a
claim that a final architecture has been selected.
