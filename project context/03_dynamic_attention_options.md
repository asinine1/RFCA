# Archived: Dynamic Attention design space

**Status:** superseded on 2026-09-16. This document records the prior
attention/context-retrieval direction and is retained for research history
only. The active plan is
[03_rfca_architecture_design.md](03_rfca_architecture_design.md).

**Architecture clarification:** the current project direction is now a
DeepSeek-V4-inspired hybrid of sliding, compressed-sparse, and
heavily-compressed attention. The generic token-level retention option later
in this document remains a useful alternative and ablation idea, but it is not
the current recommended architecture. See
[04_deepseek_inspired_architecture.md](04_deepseek_inspired_architecture.md).

## Start with the distinction

In an autoregressive Transformer, token (t) normally reads keys and values
from the causal prefix (0, ldots, t). During generation, those key/value
vectors are cached, so the cache grows with context length and each new token
can attend to a long list of previous entries.

“Dynamic Attention” can mean two different things:

1. **Dynamic weighting:** change the attention distribution while still doing
   essentially the same computation over the same memory.
2. **Dynamic allocation:** change which memory entries, heads, spans, or
   representations are actually retained or processed.

The first is easier to train and useful for understanding behavior. The second
is required if we want to claim reduced memory or compute. The project should
measure them separately.

## Option family A: change attention weights only

### A1. Soft salience gate on keys and values

A learned causal scorer produces one gate (g_t) per token. The model uses
(g_t K_t) and/or (g_t V_t), or incorporates the score as a logit bias.

**Advantages:** simple, differentiable, easy to inspect, and a good first
interface test.

**Problems:** multiplying a vector by zero does not automatically avoid the
cost of computing, storing, or loading that vector. A soft gate is therefore a
quality/behavior experiment and an *effective-use proxy*, not evidence of real
latency reduction.

This is the lowest-risk way to start coding, but it must never be presented as
actual KV-cache compression unless a real retained-entry implementation is
added and benchmarked.

### A2. Attention-logit bias or suppression

The policy adds a learned penalty to some query-key pairs before softmax. This
can encourage the model to ignore low-salience positions without altering the
cache.

**Advantages:** preserves the usual value representations and is flexible.

**Problems:** it is still generally full attention. It may learn a useful
attention pattern, but it does not by itself reduce memory traffic.

### A3. Learned attention span

Each head or layer learns how far backward it should attend, usually through a
soft mask that becomes a bounded causal span. Adaptive Attention Span is a
canonical example: it learns a context size per attention head while targeting
memory and computation control. [Adaptive Attention Span in
Transformers](https://arxiv.org/abs/1905.07799)

**Advantages:** interpretable, causal, and easier to make efficient than an
arbitrary token selector.

**Problems:** the decision is often one span per head/layer, not a different
choice for every token. It answers “how far back does this head look?” more
than “which particular tokens deserve memory?” It may therefore overlap with
Dynamic Reception unless we define the boundary carefully.

## Option family B: retain or discard memory entries

### B1. Token-level learned pruning

The model scores tokens and removes low-scoring tokens from later layers or
from the active sequence. A threshold can produce a variable sequence length;
top-(k) can enforce a fixed budget. Learned Token Pruning is an example of
adaptive token removal inside a Transformer. [Learned Token Pruning for
Transformers](https://arxiv.org/abs/2107.00910)

**Advantages:** directly changes sequence length and can reduce attention
compute when the implementation supports ragged or packed sequences.

**Problems:** removing a token may destroy information needed later; selection
and repacking have overhead; discrete decisions complicate training; and a
naive dense implementation may still perform the original work and only mask
the result.

### B2. Dynamic KV-cache eviction during decoding

At generation time, the model keeps a bounded cache and evicts entries as new
tokens arrive. The policy may use recency, attention history, learned
importance, or a combination.

This is the most direct interpretation of “what should be remembered?” for
long-context inference. H2O dynamically balances recent tokens with
heavy-hitter tokens, while Scissorhands retains tokens judged important at
earlier steps. [H2O](https://arxiv.org/abs/2306.14048),
[Scissorhands](https://arxiv.org/abs/2305.17118)

**Advantages:** fixed memory budget, direct KV-cache measurement, natural
streaming experiment, and clear recency/random baselines.

**Problems:** it primarily addresses autoregressive decoding, not ordinary
full-sequence training; a token that looks unimportant early may matter later;
the policy can be query-dependent; and eviction overhead must be included in
latency measurements.

### B3. Write gating

When token (t) is produced, a causal controller decides whether its key/value
is written to long-term memory at all. A recent short-term buffer can be kept
separately so the model does not lose local continuity.

**Advantages:** very clear memory semantics and potentially low cache use.

**Problems:** early write mistakes are irreversible unless another summary
path exists; the policy must be causal at the exact write time; and training a
binary write decision is difficult.

### B4. Query-dependent retrieval from a memory index

Instead of scanning every cached entry, the current query retrieves a subset of
past entries from an index or approximate nearest-neighbor structure.

**Advantages:** can make read cost depend on the number of retrieved entries
instead of the full context.

**Problems:** index construction and lookup become part of the architecture;
approximate retrieval can miss keys; GPU implementation is nontrivial; and the
project starts to resemble external-memory retrieval rather than a standard
Transformer attention module.

## Option family C: allocate by structure rather than token

### C1. Head- or layer-level allocation

Some heads or layers receive a full long-context cache while others use a short
recent window. DuoAttention is an example that separates retrieval heads from
streaming heads and gives them different KV-cache policies. [DuoAttention](https://arxiv.org/abs/2410.10819)

**Advantages:** easier to implement efficiently than per-token routing; fewer
discrete decisions; and it can reduce both memory and latency when the kernel
supports the two cache types.

**Problems:** the policy may be mostly global rather than input-adaptive. It
answers “which heads need long memory?” rather than “which tokens deserve
memory?” It is a strong baseline or an orthogonal ablation, but it may not
fully satisfy the intended Dynamic Attention definition.

### C2. Block- or span-level sparse routing

The context is partitioned into blocks. A query or controller selects a subset
of blocks, such as recent blocks plus a few salient older blocks.

**Advantages:** block sparsity can map better to hardware than arbitrary token
sparsity; selection is cheaper; and the retained unit is easy to visualize.

**Problems:** block boundaries can be destructive; selecting a block retains
unimportant neighbors; and the sparse attention kernel is part of the result.

### C3. Recent-plus-global memory

Always retain a recent window and reserve a small global memory budget for
salient entries. This gives the model continuity plus sparse long-range access.

**Advantages:** safer than pure eviction and straightforward to compare with
recency-only memory.

**Problems:** it introduces a hand-designed policy before the learned policy;
the global entries may become attention sinks rather than semantically useful
memory; and the two budgets become additional experimental variables.

StreamingLLM demonstrates that retaining initial attention-sink tokens together
with a recent window can stabilize streaming generation, but an attention sink
is not necessarily information that is semantically important. [Efficient
Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)

## Option family D: compress instead of discard

### D1. Merge similar tokens

Several past token representations are combined into one or more summary
vectors, often using similarity or importance. The cache stores fewer entries
but does not simply drop all information.

**Advantages:** graceful degradation and better protection against a single
bad keep/drop decision.

**Problems:** merging changes the representation itself. That makes it partly
Dynamic Reception, not pure Dynamic Attention. It also requires a fair rule for
counting summary vectors, merge operations, and memory bandwidth.

### D2. Hierarchical or compressed memory

Keep a short fine-grained memory plus a longer coarse-grained memory. Older
activations are compressed using pooling, convolution, learned transforms, or
another summary operator. The Compressive Transformer is a canonical example
of a primary memory plus compressed past memory. [Compressive Transformers for
Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507)

**Advantages:** naturally represents different memory resolutions and may be a
good bridge to Dynamic Reception.

**Problems:** it combines the questions “which information?” and “at what
resolution?” too early for an attention-first study. It should probably be
deferred until the independent mechanisms are understood.

### D3. Variable precision or quantization

Important entries use more bits while less important entries use fewer bits.

**Advantages:** can reduce bytes without dropping entries.

**Problems:** the variable is representation precision, so it overlaps with
Dynamic Reception; quantization error and hardware support become major
confounders; and it is not the cleanest first test of salience allocation.

## Option family E: route the entire attention pattern

### E1. Dynamic local/global pattern selection

For each layer, head, block, or token, the model selects among local, global,
dilated, or sparse attention patterns.

**Advantages:** can target computation directly and provides interpretable
patterns.

**Problems:** it changes both who is read and how far the model can read. That
can be difficult to separate from receptive-field adaptation.

### E2. External memory or recurrent state

The model writes a bounded state, recurrent summary, or learned memory slots
and later attends to those instead of all historical tokens.

**Advantages:** memory can be bounded independently of context length.

**Problems:** the memory-writing mechanism becomes the entire research
problem; comparisons to a standard Transformer become less direct; and it
risks becoming a different architecture rather than an ablation of attention.

## Cross-cutting choices

The option families above can be combined with several independent choices.

### Decision granularity

| Granularity | Example | Scientific meaning |
|---|---|---|
| Global | One context budget for the whole model | Does a fixed budget help? |
| Layer/head | Each head learns a span or cache policy | Which parts of the model need long memory? |
| Sequence | One budget per input sequence | Do easy and hard examples need different capacity? |
| Span/block | Keep or drop chunks | Can hardware-friendly groups be routed? |
| Token | Keep/drop individual KV entries | Which specific information deserves memory? |
| Query-dependent | Different queries read different past entries | What is relevant to the current prediction? |

Token-level routing is closest to the stated idea, but it is also the hardest
to make fast. Head/layer or block routing may be better first baselines.

### Budget policy

- **Fixed (k):** retain exactly (k) entries or a fixed fraction. Cleanest
  for matched experiments and Pareto curves.
- **Fixed recent floor plus learned remainder:** always keep the last (r)
  entries and allocate the remaining budget by salience. Safer for generation.
- **Learned threshold:** retain entries above a learned or calibrated score.
  Sequence lengths vary, but the actual budget varies too.
- **Penalty-based:** let the model choose and add a cost term for memory use.
  Flexible, but quality/cost weighting becomes a major hyperparameter.
- **Latency or byte budget:** optimize a direct hardware constraint. Most
  realistic, but requires reliable measurements and hardware-specific work.

For the first study, fixed budgets are preferable. They make DA-only versus
baseline comparisons interpretable and prevent a model from appearing better
simply because it used more memory.

### Training method

- **Continuous gate:** easiest and differentiable; no true sparsity guarantee.
- **Soft-to-hard schedule:** train with a soft gate, then anneal toward binary
  retention; simple but can change behavior late in training.
- **Straight-through estimator:** use hard decisions in the forward pass and a
  surrogate gradient; practical but the gradient is biased.
- **Threshold/top-(k) with auxiliary loss:** enforce a budget or sparsity
  target; requires tuning the auxiliary objective.
- **Distillation:** train a compressed model to imitate full-attention outputs;
  adds a teacher and can mask where the policy fails.
- **Policy-gradient or reinforcement learning:** handles discrete decisions
  directly; expensive, noisy, and probably too much for the first study.
- **Post-training policy:** keep a pretrained model fixed and learn only the
  cache controller; useful for isolating inference-time memory, but not the
  same as end-to-end architectural training.

## Causality requirements

Any candidate must satisfy all of these:

1. The decision for position (t) may use only information available at or
   before (t), never tokens (>t).
2. A token evicted at time (t) cannot reappear later unless a causal summary
   of it was stored.
3. A policy using attention scores from a query must apply those scores only to
   already-processed memory; it cannot use a future query to justify a past
   decision during evaluation.
4. Training-time full-sequence masks and inference-time incremental cache
   updates must be tested separately.
5. Selection, indexing, packing, cache movement, and fallback recency memory
   must be included in the cost accounting.

The most dangerous invalid result would be a policy that looks efficient only
because it saw the complete sequence while deciding what the earlier tokens
should remember.

## What should be compared

At minimum, every DA result should be compared against:

- full causal attention with the same model and training budget;
- a uniform fixed-budget sparse policy;
- a recency-only policy with the same budget;
- a random-retention policy with fixed seeds;
- the learned policy at several budgets;
- an oracle or offline upper bound only if its future-information advantage is
  clearly labeled and never used as a deployable method.

Quality should include validation loss/perplexity and exact retrieval accuracy
by context length and retrieval depth. Efficiency should include retained KV
entries, actual cache bytes, prefill latency, decode latency, peak accelerator
memory, and selection overhead. A soft gate can report salience and effective
retention, but not actual speedup.

## Recommended attention-first path

For this project, I recommend a staged path rather than committing to one
ambitious mechanism immediately:

### Stage 1: causal salience interface

Add a small learned scorer that produces a per-token salience signal. Use it as
a soft gate or logit bias only for debugging and analysis. Verify shape,
causality, reproducibility, and diagnostics. Label the result as a reference
allocation proxy.

### Stage 2: fixed-budget retention

Implement a genuine bounded KV cache for incremental decoding. Keep a stated
recent floor and allocate the remaining slots using the learned salience
score. Compare it with full-cache, recency-only, uniform, and random policies
at the same budget.

### Stage 3: policy ablations

Vary only one factor at a time:

- token versus block selection;
- fixed fraction versus fixed count;
- recent floor size;
- layer/head placement;
- soft, threshold, or top-(k) decisions;
- trained policy versus post-training policy.

### Stage 4: reception

Only after the attention policy has a stable result should we introduce
Dynamic Reception. Compression, merging, variable precision, and hierarchical
memory should be reserved for the later interaction study unless the
attention results show that one is essential.

The recommended first *scientific* candidate is therefore **token-level causal
KV retention with a fixed budget and a recency floor**, preceded by a soft
salience reference for debugging. It is not necessarily the final architecture;
it is the cleanest candidate for testing the project's central “what should be
remembered?” question.

## Questions we should answer before coding

1. Is the primary claim about training-time attention cost, decoding-time KV
   memory, or both?
2. Should the first hard policy be token-level or block-level for a fair
   hardware measurement?
3. What minimum recent window is required to make generation stable?
4. Should the salience score be learned end-to-end, learned after pretraining,
   or both as separate experiments?
5. What is the fixed resource budget: number of KV entries, bytes, attention
   operations, or measured latency?
6. Which retrieval task will force the policy to preserve an old, non-recent
   answer rather than merely exploit recency?
