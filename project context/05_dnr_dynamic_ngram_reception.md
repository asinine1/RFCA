# Archived: DNR — Dynamic N-Gram Reception

**Status:** superseded on 2026-09-16. This document records the prior
ACA/DNR context-retrieval direction and is retained for research history only.
The active plan is
[03_rfca_architecture_design.md](03_rfca_architecture_design.md).

## 1. What DNR is

**DNR (Dynamic N-Gram Reception)** is an adaptive local receptive-field
mechanism. At each causal position and attention layer, it chooses how many
neighboring token states should be received together as local context.

For layer `l` and position `t`, the router chooses:

```text
n[l,t] in {1, 2, 4, 8, 16}
```

and the local receiver operates on the contiguous causal span:

```text
h[t - n[l,t] + 1 : t]
```

Here “n-gram” means a contiguous sequence of model tokens. It does not
necessarily mean a linguistic word n-gram, because BPE or other subword
tokenizers can split a word into multiple tokens.

The important property is non-uniformity:

- different positions may select different `n`;
- different layers may select different `n` for the same position;
- the same layer may use different n-gram sizes in different regions;
- the fixed local window is only the baseline.

## 2. Relationship to ACA

ACA and DNR operate on different parts of the memory problem:

```text
recent/local context  --> DNR
older compressed memory --> ACA-Low + ACA-High
```

- **ACA-Low** adaptively compresses and sparsely retrieves specific older
  information.
- **ACA-High** adaptively compresses and densely reads broad older structure.
- **DNR** determines how much neighboring recent context is received together
  at the current layer and position.

DNR is therefore a dynamic replacement or generalization of fixed SWA, not a
second ACA channel. It should not be implemented as another long-context
compression pyramid; that would overlap with ACA and make the mechanisms hard
to separate experimentally.

## 3. Candidate DNR operators

### Operator A: adaptive local-attention span

The router chooses `n[l,t]`, and the local attention mask permits the current
position to attend to the previous `n[l,t]` token states.

```text
n=1   -> immediate token only
n=2   -> local bigram span
n=4   -> short phrase/local pattern
n=8   -> larger local structure
n=16  -> extended local dependency
```

**Advantages:** preserves individual token states, is easy to interpret, and
connects directly to learned attention span work. The Adaptive Attention Span
paper is an important precedent for learning context spans while controlling
memory and computation: [Adaptive Attention Span in
Transformers](https://arxiv.org/abs/1905.07799).

**Limitation:** this is adaptive local attention, not necessarily a learned
composition of an n-gram into one new token.

### Operator B: adaptive n-gram pooling

The receiver uses a shared content-aware pooling function over the selected
span and emits a local n-gram representation:

```text
r[l,t] = R_l(h[t-n[l,t]+1 : t])
```

The pooling function must preserve order information through positional
features or an internal causal attention operation. A residual connection to
the current token state should remain so that pooling does not erase exact
local information.

**Advantages:** makes “reception” a distinct representation mechanism and can
reduce the number of local interactions.

**Limitation:** pooling can blur exact boundaries or identifiers. It may also
become a second compression mechanism unless its output size and cost are
strictly controlled.

### Operator C: adaptive local grouping

The receiver partitions a local stream into variable n-gram groups and emits
one representation per group, similar in spirit to ACA's variable-rate
segmentation but restricted to recent/local context.

**Advantages:** potentially reduces the number of local tokens processed by
later attention.

**Limitation:** this overlaps substantially with ACA's adaptive segmentation.
It should be postponed unless the research question specifically becomes
adaptive local token grouping rather than adaptive local reception.

## 4. Recommended DNR definition

Start with Operator A, with a small residual local mixer if needed:

```text
router -> choose n[l,t]
local causal attention over the previous n[l,t] states
residual connection -> current hidden state
```

This gives DNR a clean first claim:

> DNR learns a non-uniform causal local receptive field, measured in token
> n-grams, while ACA handles adaptive compressed long-range memory.

The initial candidate set should be `{1, 2, 4, 8, 16}`. If powers of two are
too coarse for language behavior, a later ablation can add 3, 5, or other
intermediate sizes. The first study should not introduce unnecessary
choices before we know whether the basic adaptive signal exists.

## 5. How DNR chooses n

The router must be causal. At position `t`, it may use the current hidden state
and the prefix, but never future tokens or the answer label.

### Content-conditioned DNR

The router uses local content characteristics such as repetition, boundary
signals, token transitions, and local uncertainty to choose `n`.

This tests whether the model can recognize that some regions need larger local
receptive fields independently of the current retrieval query.

### Query-conditioned DNR

The router uses the current layer's query state to choose `n`. A query that
needs a nearby phrase or syntactic relation can request a larger local
receptive field, while a query that only needs the current token can use a
smaller one.

### Content + query DNR

The router uses both the local prefix and the current query. This is the most
expressive version, but it must be compared against the two single-signal
conditions to show whether both inputs are actually useful.

## 6. DNR and ACA experimental matrix

The first mechanism comparison should hold the model, tokenizer, training
tokens, context length, and hardware fixed:

| Condition | ACA | DNR | Purpose |
|---|---|---|---|
| Fixed baseline | fixed CSA/HCA-style compression | fixed SWA/local span | reference |
| ACA-only | adaptive ACA-Low + ACA-High | fixed local reception | test long-range adaptive compression |
| DNR-only | fixed compressed memory | adaptive n-gram reception | test local adaptive reception |
| ACA + DNR | adaptive ACA | adaptive DNR | test complementarity |

Inside the DNR-only and ACA+DNR conditions, use the content/query factorial
design only after the basic mechanisms are functioning:

| Router condition | Content signal | Query signal |
|---|---:|---:|
| Fixed | off | off |
| Content-only | on | off |
| Query-only | off | on |
| Content + query | on | on |

This keeps ACA and DNR conceptually separate while still allowing the paper to
test whether both mechanisms benefit from the same kinds of conditioning.

## 7. DNR objective and measurements

A DNR training objective can use:

```text
loss = language_or_retrieval_loss
     + lambda_compute * local_attention_work
     + lambda_memory * local_state_cost
     + lambda_switch * n_changes
```

Measure:

- next-token loss and perplexity;
- exact retrieval and distractor retrieval;
- syntactic/local dependency tasks;
- repeated-pattern and boundary-sensitive tasks;
- selected n-gram distribution by layer and position;
- average and tail local receptive field;
- local attention work;
- router overhead;
- wall-clock latency;
- peak memory;
- quality-efficiency frontier.

The model should not receive credit for selecting large n everywhere. If DNR
chooses `n=16` for nearly all positions, it has learned a fixed larger window,
not useful dynamic reception. Report entropy and variation of the n-selection
policy, along with performance under a matched average n budget.

## 8. Main risks

1. **DNR may simply reproduce fixed SWA.** Include fixed windows with the same
   average and maximum n as the adaptive model.
2. **The router may use position shortcuts.** Shuffle task layouts and test
   equivalent examples at different positions.
3. **Pooling may destroy exact local information.** Keep the first version as
   adaptive local attention with residual token states.
4. **ACA and DNR may be redundant.** Compare their error overlap: whether DNR
   fixes local failures that ACA cannot, and vice versa.
5. **Dynamic masks may not be faster in ordinary kernels.** Report theoretical
   local attention work separately from measured runtime, and only claim speed
   improvements if the implementation actually realizes them.
6. **The word n-gram can be misleading with subword tokenization.** Always
   report token-level n and specify the tokenizer.

## 9. Pre-code decisions

1. Confirm that DNR replaces the fixed SWA local path rather than adding a
   second local path beside it.
2. Choose Operator A, B, or C; Operator A is the recommended first version.
3. Lock the initial candidate set `{1, 2, 4, 8, 16}`.
4. Decide whether n is selected per layer x position, per head x position, or
   per query block. Start with layer x position.
5. Define whether the first router is content-conditioned, query-conditioned,
   or both; later run the 2 x 2 factorial.
6. Keep ACA fixed while validating DNR, then run the ACA + DNR interaction
   study.
7. Pre-register average-n budget, maximum n, local compute accounting, seeds,
   and failure criteria.
