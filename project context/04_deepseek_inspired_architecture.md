# Archived: DeepSeek-inspired adaptive multiresolution attention

**Status:** superseded on 2026-09-16. This document records the prior
ACA/context-retrieval direction and is retained for research history only. The
active plan is
[03_rfca_architecture_design.md](03_rfca_architecture_design.md).

## Part 1: ACA

The first mechanism is **ACA: Adaptive Compressed Attention**. ACA has two
compressed-memory channels:

- **ACA-Low:** the CSA-inspired lower-compression, higher-resolution channel.
  It uses adaptive segment lengths in `{1, 2, 4, 8, 16}` and sparse retrieval
  over the resulting summaries. Its job is to preserve query-specific or
  exact information.
- **ACA-High:** the HCA-inspired higher-compression, lower-resolution channel.
  It uses adaptive segment lengths in `{32, 64, 128, 256, 512}` and dense
  attention over the resulting summaries. Its job is to preserve broad
  structure, instructions, organization, and global context.

“Low” and “High” refer to compression level, not importance: ACA-Low has a
lower compression ratio and therefore finer resolution, while ACA-High has a
higher compression ratio and therefore coarser resolution.

Each channel is one adaptive variable-rate compressor. For example, ACA-High
may produce `256-32-32-64-128` in one layer and a different segmentation in
another. It is not five compression layers, and it does not maintain five
separate copies of every context region.

The local recent-memory path remains separate:

```text
context
  |
  +-- recent local memory --> SWA
  |
  +-- ACA-Low --------------> adaptive fine segments -> sparse retrieval
  |
  +-- ACA-High -------------> adaptive coarse segments -> dense global view
```

Part 1 therefore asks whether adaptive variable-rate compression in ACA-Low
and ACA-High improves the quality-efficiency frontier over fixed CSA/HCA-style
segmentation under matched budgets. Dynamic SWA and any later reception or
refinement mechanisms should be treated as separate parts until ACA itself is
validated.

## 1. What the reference architecture actually gives us

The useful reference is DeepSeek-V4's hybrid attention design. The official
model card describes three complementary behaviors:

- **Sliding-window attention (SWA):** keep a recent region at comparatively
  high resolution for local syntax and short-range dependencies.
- **Compressed sparse attention (CSA):** compress the longer history along the
  sequence dimension, use a learned indexer to score compressed blocks, and
  retrieve only the most relevant blocks for the query.
- **Heavily compressed attention (HCA):** compress the history much more and
  attend densely over the resulting short sequence, providing a cheap global
  structural view.

The official Transformers configuration exposes these as
`sliding_attention`, `compressed_sparse_attention`, and
`heavily_compressed_attention`. Its documented defaults use a CSA compression
rate of 4 and an HCA compression rate of 128; the implementation also uses a
layer schedule rather than treating every layer as identical. The exact
configuration should be pinned when we build a baseline, because changing
layer types, local-window settings, compression rates, or retrieval `top-k`
can change the result independently of the proposed router.

Sources:

- [DeepSeek Transparency Center](https://www.deepseek.com/en/transparency/)
- [DeepSeek-V4 official model card](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf)
- [Hugging Face DeepSeek-V4 documentation](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)
- [Hugging Face Transformers DeepSeek-V4 configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py)

The project should not currently state “DeepSeek is SOTA for long-context
retrieval” as an established premise. The official sources establish the
architecture and report its capabilities; “SOTA” is a benchmark-specific
claim that we must verify against a named task, dataset, metric, model,
context length, and compute budget.

## 2. Your proposed modification

The proposed system keeps the three-path idea but makes the resolution of each
path adaptive:

| Path | Reference behavior | Proposed choices | Intended signal |
|---|---|---|---|
| SWA | Fixed recent high-resolution window | Dynamic local length, initially from a bounded discrete set | How much recent context the current query needs |
| CSA | Moderately compressed history, then sparse retrieval | Segment length `s_CSA` in `{1, 2, 4, 8, 16}` | Query-specific or exact-item retrieval value |
| HCA | Uniform heavy compression and dense global attention | Segment length `s_HCA` in `{32, 64, 128, 256, 512}` | Broad structural/contextual value |

Here `s` is the number of source tokens assigned to the next compressed
segment. This is the important interpretation: the values are not five
resolution layers or five copies of the cache. A single adaptive compressor
could partition a region as `256-32-32-64-128`, emit one summary per segment,
and then continue scanning. The same compressor is called with different
segment lengths. A separate attention layer may produce a different partition
of the same history.

## 3. The important terminology correction

There are two different adaptive decisions:

1. **Dynamic attention:** which representations a query reads. Examples are
   the SWA window selected for the query and the CSA compressed blocks chosen
   by the learned indexer.
2. **Dynamic reception/resolution:** how much information is present in each
   representation before attention reads it. Examples are the CSA and HCA
   compression factors.

Therefore, changing the CSA or HCA segment length is not purely an attention
change. It changes the information available to the attention operation. This
is not a problem; it gives the project a clean staged hypothesis:

- **Attention-first:** keep compression fixed, then learn which local span and
  which compressed blocks to read.
- **Reception-second:** keep the read policy fixed, then learn the compression
  resolution supplied to each path.
- **Full model:** allow both policies to adapt and test whether the combined
  gain exceeds the cost and complexity.

This distinction will make the eventual ablation defensible at a science fair.

## 4. Proposed router semantics

Use two named, separately measured router signals rather than the vague word
“specificity.”

### 4.1 Specific retrieval value for CSA

The CSA router should estimate whether a history region contains information
that the current query may need to recover specifically. Suitable evidence
includes the query-to-block score from the learned indexer, agreement between
multiple lightweight query projections, and whether a finer representation
would improve a held-out retrieval or next-token objective.

The intended behavior is:

- high specific-retrieval value -> smaller `s_CSA` and/or more CSA blocks read;
- low specific-retrieval value -> larger `s_CSA` and/or fewer blocks read.

An isolated exact key, identifier, or password is precisely the kind of item
that should be protected by the local or CSA-specific path. HCA should not be
expected to preserve an isolated exact string by itself.

### 4.2 Structural/contextual value for HCA

“General specificity” is understandable as an intuition, but it is not yet an
operational variable. A better name is **structural density** or **global
context value**: how much the region contributes reusable organization,
instructions, topic transitions, constraints, or long-range relationships.

The initial hypothesis should be:

- high structural density -> smaller `s_HCA` such as 32 or 64, preserving more
  coarse-grained structure;
- low structural density -> larger `s_HCA` such as 256 or 512, accepting a
  rougher summary.

This keeps the division of labor coherent: CSA protects query-specific facts,
while HCA preserves a broad map of the context. The router must not be given a
handwritten label saying that a span is “structural”; it should learn under a
quality-plus-cost objective, and we can use span types only for post-hoc
interpretability checks.

## 5. The intended mechanism: one variable-rate compressor

The central mechanism is **adaptive segmentation**, not a resolution pyramid.
For each attention layer, one compressor scans the older sequence and chooses
the length of the next segment. It emits one compressed key/value representation
for that segment, then advances to the next boundary.

For HCA, a fixed compressor might produce:

```text
128 | 128 | 128 | 128 | 128 | ...
```

Your adaptive compressor might instead produce:

```text
256 | 32 | 32 | 64 | 128 | ...
```

The total span covered remains explicit. The second layout uses a finer
representation in regions judged to need it and a coarser representation in
regions that can tolerate more information loss. HCA then densely attends over
all of those variable-length summaries.

CSA uses the same basic variable-rate segmentation idea with its smaller
candidate lengths:

```text
16 | 4 | 1 | 8 | 2 | ...
```

The resulting CSA summaries are then scored by the sparse indexer, and only
the highest-scoring summaries are read. `s=1` means that a segment is retained
at token resolution; it is not a separate compression layer.

Formally, for layer `l`, the compressor creates boundaries
`a[l,0], a[l,1], ...` and segment lengths
`s[l,j] = a[l,j+1] - a[l,j]`. A shared compressor produces:

```text
z_CSA[l,j] = C_CSA[l](K/V[l, a[l,j] : a[l,j] + s_CSA[l,j]])
z_HCA[l,j] = C_HCA[l](K/V[l, a[l,j] : a[l,j] + s_HCA[l,j]])
```

The functions `C_CSA` and `C_HCA` are each one compressor, reused across the
stream. The learned router selects `s[l,j]`; it does not select among several
stacked compression modules.

The required dynamic behavior is therefore:

- different segments in the same layer can receive different lengths;
- different layers can produce different boundaries for the same source
  sequence;
- different examples and context regions can produce different segmentations;
- the fixed `128-128-128` layout is only a baseline/control.

## 6. How the variable-rate decision is made

There are two possible meanings of “resolution needed,” and they should be
separated in the research plan.

### Content-conditioned segmentation

At the time a segment is formed, a causal length router examines the available
prefix and chooses its next length. It can use information density, local
structure, repetition, instruction-like patterns, and other features of the
segment's contents.

This is the cleanest first implementation of your idea. It has one dynamic
compressor, real variable-size memory, and no requirement to store five copies
of every span. Its limitation is that it cannot know the exact future query.

### Query-conditioned segmentation

The current query influences how an old region should be segmented. This is
closer to the phrase “high precision only when necessary,” especially for an
isolated fact that one query needs but most queries do not.

However, if a region was already compressed as `256`, the model cannot later
turn it into `32-32-32...` without retaining source/detail information or
recompressing the region. Query-conditioned adaptation therefore needs one of:

- retained fine-grained source states;
- a compact refinement representation;
- recomputation of the selected region;
- or a learned splitter/compressor that operates on an available coarse source.

This remains one dynamic compressor; it is a question of when the boundaries
are chosen and what source is available to the compressor.

The strongest eventual design is a hybrid: content-conditioned variable-rate
segmentation is persisted, while a query can request selective refinement of a
small number of segments. The first study can establish the variable-rate
compressor itself before adding that refinement path.

The cache then looks like this:

```text
each attention layer l
  |
  +-- raw recent region ---------- dynamic SWA window W[l,t]
  |
  +-- one CSA compressor ---------- segments of 1/2/4/8/16 tokens
  |       -> learned indexer -> sparse top-k summaries
  |
  +-- one HCA compressor ---------- segments of 32/64/128/256/512 tokens
          -> dense attention over variable-length summaries
```

The variable segment metadata—boundaries, lengths, padding, and any refinement
data—must be included in the memory and latency accounting.

## 7. Factorial test of content and query conditioning

We should test both meanings of “resolution needed.” This is not an
unnecessary expansion; it is the experiment that tells us whether the benefit
comes from recognizing information density in the memory itself, responding to
the current query, or combining both.

Use a 2 x 2 factorial design:

| Condition | Content-conditioned segmentation | Query-conditioned refinement | Scientific question |
|---|---:|---:|---|
| Fixed baseline | off | off | What does uniform segmentation provide? |
| Content-only | on | off | Can the compressor predict useful resolution from the memory region itself? |
| Query-only | off | on | Can the current query request high precision only where it needs it? |
| Content + query | on | on | Are the two mechanisms complementary? |

The **content-only** condition chooses variable segments while the memory is
formed. The **query-only** condition starts from a fixed source representation
and lets the current query request a variable segmentation or refinement of
selected regions. The source representation, recomputation, and refinement
costs must be counted; otherwise query-only would receive hidden extra memory.

The **content + query** condition persists a content-adaptive segmentation and
then lets the query selectively refine a small number of existing segments.
This is the most direct test of the phrase “high precision only when
necessary.” It remains one adaptive compressor: the query changes which
segments are split or recompressed, rather than selecting among four separate
compression layers.

The main hypotheses are:

- content-only should help on structural, repetitive, and information-density
  variation;
- query-only should help most on exact retrieval and distractor-heavy tasks;
- content + query should perform best on mixed tasks if the mechanisms are
  complementary;
- if the combined result does not exceed the individual results, the
  interaction itself is still informative.

Report the interaction effect rather than only ranking the four models. For a
quality metric, a simple difference-in-differences is:

```text
interaction = quality(content+query)
            - quality(content-only)
            - quality(query-only)
            + quality(fixed baseline)
```

Run the same calculation for cache bytes, attention work, and wall-clock
latency. A quality gain that disappears after router, refinement, or
recomputation costs are included should be reported as an overhead result,
not hidden.

The full study should compare at least:

| Condition | SWA | CSA | HCA | Purpose |
|---|---|---|---|---|
| Reference | fixed | fixed 4 | fixed 128 | faithful fixed-resolution baseline |
| Uniform-control | fixed | fixed 4 | fixed 128 | measures the cost of removing variable segmentation |
| DA-only | dynamic per layer x query | fixed 4 + dynamic blocks/top-k | fixed 128 | tests selective reading |
| DR-only | fixed | one dynamic variable-rate compressor | one dynamic variable-rate compressor | tests adaptive segmentation |
| Full | dynamic per layer x query | dynamic | dynamic | tests interaction |
| Oracle/upper bound | generous fixed budget | fine resolution | fine resolution | estimates available-quality ceiling |

The DA-only condition should not silently change the number of cached bytes.
The DR-only condition should not silently change which blocks are selected.
Otherwise the experiment cannot tell whether a gain came from attention,
resolution, or simply a larger budget.

## 8. Training objective and safeguards

A useful abstract objective is:

```text
loss = language_or_retrieval_loss
     + lambda_memory * normalized_cache_bytes
     + lambda_compute * normalized_attention_work
     + lambda_switch * router_switches
```

The last term discourages unstable resolution changes. A hard minimum recent
window is also sensible so that the model cannot discard all local syntax.

The router must be causal: at position `t`, it may use the current query and
the prefix, but never the answer token, future tokens, or a label derived from
future context. During evaluation, report both quality and the actual selected
resolution distribution.

For a first implementation, use a straight-through choice, Gumbel-softmax, or
a small soft mixture during warm-up, followed by a hard discrete choice for
measurement. The final claim must be based on the hard path that is actually
timed, not only on a soft gate.

## 9. Risks that need to become research questions

1. **Variable compression can destroy boundaries.** A query may need the
   beginning of a block, not merely its average. Overlap or boundary tokens
   may be required, and their byte cost must be counted.
2. **Segment length and top-k are confounded.** A finer CSA segmentation can make
   retrieval easier even when the number of selected blocks is unchanged.
   Keep one fixed while studying the other.
3. **The router may learn length or formatting shortcuts.** Use matched,
   shuffled, and adversarial controls so it cannot identify “important” spans
   only because they occur near the end or have a recognizable delimiter.
4. **A password-like exact fact is not a global summary problem.** Its correct
   protection mechanism is exact/local retention or CSA retrieval; relying on
   HCA alone would be an architectural mistake, not an interesting success.
5. **More adaptive choices may cost more than they save.** Include router
   FLOPs, indexer cost, metadata, padding, kernel inefficiency, and cache bytes
   in the efficiency measurement.
6. **Full DeepSeek-scale replication is unnecessary.** The science-fair
   contribution can be a controlled mechanism study on a small open model,
   provided the claim is explicitly about the adaptive policy under matched
   budgets rather than about reproducing DeepSeek-V4 training.

## 10. Current working hypothesis

Under matched model, data, context-length, quality, and hardware budgets, a
causal router that assigns:

- a local attention span based on immediate query needs,
- finer CSA resolution to regions with high specific-retrieval value, and
- finer HCA resolution to regions with high structural/contextual value

will improve the quality-efficiency frontier over fixed-resolution hybrid
attention. The strongest result would not merely be higher accuracy. It would
show that the router makes interpretable choices, preserves exact retrieval,
retains broad structure, and reduces measured cache/attention cost on the
same hardware budget.

## 11. What to lock before code

1. Choose one small causal open model and one faithful fixed hybrid baseline.
2. Define whether the first router acts per layer, head, block, or query.
3. Choose the exact local-window candidate set and the cache budget.
4. Define “specific retrieval value” using a causal indexer signal.
5. Define content-conditioned and query-conditioned segmentation as separate
   mechanisms with separate source-memory accounting.
6. Define “structural/contextual value” through a measurable auxiliary target
   or a cost-regularized end-to-end objective.
7. Select long-context tasks that separately test exact retrieval, multi-hop
   retrieval, instruction retention, and global coherence.
8. Pre-register the primary metric, compute/memory measurement method, seeds,
   ablations, and failure criteria before implementation.

The next pre-code decision is therefore not the model size. It is the scope of
the first causal experiment: dynamic SWA span only, or dynamic SWA span plus
CSA top-k/read-budget selection with compression held fixed.
