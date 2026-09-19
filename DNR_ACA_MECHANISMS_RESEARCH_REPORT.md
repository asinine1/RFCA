# Archived research report: Dynamic Memory ACA and DNR Mechanisms

> **Archived on 2026-09-16.** This report covers the superseded ACA/DNR
> context-retrieval project. It is preserved for literature history and is not
> the active project plan. See the [active project context](project%20context/README.md).

**Prepared:** 2026-09-16  
**Purpose:** internal technical learning report, not an ISEF Research Plan, paper, abstract, poster, novelty claim, or experimental result

## Executive conclusion

The project proposes a causal language-model memory system with two complementary adaptive mechanisms:

- **ACA (Adaptive Compressed Attention)** changes the resolution and selection of **older long-range memory**. ACA-Low retains relatively fine compressed summaries and retrieves a sparse query-relevant subset. ACA-High stores much coarser summaries and reads them densely to provide a cheap global view.
- **DNR (Dynamic N-Gram Reception)** changes how much **recent local context** each layer and token can read. The recommended first version is a dynamically selected causal sliding-window span, not token pooling and not a second compression hierarchy.

The cleanest implementation is not three full parallel attention systems in every layer. A closer and more tractable interpretation of the DeepSeek-V4 reference is:

```text
each attention layer
  |
  +-- DNR local branch: raw recent K/V states, dynamic causal span
  |
  +-- one older-memory branch, selected by the layer schedule:
        ACA-Low layer: adaptive fine summaries -> learned indexer -> top-k summaries
        ACA-High layer: adaptive coarse summaries -> dense read of all summaries
```

At the system level, ACA still has Low and High channels, but they are **interleaved across layers**, rather than necessarily computed together in every layer. This matches the official DeepSeek-V4 architecture more closely: V4 uses interleaved CSA and HCA layers, and both compressed layer types include a raw local sliding-window branch. The project then replaces fixed local SWA with DNR and fixed compression factors with ACA's learned variable-rate segmentation.

This is a promising research direction, but it is not implementation-ready and novelty is not established. The closest prior work already includes input-dependent dynamic attention spans, hierarchical compressed memory, adaptive KV merging, and adaptive compression by layer/head. The strongest potentially distinctive contribution is the **joint, causal, matched-budget study** of:

1. per-layer/per-position local span selection;
2. variable-length long-memory segmentation at two compression regimes;
3. sparse exact-item retrieval versus dense global summaries; and
4. their interaction under honest cache, computation, and wall-clock accounting.

The project should not claim that either ACA or DNR is new until a systematic literature review and experiments show exactly which component differs from prior work.

## 1. What the current project actually contains

The workspace is a pre-code planning notebook. It contains no model implementation, finalized tokenizer, dataset, trained checkpoint, benchmark result, latency measurement, memory measurement, or verified novelty result. The source documents repeatedly mark the architecture as an early proposal:

- [Dynamic Memory concept](project%20context/02_dynamic_memory_concept.md) defines the umbrella question and staged factorial study.
- [Dynamic Attention design space](project%20context/03_dynamic_attention_options.md) surveys alternatives but explicitly does not select a final architecture.
- [DeepSeek-inspired architecture](project%20context/04_deepseek_inspired_architecture.md) is the current ACA design.
- [DNR architecture](project%20context/05_dnr_dynamic_ngram_reception.md) is the current local-reception design.
- [2027 rules baseline](project%20context/01_rules_baseline_2027.md) sets the computational, non-human, synthetic-first science-fair boundary.

The working scientific question is:

> Under matched model, data, training-token, context-length, and compute budgets, does jointly adapting long-range memory resolution/allocation and local receptive-field size improve the long-context quality-efficiency frontier over either mechanism alone or a fixed baseline?

That is the right kind of question because a negative result remains meaningful. A router that adds overhead without improving the frontier is still a valid finding.

## 2. Required background: what attention memory is

For hidden state \(h_t\) at token position \(t\), an attention layer produces a query, key, and value:

\[
q_t = h_t W_Q, \qquad k_t = h_t W_K, \qquad v_t = h_t W_V.
\]

In causal self-attention, token \(t\) may read only positions \(j \leq t\):

\[
a_{t,j} = \operatorname{softmax}_j\left(\frac{q_t k_j^\top}{\sqrt{d_k}} + M_{t,j}\right),
\qquad
o_t = \sum_{j \leq t} a_{t,j}v_j,
\]

where \(M\) is a causal or sparse mask. During autoregressive decoding, past keys and values are cached so the model does not recompute the entire prefix for every new token. This **KV cache** grows linearly with context length, while dense prefill attention has quadratic token-pair work.

The project separates three ways to reduce this burden:

1. **Local restriction:** read only a recent window. This is the role DNR adapts.
2. **Sequence compression:** replace several old K/V entries with one summary. This is the central ACA operation.
3. **Sparse selection:** from many compressed entries, read only the most relevant ones. This is ACA-Low's indexer/top-k step.

These are not interchangeable. A zero-valued soft gate still leaves an entry computed and stored unless the implementation actually skips or removes it. A shorter theoretical attention mask does not guarantee faster execution unless the kernel avoids the masked work.

## 3. The real DeepSeek-V4 reference architecture

ACA is inspired by DeepSeek-V4, so the reference must be represented accurately.

The official [DeepSeek-V4 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/efc8551/DeepSeek_V4.pdf) defines two compressed attention layer types:

- **Compressed Sparse Attention (CSA):** compress every fixed \(m\) source tokens into one entry, score compressed entries with a learned Lightning Indexer, retain the query-specific top \(k\), and run core attention over those selected summaries plus a raw local window.
- **Heavily Compressed Attention (HCA):** compress every fixed \(m'\) tokens into one entry with \(m' \gg m\), attend densely to the compressed sequence, and also include a raw local window.

The official V4 configurations use fixed compression factors \(m=4\) for CSA and \(m'=128\) for HCA. The local window is 128 tokens. CSA and HCA are used in an **interleaved layer schedule**; they are not described as two long-range branches simultaneously evaluated inside every attention layer. The Hugging Face implementation documentation independently exposes each decoder block as one of sliding attention, CSA, or HCA and documents the fixed rates and top-k behavior in [DeepSeek-V4 architecture documentation](https://huggingface.co/docs/transformers/model_doc/deepseek_v4).

### 3.1 DeepSeek's fixed compressor

For HCA, the report first projects token states into a shared K/V representation \(C\) and a vector of compression logits \(Z\):

\[
C = HW^{KV}, \qquad Z = HW^Z.
\]

For each fixed block of \(m'\) tokens, it normalizes compression weights across positions and produces one summary:

\[
S_{i} = \operatorname{softmax}_{\text{position}}(Z_i + B),
\qquad
C_i^{\text{comp}} = \sum_{j \in \text{block }i} S_j \odot C_j.
\]

CSA uses a related **overlapping two-block compressor** with two projected streams, then compresses the sequence by a net factor of \(m\). The project notes currently abstract this detail away as \(C(K/V[a:b])\). That abstraction is acceptable for early planning, but the actual experiment must specify whether ACA inherits DeepSeek's weighted compressor, changes it, or uses a simpler pooling operator.

### 3.2 DeepSeek's sparse indexer

For CSA, query token \(t\) produces lightweight indexer queries. A learned score ranks preceding compressed blocks, and the top \(k\) summaries are sent to core attention. Abstractly:

\[
I_{t,s} = \sum_h w^I_{t,h}\,\operatorname{ReLU}\left(q^I_{t,h}\cdot K^{I,\text{comp}}_s\right),
\]

\[
\mathcal{R}_t = \operatorname{TopK}_s(I_{t,s}, k).
\]

The core attention reads compressed K/V entries indexed by \(\mathcal{R}_t\), concatenated with the raw recent-window K/V entries.

### 3.3 Why the local branch exists

A compressed block becomes causally readable only after all source tokens in that block exist. Therefore, a token cannot use a compressed summary of its own unfinished block. DeepSeek adds a raw sliding-window branch to CSA and HCA to preserve local dependencies and strict causality. This is exactly the branch DNR should adapt.

### 3.4 What ACA changes

DeepSeek uses **fixed** compression rates. ACA proposes to replace them with **learned variable segment lengths**. That is the central difference:

```text
DeepSeek CSA:  4 | 4 | 4 | 4 | ...
ACA-Low:      16 | 4 | 1 | 8 | 2 | ...

DeepSeek HCA: 128 | 128 | 128 | 128 | ...
ACA-High:     256 | 32 | 32 | 64 | 128 | ...
```

The project must say "DeepSeek-inspired," not imply that DeepSeek-V4 already uses adaptive variable-rate segments.

## 4. ACA: Adaptive Compressed Attention

ACA is best understood as an **adaptive compressed-memory and read subsystem**. Its name includes “attention,” but its proposed change spans two separate operations:

- how old K/V states are compressed into summaries; and
- which summaries a query reads.

### 4.1 ACA-Low

ACA-Low is the fine-resolution older-memory channel.

- Candidate segment lengths: \(s^L \in \{1,2,4,8,16\}\).
- Intended information: identifiers, isolated facts, retrieval keys, exact or query-specific details.
- Read behavior: a learned indexer scores summaries and selects a sparse top-k subset for the current query.
- Main benefit sought: retain exact-item retrieval at lower cache/read cost than keeping every old token.

An \(s=1\) segment means one source token produces one summary. It should not be called “exact retention” unless the compressor has an identity path or the experiment proves it preserves the necessary information. A learned projection can still alter or lose information even when the segment contains one token.

### 4.2 ACA-High

ACA-High is the coarse global-memory channel.

- Candidate segment lengths: \(s^H \in \{32,64,128,256,512\}\).
- Intended information: topic structure, instructions, organization, transitions, constraints, and broad long-range relationships.
- Read behavior: dense attention over all coarse summaries.
- Main benefit sought: provide a global map whose sequence length is short enough to read cheaply.

ACA-High should not be expected to preserve a password-like exact string by itself. The local or ACA-Low path must handle exact detail.

### 4.3 Variable-rate segmentation

For channel \(c \in \{L,H\}\), layer \(\ell\), and segment index \(j\), define channel-specific boundaries:

\[
a^c_{\ell,0}=0, \qquad
s^c_{\ell,j}=a^c_{\ell,j+1}-a^c_{\ell,j},
\]

with \(s^L_{\ell,j}\) and \(s^H_{\ell,j}\) drawn from their respective candidate sets. The project files currently use one generic \(a[\ell,j]\) for both channels, but independent segmentations require separate boundary variables.

A variable-length extension of the HCA-style weighted compressor could be:

\[
C^c_i=h_iW^c_C, \qquad Z^c_i=h_iW^c_Z,
\]

\[
\alpha^c_{j,i}
=\operatorname{softmax}_{i\in[a^c_{\ell,j},a^c_{\ell,j+1})}
\left(Z^c_i+B^c_{s^c_{\ell,j},r(i)}\right),
\]

\[
z^c_{\ell,j}=\sum_{i=a^c_{\ell,j}}^{a^c_{\ell,j+1}-1}
\alpha^c_{j,i}\odot C^c_i.
\]

Here \(B^c_{s,r}\) is a learnable within-segment positional bias for a segment of length \(s\), and \(r(i)\) is the token's relative position within that segment. This is a proposed formalization, not a mechanism already fixed by the project documents.

### 4.4 The causal boundary problem

“The router examines a segment and chooses its length” is ambiguous during streaming generation. A causal system cannot examine future tokens and then retroactively decide how long an earlier segment should have been.

There are three valid causal designs:

1. **Start-time length prediction:** at boundary \(a_j\), use the current prefix to predict the next length \(s_j\). Simple, but the router cannot see the future contents of that segment.
2. **Online stop/continue routing:** accumulate tokens and decide after each token whether to close the current segment. This lets the router use the segment observed so far, but the summary is unavailable until the segment closes.
3. **Fixed microchunks plus causal merging:** first create small fixed chunks, then merge adjacent completed chunks. This is easier to batch and compare, but makes the base chunk size part of the mechanism.

For a first study, online stop/continue or fixed microchunks plus merging is more faithful to “content-conditioned segmentation” than predicting a whole segment length before seeing its contents. Whichever design is selected must be identical during training-time simulation and incremental decoding.

### 4.5 Content-conditioned versus query-conditioned ACA

The project proposes a 2x2 design:

| Condition | Persistent content segmentation | Query-time refinement |
|---|---:|---:|
| Fixed | Off | Off |
| Content-only | On | Off |
| Query-only | Off | On |
| Content + query | On | On |

**Content-conditioned ACA** chooses or closes segments using only information already available when memory is formed. It can create a genuinely smaller persistent cache.

**Query-conditioned ACA** asks for finer resolution after a later query arrives. But lost detail cannot be recovered from a coarse summary. Query-time refinement therefore requires at least one of:

- retaining raw source hidden states;
- retaining smaller latent microchunks;
- recomputing selected source tokens;
- storing a hierarchy from which coarse segments can be split.

Those costs are not optional bookkeeping. If raw source states remain available, the memory has not actually been reduced by the amount implied by the coarse ACA summaries. The most defensible first ACA experiment is therefore **content-conditioned persistent segmentation only**. Query refinement should be a later mechanism with explicit source-memory and recomputation accounting.

### 4.6 Sparse and dense reads

In an ACA-Low layer, let \(M^L_\ell\) be the number of completed fine summaries. The indexer ranks them and selects \(k\):

\[
\mathcal{R}_{\ell,t}=\operatorname{TopK}_{j<M^L_\ell}
I_{\ell,t,j}.
\]

The long-range read cost of core attention is approximately proportional to \(T k\), but the indexer may still score many or all \(M^L_\ell\) summaries, so its own cost must be measured.

In an ACA-High layer, all \(M^H_\ell\) summaries are read:

\[
o^H_{\ell,t}=\operatorname{Attention}
\left(q_{\ell,t}, K^H_{\ell,<t}, V^H_{\ell,<t}\right).
\]

If average segment length is \(\bar{s}^H\), then \(M^H \approx T/\bar{s}^H\), giving a compressed dense-attention term proportional to \(T^2/\bar{s}^H\), rather than \(T^2\). Actual throughput still depends on packing, kernels, and batch shape.

### 4.7 Length and positional semantics

Variable segments introduce a problem absent from equal-size chunks: each summary represents a different amount of source text. Ordinary softmax treats every summary as one attention item. Five 32-token summaries receive five opportunities to attract attention, while one 160-token summary receives one. This creates an implicit preference for finely segmented regions.

The model therefore needs experiments around:

- a length embedding attached to each summary;
- a representative position, such as the segment endpoint or weighted centroid;
- a possible \(\log s_j\) mass correction to the attention logit;
- normalization that prevents long and short summaries from having systematically different magnitudes;
- boundary overlap or explicit boundary tokens.

No correction should be assumed correct. “No correction,” length embedding, and mass-corrected logits are useful ablations.

### 4.8 ACA training objective

The project proposes:

\[
\mathcal{L}_{ACA}
=\mathcal{L}_{task}
+\lambda_{mem}C_{cache}
+\lambda_{comp}C_{attention}
+\lambda_{switch}C_{switch}.
\]

A rigorous implementation must define every term in measurable units:

- \(C_{cache}\): actual bytes for summaries, unfinished-token buffers, indexer keys, boundaries, lengths, position data, raw refinement states, and padding;
- \(C_{attention}\): executed or analytically counted query-key interactions, separated into local, indexer, sparse core, and dense global work;
- \(C_{switch}\): number or rate of adjacent segment-length changes, if instability is a demonstrated issue;
- router cost: parameters, FLOPs, latency, activations, and any load-balancing or entropy regularization.

Discrete lengths can be trained with a straight-through estimator, Gumbel-softmax, or a soft warm-up followed by hard routing. Final efficiency claims must use the hard path that is actually executed and timed.

## 5. DNR: Dynamic N-Gram Reception

DNR controls recent local context. At layer \(\ell\) and token position \(t\), a router selects:

\[
n_{\ell,t}\in\{1,2,4,8,16\}.
\]

The recommended first operator is dynamic local attention. Define the local mask:

\[
M^{DNR}_{\ell,t,j}=
\begin{cases}
0,&\max(0,t-n_{\ell,t}+1)\le j\le t,\\
-\infty,&\text{otherwise.}
\end{cases}
\]

Then the local output is standard causal attention under that row-specific span:

\[
o^{local}_{\ell,t}
=\operatorname{Attention}
\left(q_{\ell,t},K_{\ell},V_{\ell};M^{DNR}_{\ell,t}\right).
\]

This convention makes \(n\) the total number of visible token states **including the current position**. Therefore \(n=1\) is self-only and includes no preceding neighbor. If the intended meaning is “\(n\) previous tokens plus the current token,” the mask and all compute accounting must use \(n+1\). The current notes mix “previous \(n\) states” with a slice that includes the current position; this must be resolved before coding.

“N-gram” here means a contiguous group of model tokens, not necessarily words. With a BPE or unigram tokenizer, one word may span multiple tokens.

### 5.1 Recommended DNR operator

The project considers three operators:

1. **Adaptive local-attention span:** retain individual token K/V states and choose how far back the current query can read.
2. **Adaptive n-gram pooling:** combine the selected local span into one new representation.
3. **Adaptive local grouping:** partition recent tokens into variable groups and emit one representation per group.

Only the first is cleanly distinct from ACA. Pooling and grouping turn DNR into another compression system and confound the joint study. Operator 1 should be the first and primary DNR definition.

### 5.2 DNR router

A minimal hard router is:

\[
r_{\ell,t}=W^2_\ell\,\phi(W^1_\ell x_{\ell,t}+b^1_\ell)+b^2_\ell,
\]

\[
p_{\ell,t}=\operatorname{softmax}(r_{\ell,t}),
\qquad
n_{\ell,t}=\mathcal{N}[\arg\max p_{\ell,t}],
\]

where \(\mathcal{N}=[1,2,4,8,16]\). During training, the hard decision can use a straight-through or Gumbel estimator.

The router input \(x_{\ell,t}\) must make the experimental factors identifiable:

- **Content-only:** a causal summary of the recent prefix that is defined separately from the current attention query.
- **Query-only:** the current layer query state.
- **Content + query:** both inputs concatenated or combined through explicitly separate projections.

If both conditions use the same hidden state \(h_{\ell,t}\), the planned content/query factorial is only a label change, not two experimentally distinct signals. The architecture must specify what information each branch receives.

### 5.3 DNR training objective

The proposed objective is:

\[
\mathcal{L}_{DNR}
=\mathcal{L}_{task}
+\lambda_{local}\frac{1}{LT}\sum_{\ell,t}n_{\ell,t}
+\lambda_{switch}\sum_{\ell,t}\mathbf{1}
[n_{\ell,t}\ne n_{\ell,t-1}].
\]

The average-span term is a proxy for local attention work. A memory term should be used cautiously: during streaming decode, the model must usually retain at least the maximum recent window so a future token can choose the largest \(n\). DNR can reduce **read work** without reducing local KV storage below the maximum permitted span. During full-sequence training, a dense masked implementation may not even reduce executed work.

### 5.4 DNR efficiency reality

The ideal local token-pair count is:

\[
C_{DNR}^{ideal}=\sum_{\ell=1}^{L}\sum_{t=1}^{T}n_{\ell,t}.
\]

For fixed span \(w\), this is approximately \(LTw\). DNR helps theoretically when average \(n\) is less than the matched fixed span. But ordinary dense attention kernels may compute a full \(T\times T\) or fixed-window block and only mask unused entries. Real speedups require variable-length, block-sparse, ragged, or grouped-by-span execution.

The implementation should report three separate numbers:

1. ideal selected token-pair work;
2. actual kernel operations or profiler-estimated work;
3. measured prefill and decode latency.

### 5.5 The closest prior work and DNR novelty risk

[Adaptive Attention Span in Transformers](https://arxiv.org/abs/1905.07799) already learns per-head spans, includes a dynamic input-conditioned span \(z_t=S\sigma(v^Tx_t+b)\), penalizes span length, and reports reduced FLOPs in character-level language modeling. Its dynamic variant changes span with the input at each time step.

Therefore, “a model dynamically chooses a local attention span” is not itself a novel claim. DNR differs only if the study demonstrates a specific new mechanism or scientific question, such as:

- discrete per-layer/per-position span routing integrated into a compressed long-memory architecture;
- an experimentally isolated interaction between local span adaptivity and variable-rate long-memory compression;
- a new causal budget-matching method or hardware-realized routing kernel;
- a result showing when local adaptivity and long-memory adaptivity repair different error classes.

Changing the name from adaptive span to “Dynamic N-Gram Reception” does not establish novelty.

## 6. How ACA and DNR should work together

### 6.1 Recommended layer-level dataflow

For each layer \(\ell\) and position \(t\):

1. Project the current hidden state to a query.
2. DNR selects the recent raw-token span \(n_{\ell,t}\).
3. The layer's long-memory type supplies older summaries:
   - ACA-Low layer: route variable fine segments, index all eligible summaries, and retrieve top-k;
   - ACA-High layer: route variable coarse segments and expose all eligible summaries.
4. Concatenate or jointly mask local raw K/V entries and eligible older summaries.
5. Apply attention, output projection, residual connection, and feed-forward block.

```text
h[l,t]
  |
  +--> DNR router --> n[l,t] --> recent raw K/V ------------------+
  |                                                               |
  +--> query projection ------------------------------------------+--> core attention
                                                                  |
older completed memory --> ACA-Low variable summaries --> top-k --+
                    OR --> ACA-High variable summaries --> dense --+
```

This design makes DNR the adaptive replacement for the fixed local window that DeepSeek includes in compressed layers. It avoids adding a fourth attention path.

### 6.2 Eligibility and boundary rule

A compressed summary is eligible only after its source segment has fully closed. If segment \(j\) covers positions \([a_j,a_{j+1})\), query \(t\) may read it only when:

\[
a_{j+1}-1 \le t.
\]

The unfinished segment remains in a raw buffer and is covered by DNR if it is recent enough. Buffer bytes must be included in cache measurements.

### 6.3 Avoiding double-counting

Recent raw tokens may overlap with a just-completed compressed segment. A clear policy is required:

- retain both and allow redundancy;
- mask the compressed copy while its raw tokens remain in the DNR window;
- or evict raw tokens as they age beyond the maximum DNR window.

Masking the compressed copy while raw detail exists may reduce duplicated attention, but it changes the reference behavior and should be ablated.

### 6.4 Full joint objective

A joint model could optimize:

\[
\mathcal{L}
=\mathcal{L}_{task}
+\lambda_B\frac{B_{actual}}{B_{budget}}
+\lambda_A\frac{A_{actual}}{A_{budget}}
+\lambda_R C_{router}
+\lambda_S C_{switch}.
\]

The safest scientific comparison uses **hard external budgets** first, then penalty-based budgets as a secondary study. Penalty coefficients can otherwise let one model buy better quality by simply spending more memory or compute.

## 7. Complexity and resource accounting

Let:

- \(T\): sequence length;
- \(L\): number of layers;
- \(L_L,L_H\): counts of ACA-Low and ACA-High layers;
- \(M_L,M_H\): numbers of completed summaries in those layers;
- \(k\): ACA-Low retrieval top-k;
- \(\bar n\): average DNR span.

An idealized accounting is:

| Component | Approximate token-pair/read work | Stored sequence entries |
|---|---:|---:|
| Full attention | \(O(LT^2)\) | \(O(LT)\) |
| DNR local branch | \(O(LT\bar n)\) | usually \(O(Ln_{max})\) for streaming local cache |
| ACA-Low core read | \(O(L_LT k)\) | \(O(L_LM_L)\) plus indexer cache |
| ACA-Low indexer | potentially \(O(L_LT M_L)\) before optimized search | included above plus index keys |
| ACA-High dense read | \(O(L_HTM_H)\) | \(O(L_HM_H)\) |

This table is deliberately approximate. The real report must also include:

- K/V and router dimensionalities;
- compression projection and pooling work;
- boundary metadata and length embeddings;
- unfinished-token buffers;
- alignment and padding;
- indexer projections and scoring;
- sparse gather/scatter and repacking;
- accelerator memory allocator behavior;
- batch-size and sequence-length dependence;
- training activations versus decode-time KV cache;
- parameter count and active-parameter count.

A quality gain at higher actual cost is not an efficiency improvement. The primary result should be a Pareto frontier over quality versus cache bytes, executed work, and latency.

## 8. Experimental design required to identify the mechanisms

### 8.1 Build order

1. **Fixed reference:** small causal Transformer with a pinned layer schedule, fixed local window, fixed fine compression, fixed coarse compression, and fixed top-k.
2. **DNR-only:** replace the fixed local window with adaptive span; keep all compression fixed.
3. **ACA content-only:** keep local span and retrieval policy fixed; replace fixed compression with causal variable segmentation.
4. **ACA query refinement:** add only after source-memory and recomputation costs are fully specified.
5. **Joint ACA + DNR:** run the matched four-condition factorial and measure the interaction.

This order is more diagnostic than implementing the full system at once.

### 8.2 Core factorial

| Condition | Long-memory compression | Local span |
|---|---|---|
| Fixed baseline | fixed | fixed |
| ACA-only | adaptive | fixed |
| DNR-only | fixed | adaptive |
| ACA + DNR | adaptive | adaptive |

For quality metric \(Q\), measure the interaction:

\[
\Delta_{interaction}
=Q_{ACA+DNR}-Q_{ACA}-Q_{DNR}+Q_{fixed}.
\]

Run the same difference-in-differences for cache bytes, ideal work, executed work, and latency. A positive quality interaction paired with a worse cost interaction is not automatically a win.

### 8.3 Necessary controls

- full causal attention at the same model/training budget;
- fixed local spans at DNR's matched mean and maximum span;
- fixed fine/coarse compression at ACA's matched average summary count;
- random router with the same routing distribution;
- position-only router to detect shortcut value;
- uniform segmentation with the same cache bytes;
- oracle/future-aware routing only as a labeled upper bound;
- hard routed execution versus soft-mask proxy;
- identical tokenizer, initialization policy, optimizer, learning-rate schedule, effective batch, training tokens, context-length curriculum, and evaluation protocol;
- multiple seeds and all failed runs.

### 8.4 Tasks that separate the paths

Synthetic data should deliberately isolate mechanisms:

| Task | Expected dependency | Main path tested |
|---|---|---|
| nearby syntax/bracket closure | short contiguous local span | DNR |
| local phrase reconstruction with distractors | variable local span | DNR |
| old isolated key-value pair | precise old detail | ACA-Low |
| multiple old keys with similar distractors | sparse exact retrieval | ACA-Low/indexer |
| document-wide instruction retention | broad global constraint | ACA-High |
| topic/section hierarchy | coarse structure | ACA-High |
| local clue that selects an old fact | local + sparse long-range | DNR x ACA-Low |
| global instruction governing a local pattern | local + global structure | DNR x ACA-High |

After deterministic synthetic tasks, use only a verified public-domain or explicitly licensed language corpus, recording the exact version, license, retrieval date, preprocessing, and hash.

### 8.5 Measurements

Quality:

- validation loss and perplexity;
- exact-match retrieval accuracy by context length and depth;
- distractor error rate;
- local dependency accuracy;
- instruction-retention and global-coherence measures;
- uncertainty or calibration where applicable.

Efficiency:

- actual cache bytes, not just summary count;
- peak accelerator memory;
- theoretical and executed attention work;
- training throughput and step time;
- prefill latency;
- single-token and batched decode latency;
- router/indexer/compressor overhead;
- energy only if measured with a defensible method.

Router behavior:

- span/segment-length distribution per layer and position;
- entropy and collapse rate;
- transition/switch frequency;
- correlation with boundaries, repetition, retrieval depth, and task type;
- selected-block recall for known relevant segments;
- error overlap between ACA-only and DNR-only.

## 9. Prior-work landscape and what it means for novelty

| Prior work | Mechanism | Overlap with this project | Remaining distinction to test |
|---|---|---|---|
| [Adaptive Attention Span](https://arxiv.org/abs/1905.07799) | learned per-head span, including input-dependent dynamic span | very close to DNR Operator A | discrete per-layer/position routing and interaction with ACA may differ |
| [Compressive Transformer](https://arxiv.org/abs/1911.05507) | short fine memory plus compressed older memory | strong conceptual overlap with ACA's multiresolution memory | ACA uses two compression regimes and sparse/dense reads with adaptive segment lengths |
| [DeepSeek-V4 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/efc8551/DeepSeek_V4.pdf) | fixed CSA/HCA compression, sparse indexer, dense coarse attention, raw local branch | direct reference architecture | ACA changes fixed rates into causal variable-rate segments; DNR adapts the local branch |
| [Dynamic Context Pruning](https://arxiv.org/abs/2305.15805) | learned removal of uninformative causal context | overlaps adaptive memory allocation | pruning differs from segment compression and local contiguous span choice |
| [KVMerger](https://arxiv.org/abs/2407.08454) | adaptive similarity-based KV merging | overlaps adaptive compressed cache | proposed ACA is trained variable contiguous segmentation with two read regimes |
| [UNComp](https://arxiv.org/abs/2410.03090) | uncertainty-aware adaptive compression across layers/heads | directly weakens any broad “adaptive compression is new” claim | ACA's segment-level causal routing and joint DNR interaction may remain distinct |
| [H2O](https://arxiv.org/abs/2306.14048) and [Scissorhands](https://arxiv.org/abs/2305.17118) | cache eviction based on importance/history | overlap with “what deserves memory” | eviction is different from learned variable-rate summarization |
| [DuoAttention](https://arxiv.org/abs/2410.10819) | retrieval heads versus streaming heads | overlaps structural allocation by head | ACA/DNR are input-adaptive segment/span policies rather than a mostly fixed head classification |

The honest novelty position today is:

- **Not established:** adaptive attention spans, compressed old memory, sparse long-context retrieval, and adaptive KV compression all predate this project.
- **Potentially research-worthy:** the exact combination and factorial separation of per-position local span, two-regime variable contiguous compression, and sparse-versus-dense long-memory access under matched budgets.
- **What must be demonstrated:** an identifiable mechanism, not just a new acronym; a difference from the closest methods; and a reproducible quality-efficiency result.

## 10. Critical unresolved decisions

These must be locked before implementation begins:

1. **Layer topology:** interleaved ACA-Low/ACA-High layers with DNR local branches, or both ACA branches in every layer? The interleaved version is recommended.
2. **Baseline schedule:** exact number/order of local, Low, and High layers in the small model.
3. **Compressor:** DeepSeek-style learned weighted pooling, mean pooling, local attention pooling, or another operator.
4. **ACA-Low overlap:** inherit DeepSeek's overlapping two-block compressor or use simpler non-overlapping variable segments.
5. **Boundary policy:** start-time length prediction, online stop/continue, or microchunk merging.
6. **Summary position:** endpoint, midpoint, weighted centroid, or learned position.
7. **Length correction:** none, length embedding, log-length attention correction, or a combination.
8. **DNR convention:** span includes current token or counts only previous tokens.
9. **DNR granularity:** layer x position, head x position, or query block. Layer x position is the best first choice.
10. **Router inputs:** operationally distinct content and query signals.
11. **Budget:** primary hard constraints in bytes and selected interactions, plus secondary latency measurement.
12. **Query refinement:** postpone or define the retained source and recomputation mechanism completely.
13. **Degeneracy criteria:** thresholds for always-maximum span, fixed-like segment distributions, low entropy, or failure to beat matched fixed policies.
14. **Model and data:** one small causal model, tokenizer, synthetic generator, licensed corpus, context lengths, and seed count.

## 11. Specific corrections to the current notes

The source documents contain several inconsistencies that should be repaired before they guide code:

1. The [concept note](project%20context/02_dynamic_memory_concept.md) defines Dynamic Reception at lines 146-151 as variable segmentation of CSA/HCA memory. The newer [DNR note](project%20context/05_dnr_dynamic_ngram_reception.md) defines it as adaptive local attention. Use the newer DNR definition for Part 2, and describe ACA segmentation as ACA's compression-resolution component.
2. The ACA diagrams can be read as three parallel paths in every layer. Official DeepSeek-V4 interleaves CSA/HCA layer types while adding a local branch to compressed layers. The report should explicitly choose whether the student model follows that schedule or intentionally departs from it.
3. The ACA equations need channel-specific boundaries and a real compressor definition.
4. The DNR mask needs an unambiguous inclusive/exclusive convention.
5. The ACA experiment table contains “Reference” and “Uniform-control” rows with the same stated settings. One should be redefined or removed.
6. The DNR maximum of 16 is not directly comparable with DeepSeek's 128-token local window. The small-model baseline must select a scale-appropriate fixed window and include matched mean/max controls.
7. “Content-conditioned” and “query-conditioned” are not experimentally distinct until their router inputs and timing differ.
8. Neither a soft gate nor a dense masked implementation may be reported as actual compression or speedup.

## 12. A concise implementation specification to aim toward

The following is a coherent first target, subject to the unresolved choices above:

```text
Model:
  small decoder-only Transformer
  fixed interleaved schedule of Low and High compressed-attention layers

Every compressed-attention layer:
  DNR local branch:
    layer-position router chooses n in {1,2,4,8,16}
    hard causal local attention over raw recent K/V

Low layer older-memory branch:
  causal content segmenter chooses s in {1,2,4,8,16}
  learned weighted compressor emits one summary per completed segment
  learned indexer scores eligible summaries
  hard top-k summaries join local K/V in core attention

High layer older-memory branch:
  causal content segmenter chooses s in {32,64,128,256,512}
  learned weighted compressor emits one summary per completed segment
  every eligible summary joins local K/V in core attention

Training:
  task loss under fixed external cache/read budgets
  soft warm-up only if needed
  hard routing for final training stage and all timed evaluation

Primary comparison:
  fixed / ACA-only / DNR-only / ACA+DNR
  identical model, data, tokens, context, optimizer, seeds, and hardware
```

This is not the only valid implementation, but it is the first version that makes the project's mechanisms distinct, causal, testable, and reasonably faithful to the chosen reference.

## 13. Science-fair boundary

The project should remain a non-human computational study: deterministic synthetic sequences first, then only verified public-domain or explicitly licensed text. No private chats, identifiable records, participant testing, surveys, or public user-study evaluation should enter the initial work.

The current [rules baseline](project%20context/01_rules_baseline_2027.md) says all-project forms, Adult Sponsor review, research plan, support disclosure, and final fair review still apply. It also records that AI assistance must be logged and disclosed and that the student must independently produce restricted submission materials and interpretation under the current rules. This report is therefore an internal learning aid; it should not be pasted into an ISEF Research Plan, abstract, paper, poster, conclusions, or bibliography.

## 14. Bottom line

ACA and DNR make sense as two different adaptive controls:

- ACA decides **how old memory is summarized and which fine summaries are retrieved**.
- DNR decides **how much raw recent context the current token reads**.

Their combination becomes scientifically interesting only if the project shows that they solve different failures and jointly improve a measured frontier. The most serious risks are router collapse, hidden source-memory costs, non-causal segmentation, summary-length bias, indexer overhead, and dynamic masks that remain dense in practice.

The next correct step is not large-scale training. It is to lock the 14 unresolved decisions, formalize the small fixed baseline, implement one mechanism at a time, and pre-register matched resource budgets and failure criteria.
