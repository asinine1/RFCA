# RFCA research direction and competitive benchmark

**Status:** active direction document, checked 2026-09-18. This document
records the literature and fair-project review that should guide the first
implementation. It is not a novelty certification, submitted research plan,
or statement of experimental results.

## 1. Executive decision

Keep the RFCA direction, but narrow the project around one falsifiable
mechanistic question:

> Under matched parameter and compute budgets, does a limited-pass,
> model-generated diffusion future canvas improve causal next-token decisions
> on tasks that require long-range dependencies?

The broad idea “combine diffusion and autoregression” is already occupied by
academic systems and prior student projects. The possible contribution is the
specific interface and evaluation protocol:

1. a provisional future canvas is maintained rather than fully committed;
2. a separate solidifier reads the full canvas and commits exactly one token;
3. the canvas is shifted, refreshed, or re-noised after commitment;
4. training uses imperfect model-generated canvases rather than clean future
   labels;
5. the contribution is tested against matched no-canvas, summary-canvas,
   block-diffusion, and compute controls.

No public official project reviewed here matches all five elements together.
That is evidence for further investigation, not proof of global novelty.

## 2. Project definition

At generation step `t`:

1. The committed prefix contains only permanently selected tokens.
2. A future canvas of horizon `H` contains provisional representations.
3. A bidirectional diffusion component makes a limited number of refinement
   passes.
4. A causal solidifier reads the committed prefix and every canvas position.
5. The solidifier commits one next token.
6. The canvas is rolled forward and partially or fully refreshed.

The working mathematical description is:

```text
C_t^(k+1) = D_theta(x_<t, C_t^(k), k)
p_phi(x_t | x_<t, C_t^(K)) = G_phi(x_<t, R(C_t^(K)))
C_(t+1)^(0) = U(C_t^(K), x_t, noise, position)
```

The important boundary is causal commitment. The solidifier may read the
model-generated canvas, but it may not read clean target suffix tokens in the
main training or inference path.

## 3. Literature boundary

### Closest academic systems

| Work | Overlap with RFCA | Difference and implication |
|---|---|---|
| [TiDAR: Think in Diffusion, Talk in Autoregression](https://arxiv.org/abs/2511.08923) | Uses diffusion-style drafting and autoregressive sampling in a unified hybrid language model. | Its central mechanism is block drafting/speculative-style acceptance. RFCA instead proposes a continuously rolling canvas whose full provisional state conditions a separate one-token solidifier. TiDAR is the most important direct novelty threat. |
| [Block Diffusion](https://arxiv.org/abs/2503.09573) | Generates autoregressively over blocks and uses discrete diffusion inside each block. | It commits a block before proceeding. RFCA commits one token while preserving and updating a longer provisional future. It should be a required baseline. |
| [DiffCoT](https://aclanthology.org/2026.findings-acl.1939/) | Uses a sliding-window diffusion-style process for retrospective correction while retaining token-level autoregressive generation. | It is designed around reasoning-step revision. RFCA proposes a general future canvas that influences each next commitment. The overlap means “revisable future structure” cannot be presented as unprecedented. |
| [DiffusionGemma mechanics](https://ai.google.dev/gemma/docs/diffusiongemma/explained) | Uses a causal prefix plus a bidirectional diffusion canvas, with a 256-token canvas in the documented model. | DiffusionGemma fully denoises a canvas block, appends it, and starts another block. It does not establish the RFCA full-canvas, one-token, rolling solidifier. It does establish that “causal prefix plus canvas” is already a real design pattern. |
| [Deferred Commitment Decoding](https://arxiv.org/abs/2601.02076) | Treats token commitment and deferral as a central problem using a confidence-aware sliding window. | It is primarily a decoding policy, whereas RFCA makes the canvas-to-solidifier interface a learned architectural object. It is still an important adjacent comparison. |

### Student project precedents

| Project | Official record | Relevance |
|---|---|---|
| [RSTOD: Novel Auxiliary Learning Techniques for Efficient and Controllable Task-Oriented Conversational Agents](https://abstracts.societyforscience.org/Home/FullAbstract?ProjectId=23319) | ISEF 2023, Robotics and Intelligent Machines finalist. | Explicitly compared autoregressive and diffusion language models for dialogue. RFCA must not claim to be the first student project to compare those families. |
| [A Single Usage Is All You Need](https://abstracts.societyforscience.org/Home/FullAbstract?ProjectId=23819) | ISEF 2023, Robotics and Intelligent Machines finalist. | Treated generation order and decoding schedule as the research problem. RFCA differs by preserving one-token causal commitment while conditioning on a revisable future. |
| [State Space Models Are All You Need](https://abstracts.societyforscience.org/Home/FullAbstract?ProjectId=25091) | ISEF 2024, Robotics and Intelligent Machines, ISEF Third Award, AAAI First Award, and NSA Third Place. | A strong architecture-research precedent: named mechanisms, long-range benchmarks, matched baselines, parameter-aware comparisons, and efficiency analysis. |
| [HAGU: AI-Generated Harmonically Rich Classical Guitar Pieces](https://abstracts.societyforscience.org/Home/FullAbstract?ProjectId=24874) | ISEF 2024, Technology Enhances the Arts, Second Award. | Used future n-gram prediction and n-stream self-attention in music generation, then proposed a task-specific playability metric. It shows the value of evaluating the claimed mechanism directly rather than relying only on generic loss. |
| [DOTBot](https://abstracts.societyforscience.org/Home/FullAbstract?ProjectId=28344) | ISEF 2026, Robotics and Intelligent Machines finalist. | Combines diffusion-based visual reconstruction, mutual-information alignment, and an autoregressive structured reasoning head. It is not a textual future canvas, but it is a warning against claiming that diffusion plus structured autoregressive commitment is itself new. |

## 4. Fair-project comparison

### DMRSEF / regional level

The [2024 DMRSEF awards list](https://clas.ucdenver.edu/denversciencefair/sites/default/files/attached-files/2024_awards_list_for_website.pdf)
shows several useful local comparisons.

- [Feel the Ball: Convert Ball Motion to Touch for Vision/Hearing-Impaired
  Sport Audiences](https://symposium.foragerone.com/2024-dmrsef/presentations/60522)
  won first place in Senior Computer Sciences & Mathematics and second place
  for Senior Best in Fair. Its pipeline—video, tracking, representation,
  haptic output—is substantially easier to explain than RFCA, which means the
  RFCA poster must make its state transitions visually obvious.
- [Predictive Football Analysis](https://www.jsr.org/hs/index.php/path/article/view/7372)
  won second place in Senior Computer Sciences & Mathematics and was first
  alternate for Senior Best in Fair. It combined next-play prediction with a
  Double Deep Q-Network decision loop. The lesson is to isolate the value of
  each algorithmic component rather than show only an end-to-end score.

The [2025 DMRSEF results](https://clas.ucdenver.edu/denversciencefair/node/572/attachment)
list Amy Zhang’s *AI Guides, Vision-Impaired Act* as first place in Senior
Computer Science & Mathematics and Eshaan Yendamuri’s *MotionMatics* as second
place. Amy’s multi-stage AI system is the closest regional conceptual analogy:
multiple model stages transform an uncertain intermediate representation into
an actionable output. RFCA is more fundamental and less application-driven,
but therefore needs a clearer explanation and stronger controlled evidence.

### CSEF / state level

The official [2025 CSEF results](https://csef.natsci.colostate.edu/csef-2025/)
list:

- Amy Zhang’s *AI Guides, Vision-Impaired Act* — first place in Senior
  Mathematics & Computer Sciences and first Best-of-CSEF;
- Devang Pandey’s *Enhancing Brain Tumor Classification with Synthetic Data
  via Gen Models and Hybrid Training* — second place;
- Tanush Shekhar and Jack Cerullo’s *Revolutionizing CryoEt* — honorable
  mention.

These projects show that “hybrid” or “AI-powered” is not enough. Stronger
projects define what each component contributes and compare against credible
alternatives.

The official [2024 CSEF results](https://csef.natsci.colostate.edu/csef-2024/)
also list Elton Cao’s *National Ground-Level NO2 Predictions via Satellite
Imagery Driven Hybrid Neural Networks* as a third-place Mathematics & Computer
Sciences project with additional awards. Its most useful lesson for RFCA is
experimental: test leakage, unseen conditions, and credible non-neural or
simpler baselines.

The [2026 CSEF results](https://csef.natsci.colostate.edu/csef-2026/)
include Amy Zhang’s *Just Look, Don’t Type* as a second-place project and
Eshaan Yendamuri’s *Confidence-Driven Dynamic Model Ensembles* as an honorable
mention in the same broad category. These are useful local benchmarks for
adaptive, staged, and model-selection systems.

### ISEF level

The [2026 ISEF awards list](https://www.societyforscience.org/press-release/regeneron-isef-2026-full-awards/)
shows the top of the relevant categories. Robotics and Intelligent Machines
first awards included an ecological monitoring robot and a physics-aware
hyperspectral-imaging system; Software Design first awards included M.A.N.T.I.S.
and ExpressBuddy. These are not direct RFCA analogs, but they illustrate the
top-level bar: a sharply defined method, a concrete scientific or engineering
payoff, and extensive validation.

At the student architecture/LLM level, the strongest comparisons are:

- *State Space Models Are All You Need*: named architecture changes, long-range
  sequence evaluation, and resource-aware baselines;
- *HAGU*: future-structure modeling plus a custom evaluation metric;
- *RSTOD*: direct AR-versus-diffusion language-model comparison;
- *DOTBot*: a broad diffusion-plus-autoregressive structured-reasoning system;
- *Keep Your Data Close*, *MERIT*, and *Shadow*: examples of advanced LLM work
  where the mechanism and evaluation question are very narrowly defined.

My inference from these records is that ISEF-level architecture projects are
not rewarded for complexity by itself. They need a contribution that can be
stated in one sentence, multiple meaningful controls, and enough evidence that
the student can defend every design choice.

## 5. Category recommendation

For CSEF, the natural category is **Mathematics & Computer Sciences**.

For ISEF, the strongest current fit is probably **Software Design →
Algorithms**. The [ISEF Software Design category](https://www.societyforscience.org/isef/categories-and-subcategories/software-design/)
explicitly includes algorithms involving data processing, automated reasoning,
and computing procedures.

Robotics and Intelligent Machines—especially Machine Learning or Cognitive
Systems—is also possible. The [ISEF Robotics and Intelligent Machines
category](https://www.societyforscience.org/isef/categories-and-subcategories/robotics-intelligent-machines/)
is more appropriate if the finished project is presented primarily as a
machine-intelligence system rather than as a generation algorithm.

The 2027 category structure and rules should be rechecked when officially
posted. Category choice should follow the completed study, not be fixed before
the experiment has clarified what was actually investigated.

## 6. Recommended implementation scope

The current design contains too many possible projects at once: a new
diffusion model, a 256-token canvas, rolling updates, on-policy training,
variable commitment, parameter sharing, and reasoning benchmarks. The first
implementation should be deliberately smaller.

### Core version

- `H = 16, 32, 64`; treat 256 as a later scaling test, not a requirement.
- (K = 2, 4, 8) denoising passes.
- One-token commitment only.
- Frozen diffusion model plus a learned solidifier or adapter.
- Synthetic structured tasks before natural-language modeling.

### Required controls

1. Standard causal autoregressive model with no canvas.
2. First-position-only canvas read.
3. Pooled canvas read.
4. Full-canvas solidifier.
5. Shuffled-canvas control.
6. Canvas-replaced-with-noise diagnostic.
7. Block-diffusion baseline if feasible.

The full-canvas model should only be credited if it beats the simpler
interfaces under the same data and compute accounting.

### Recommended tasks

Begin with:

- bracket and delimiter completion;
- delayed copying;
- symbolic transformations;
- long-range format constraints;
- small synthetic dependency or planning tasks.

Only add natural-language modeling after the interface works. Do not make
“reasoning” the headline claim unless reasoning-specific tests and failure
analysis are added.

### Metrics

Report both quality and resource use:

- exact match and structured validity;
- dependency-specific accuracy;
- negative log-likelihood or perplexity where appropriate;
- wall-clock time per committed token;
- denoiser calls per committed token;
- peak memory;
- total training compute;
- sensitivity to canvas corruption and position shuffling.

The primary result should be a quality-versus-cost comparison. A score
improvement caused only by spending substantially more compute is not a fair
RFCA result.

## 7. Go/no-go gates

- If the solidifier ignores the canvas, turn the project into an interface
  ablation study. That is still a valid result.
- If rolling refresh is too expensive, use a smaller horizon or frozen
  diffusion plus an adapter.
- If warm-starting creates stale-canvas drift, compare shift-only, tail
  re-noising, and full refresh.
- If clean target suffixes produce a large gain, treat it as a leakage warning,
  not as evidence of success.
- If no quality gain appears, analyze whether the failure comes from the
  canvas representation, rolling update, training curriculum, or compute cost.

A carefully controlled null result is scientifically stronger than an
uncontrolled claim that a complicated hybrid model “seemed better.”

## 8. Claim language

### Safe wording

> We developed and evaluated a rolling hybrid architecture in which a
> limited-pass diffusion canvas conditions an autoregressive next-token
> solidifier.

### Wording to avoid unless directly established

- “the first diffusion-autoregressive language model”;
- “the model visualizes its future”;
- “the model plans its answer”;
- “better reasoning” without task-specific evidence;
- “faster generation” without matched wall-clock measurements.

## 9. Recommended working title

**Testing Whether a Provisional Diffusion Future Improves Causal Next-Token
Generation Under Matched Compute**

This is less dramatic than a “reasoning through a future canvas” title, but it
is more defensible, easier for judges to evaluate, and better aligned with the
actual experiment.

## 10. Immediate next actions

1. Freeze the research question and claim language above.
2. Implement the no-canvas causal baseline first.
3. Implement a small diffusion canvas with `H=16` or `H=32`.
4. Add first-position-only, pooled, and full-canvas solidifiers.
5. Build one synthetic long-range task with a prespecified metric.
6. Log compute, denoiser calls, memory, and all random seeds from the first
   experiment.
7. Do not begin 256-token scaling, variable-length commitment, or reasoning
   benchmarks until the interface-ablation result is clear.

This file should be updated whenever a go/no-go gate changes the active scope.
