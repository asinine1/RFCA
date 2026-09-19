# RFCA quality benchmark before GPU scaling

**Date:** 2026-09-18

## Question

Does the diffusion canvas improve held-out commitment quality enough to justify
its additional parameters and denoising compute, compared with a causal-only
student?

## Controls

All runs used the local from-scratch byte-tokenizer student, `smoke` width,
batch size 32, 600 updates, two theorizer passes, the same learning rate and
seed, and MPS. The ordinary causal control used zero canvas layers. A second
causal control used six prefix layers so its parameter count and throughput
were comparable to RFCA.

The ordered task contains four keyed words whose continuation has one of 24
possible orders. The standard task uses the existing deterministic synthetic
text windows. These are controlled feasibility tests, not natural-language
benchmarks.

## Held-out results

| Task / model | Params | Loss | Accuracy | Approx. train steps/s |
| --- | ---: | ---: | ---: | ---: |
| Ordered RFCA, seed 2027 | 5,079,808 | 0.4556 | 0.7832 | 33.3 |
| Ordered causal-only, seed 2027 | 2,456,832 | 0.5381 | 0.7437 | 65.5 |
| Ordered RFCA, seed 2028 | 5,079,808 | 0.4571 | 0.7827 | 34.8 |
| Ordered causal-only, seed 2028 | 2,456,832 | 0.4908 | 0.7603 | 67.3 |
| Ordered matched causal-only | 6,653,184 | 0.4624 | 0.7720 | 27.6 |
| Standard RFCA | 5,075,712 | 0.3843 | 0.8569 | 37.6 |
| Standard causal-only | 2,452,736 | 0.2748 | 0.9111 | 76.2 |
| Standard matched causal-only | 6,649,088 | 0.1340 | 0.9536 | 32.3 |

The ordered RFCA advantage over the smaller causal control averaged about
3.1 accuracy points across the two seeds. Against the matched-capacity causal
control it was only 1.1 points. The standard task reversed the result: the
matched causal control beat RFCA by 9.7 accuracy points and had substantially
lower loss.

## Canvas and rolling checks

On the ordered RFCA checkpoint, removing the canvas reduced accuracy from
0.7832 to 0.0479, while pooling the canvas reduced it to 0.6851. Shuffling
canvas content while preserving destination slots reduced accuracy only to
0.7700. On the second seed, full/shuffled accuracy was 0.7827/0.7456. The
current RFCA student therefore uses the canvas, but its order sensitivity is
small and it appears to exploit a broadcast summary.

On the standard RFCA checkpoint, the eight-commit retained-canvas accuracy was
0.8750, 0.2969, 0.0781, 0.0859, 0.0547, 0.1016, 0.0625, and 0.1250. The
reset control was generally better after the first commit. The causal-only
control was also imperfect under this rolling recipe, but it remained better
at most later commits. The current training curriculum does not make retained
rolling canvas state reliable.

## Larger solidifier experiment

The initial RFCA model had only a shallow solidifier. To test the proposed
“dense solidifier versus smaller theorizer” split, the next comparison added
four extra causal blocks after the solidifier read the canvas. The RFCA model
therefore had a stronger commit head than theorizer, while the control used the
same four extra blocks with no canvas at all.

| Task / model | Params | Loss | Accuracy | Approx. train steps/s |
| --- | ---: | ---: | ---: | ---: |
| Standard RFCA, large solidifier | 7,960,576 | 0.1383 | 0.9551 | 28.0 |
| Standard dense control, large solidifier | 6,386,944 | 0.2107 | 0.9229 | 34.6 |
| Ordered RFCA, large solidifier | 7,964,672 | 0.4965 | 0.7583 | 23.7 |
| Ordered dense control, large solidifier | 6,391,040 | 0.4741 | 0.7695 | 27.0 |

On the standard task, the larger-solidifier RFCA model improved accuracy by
3.2 points and reduced loss by 0.0721 versus the dense control. This reverses
the earlier standard-task result and supports the hypothesis that the previous
RFCA solidifier was simply too weak to turn canvas information into a good
commitment. The cost was about 1.25x parameters and 19% lower training
throughput.

The ordered task did not show the same raw-score improvement: the dense control
was 1.1 points more accurate. However, the RFCA canvas was not ignored. On the
RFCA checkpoint, removing the canvas changed accuracy from 0.7583 to 0.6255
and pooling the canvas reduced it to 0.7100. Shuffling canvas slots barely
changed accuracy (0.7603), so this probe demonstrates canvas dependence but not
strong positional dependence. The standard probe similarly showed a large
canvas ablation effect (0.9551 full versus 0.9375 with no canvas), while
slot-shuffling preserved 0.9551 accuracy.

The larger-solidifier rolling test remains negative. On the standard task,
retained-canvas accuracy over commits 1–8 was
0.9766, 0.4922, 0.4297, 0.2656, 0.1016, 0.2031, 0.0469, and 0.1875;
the reset control was 0.9766, 0.5547, 0.5547, 0.5703, 0.4922, 0.8672,
0.5547, and 0.6641. The dense control produced identical retained and reset
predictions, as expected because it has no canvas state. A stronger one-shot
solidifier therefore does not by itself solve rolling-state drift.

## Teacher-forced rolling curriculum

The next run kept the larger-solidifier architecture but trained 50% of
standard-task batches with a teacher-forced retained canvas. For a randomly
chosen depth from one to four commits, training appended the known committed
token, shifted the hidden canvas one slot, refreshed only the new tail, and
optimized the next solidifier decision. This matches the rolling transition
used by inference while avoiding compounding token mistakes during training.

The clean one-shot score was lower than the clean-only run: loss/accuracy were
`0.2224/0.9312` versus `0.1383/0.9551`. Throughput also fell from 28.0 to
15.7 steps/s because many batches executed multiple retained commits. That is
the expected cost of training on the harder state distribution.

The parameter-matched causal control used three prefix blocks and four
solidifier blocks with no canvas (`7,436,032` parameters versus RFCA's
`7,960,576`). Its clean diagnostic loss/accuracy were `0.1862/0.9370`, so
causal capacity alone slightly outperformed RFCA on the one-shot metric. Its
rolling retained path, however, was exactly identical to its reset path because
there is no retained canvas to carry information.

The rolling result changed substantially:

| Commit | Curriculum retained | Curriculum reset | Clean-only retained | Clean-only reset | Matched causal |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9375 | 0.9375 | 0.9766 | 0.9766 | 0.9531 |
| 2 | 0.9141 | 0.9141 | 0.4922 | 0.5547 | 0.5391 |
| 3 | 0.9453 | 0.8984 | 0.4297 | 0.5547 | 0.5547 |
| 4 | 0.8750 | 0.8516 | 0.2656 | 0.5703 | 0.6406 |
| 5 | 0.9062 | 0.9375 | 0.1016 | 0.4922 | 0.4922 |
| 6 | 0.8516 | 0.9609 | 0.2031 | 0.8672 | 0.8203 |
| 7 | 0.8516 | 0.8672 | 0.0469 | 0.5547 | 0.5391 |
| 8 | 0.8984 | 0.9453 | 0.1875 | 0.6641 | 0.6641 |

The curriculum prevents the retained canvas from collapsing: accuracy stays
above 85% through all eight commits, whereas the clean-only model falls below
50% after the second commit. It does not make retention universally better
than reset—the reset path wins at commits 5, 6, and 8—but retained state is no
longer catastrophic. The matched causal model is stronger at commit 1 and
occasionally later, but it has no mechanism to preserve provisional future
information; its retained and reset scores are always identical. This is the
first local evidence that the solidifier can learn to operate on the imperfect
states produced by rolling, and that the benefit is specifically about retained
canvas state rather than only additional causal depth.

## Decision

The teacher-forced curriculum is a meaningful improvement for the actual
rolling failure mode. The larger solidifier plus imperfect-canvas training now
supports stable retained-state accuracy on this synthetic benchmark. It costs
one-shot quality and compute, and the result is still byte-level synthetic
evidence rather than a frozen DiffusionGemma or real-text result. Do not spend
GPU hours on a long RFCA run yet.

The next local experiments should establish one of the following before
scaling:

- repeat the curriculum with a compatible frozen DiffusionGemma theorizer and
  a held-out real-text corpus;
- prevent the theorizer from broadcasting a global summary to every slot;
- compare the larger-solidifier design on a held-out real-text corpus; or
- narrow the research claim to structured future-planning tasks where the
  canvas demonstrably helps.

## Frozen-theorizer benchmark readiness

The frozen solidifier harness now supports the same comparison with
`--canvas-mode full` and `--canvas-mode none`, plus held-out per-commit rolling
metrics. The no-canvas control adds one causal prefix layer so its trainable
capacity is close to the full-canvas solidifier.

The official Transformers checkpoint,
`google/diffusiongemma-26B-A4B-it`, has been downloaded locally in 11
safetensor shards. Its configuration exposes the expected
`DiffusionGemmaForBlockDiffusion` interface, 256 canvas slots, 2,816-dimensional
hidden states, and a 262,144-token vocabulary. The checkpoint is approximately
48 GiB, matching the machine's total RAM, so it has not been loaded locally.

The harness was run end-to-end for both paths against project-report prose with
the tiny public checkpoint. The full and causal controls had 34,025,985 and
34,151,808 trainable parameters respectively, but both achieved zero held-out
token accuracy. This is expected from the random/test-scale checkpoint and is
recorded only as interface evidence, not a quality result. The meaningful
comparison is ready for a CUDA host with enough memory for the official model.

Run artifacts are under `runs/quality-*`, including the checkpoints,
diagnostic JSON, and rolling reports.
