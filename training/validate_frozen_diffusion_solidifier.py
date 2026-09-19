#!/usr/bin/env python3
"""Run local no-GPU integration gates for the frozen-theorizer student path.

This is intentionally stricter than a forward smoke test.  It checks that the
frozen DiffusionGemma model remains unchanged, the causal solidifier receives
nonzero gradients, the canvas is shifted rather than appended, multiple
teacher-forced rolling commits remain executable, and the student output
actually responds to canvas changes.  Passing this file proves plumbing and
trainability; it does not prove that a tiny/random checkpoint carries useful
language information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from frozen_diffusion_solidifier import (
    DEFAULT_MODEL_ID,
    SOLIDIFIER_PRESETS,
    FrozenDiffusionGemmaTheorizer,
    TrainableAutoregressiveSolidifier,
    append_prefix_token,
    choose_device,
    choose_dtype,
    limit_prefix_length,
    prefix_inputs_from_ids,
)


def max_abs_difference(before: torch.Tensor, after: torch.Tensor) -> float:
    return float((before.detach().float() - after.detach().float()).abs().max().item())


def parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or infinity")


def run_validation(args) -> dict:
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    theorizer = FrozenDiffusionGemmaTheorizer(
        args.model_id, device, dtype, canvas_length=args.canvas_length
    )
    if args.solidifier_preset not in SOLIDIFIER_PRESETS:
        raise ValueError(f"unknown solidifier preset: {args.solidifier_preset}")

    prompt = "The frozen theorizer sketches a provisional future."
    prefix_inputs = theorizer.tokenize([prompt])
    prefix_ids = limit_prefix_length(
        prefix_inputs["input_ids"].clone(), args.max_prefix_length
    )
    prefix_inputs = prefix_inputs_from_ids(
        prefix_ids, theorizer.tokenizer.pad_token_id
    )
    state = theorizer.initialize(batch_size=1)
    frozen_before = parameter_snapshot(theorizer.model)

    state, canvas_hidden, canvas_logits, prefix_hidden = theorizer.refine(
        prefix_inputs,
        state,
        args.diffusion_steps,
        args.entropy_bound,
        args.denoiser_sampling,
    )
    assert_finite("canvas hidden", canvas_hidden)
    assert_finite("canvas logits", canvas_logits)
    assert_finite("prefix hidden", prefix_hidden)
    expected_canvas_shape = (1, theorizer.canvas_length, theorizer.hidden_size)
    if tuple(canvas_hidden.shape) != expected_canvas_shape:
        raise AssertionError(
            f"unexpected canvas hidden shape {tuple(canvas_hidden.shape)} != {expected_canvas_shape}"
        )

    rolled = theorizer.roll(state)
    if not torch.equal(rolled.canvas_ids[:, :-1], state.canvas_ids[:, 1:]):
        raise AssertionError("roll did not shift every canvas slot left by one")
    if state.self_conditioning_logits is not None:
        if not torch.equal(
            rolled.self_conditioning_logits[:, :-1, :],
            state.self_conditioning_logits[:, 1:, :],
        ):
            raise AssertionError("roll did not shift self-conditioning logits")

    solidifier = TrainableAutoregressiveSolidifier(
        theorizer, args.solidifier_preset, args.max_prefix_length
    ).to(device)
    solidifier.train()
    solidifier_before = parameter_snapshot(solidifier)
    used_token_ids = set(prefix_ids[0].tolist())
    excluded_token_ids = {
        theorizer.tokenizer.pad_token_id,
        theorizer.tokenizer.bos_token_id,
        theorizer.tokenizer.eos_token_id,
    }
    target_id = next(
        token_id
        for token_id in range(theorizer.vocab_size)
        if token_id not in used_token_ids and token_id not in excluded_token_ids
    )
    target = torch.tensor([target_id], device=device, dtype=torch.long)

    full_logits = solidifier(prefix_ids, prefix_inputs["attention_mask"], prefix_hidden, canvas_hidden)
    zero_canvas_logits = solidifier(
        prefix_ids,
        prefix_inputs["attention_mask"],
        prefix_hidden,
        torch.zeros_like(canvas_hidden),
    )
    shuffled_canvas_logits = solidifier(
        prefix_ids,
        prefix_inputs["attention_mask"],
        prefix_hidden,
        canvas_hidden.flip(1),
    )
    assert_finite("solidifier logits", full_logits)
    canvas_zero_delta = max_abs_difference(full_logits, zero_canvas_logits)
    canvas_shuffle_delta = max_abs_difference(full_logits, shuffled_canvas_logits)
    if canvas_zero_delta <= args.minimum_canvas_delta:
        raise AssertionError(
            f"solidifier is insensitive to canvas removal (max delta={canvas_zero_delta:.3e})"
        )
    if canvas_shuffle_delta <= args.minimum_canvas_delta:
        raise AssertionError(
            f"solidifier is insensitive to canvas order (max delta={canvas_shuffle_delta:.3e})"
        )

    optimizer = torch.optim.AdamW(solidifier.parameters(), lr=args.learning_rate)
    loss = F.cross_entropy(full_logits, target)
    loss.backward()
    gradient_norm = 0.0
    trainable_gradient_tensors = 0
    for parameter in solidifier.parameters():
        if parameter.grad is not None:
            trainable_gradient_tensors += 1
            gradient_norm += float(parameter.grad.detach().float().norm().item())
    if trainable_gradient_tensors == 0 or gradient_norm <= 0.0:
        raise AssertionError("solidifier received no usable gradient")
    frozen_gradient_tensors = sum(
        parameter.grad is not None for parameter in theorizer.model.parameters()
    )
    if frozen_gradient_tensors:
        raise AssertionError("frozen theorizer received gradients")
    optimizer.step()

    student_delta = max(
        max_abs_difference(solidifier_before[name], parameter)
        for name, parameter in solidifier.named_parameters()
    )
    frozen_after = parameter_snapshot(theorizer.model)
    frozen_delta = max(
        max_abs_difference(frozen_before[name], frozen_after[name])
        for name in frozen_before
    )
    if student_delta <= 0.0:
        raise AssertionError("optimizer step did not change solidifier parameters")
    if frozen_delta != 0.0:
        raise AssertionError(f"frozen theorizer changed by {frozen_delta:.3e}")

    # A short fixed-feature overfit check proves that the student can learn a
    # commitment from the frozen interface without re-training the theorizer.
    overfit_losses = []
    for _ in range(args.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        repeated_logits = solidifier(
            prefix_ids, prefix_inputs["attention_mask"], prefix_hidden, canvas_hidden
        )
        repeated_loss = F.cross_entropy(repeated_logits, target)
        repeated_loss.backward()
        optimizer.step()
        overfit_losses.append(float(repeated_loss.detach().item()))
    if overfit_losses[-1] >= overfit_losses[0]:
        raise AssertionError(
            f"solidifier did not lower a fixed-feature loss: {overfit_losses[0]:.4f} -> {overfit_losses[-1]:.4f}"
        )

    # Run the actual retained-canvas loop for several commits, including the
    # causal prefix update used at inference time.
    rolling_state = theorizer.initialize(batch_size=1)
    rolling_prefix = prefix_ids.clone()
    rolling_records = []
    for commit in range(args.commits):
        rolling_inputs = prefix_inputs_from_ids(
            rolling_prefix, theorizer.tokenizer.pad_token_id
        )
        rolling_state, rolling_canvas, _, rolling_prefix_hidden = theorizer.refine(
            rolling_inputs,
            rolling_state,
            args.diffusion_steps,
            args.entropy_bound,
            args.denoiser_sampling,
        )
        rolling_logits = solidifier(
            rolling_prefix,
            rolling_inputs["attention_mask"],
            rolling_prefix_hidden,
            rolling_canvas,
        )
        assert_finite(f"rolling logits commit {commit + 1}", rolling_logits)
        committed = rolling_logits.argmax(dim=-1, keepdim=True)
        rolling_prefix = append_prefix_token(
            rolling_prefix, committed, args.max_prefix_length
        )
        rolling_state = theorizer.roll(rolling_state)
        rolling_records.append(
            {
                "commit": commit + 1,
                "prefix_length": int(rolling_prefix.shape[1]),
                "canvas_length": int(rolling_canvas.shape[1]),
            }
        )

    result = {
        "status": "passed",
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype),
        "canvas_shape": list(canvas_hidden.shape),
        "solidifier_preset": args.solidifier_preset,
        "canvas_zero_max_delta": canvas_zero_delta,
        "canvas_shuffle_max_delta": canvas_shuffle_delta,
        "trainable_gradient_tensors": trainable_gradient_tensors,
        "gradient_norm_sum": gradient_norm,
        "student_max_parameter_delta": student_delta,
        "frozen_max_parameter_delta": frozen_delta,
        "overfit_loss_start": overfit_losses[0],
        "overfit_loss_end": overfit_losses[-1],
        "rolling_records": rolling_records,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--canvas-length", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--entropy-bound", type=float, default=0.5)
    parser.add_argument(
        "--denoiser-sampling", choices=("multinomial", "argmax"), default="multinomial"
    )
    parser.add_argument("--solidifier-preset", choices=tuple(SOLIDIFIER_PRESETS), default="smoke")
    parser.add_argument("--max-prefix-length", type=int, default=64)
    parser.add_argument("--commits", type=int, default=4)
    parser.add_argument("--overfit-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-canvas-delta", type=float, default=1e-6)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.diffusion_steps < 1 or args.commits < 1 or args.overfit_steps < 1:
        parser.error("diffusion steps, commits, and overfit steps must be >= 1")
    result = run_validation(args)
    print(json.dumps(result, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"saved={output}")


if __name__ == "__main__":
    main()
