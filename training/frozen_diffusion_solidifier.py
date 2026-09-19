#!/usr/bin/env python3
"""Train a small solidifier against a frozen DiffusionGemma theorizer.

This is the first real frozen-theorizer path for RFCA.  DiffusionGemma stays
in inference mode and supplies the continuous decoder canvas plus the causal
encoder state.  A separate trainable causal solidifier consumes that state
and the canvas; a lightweight adapter-only control is also available.  The
training loop can teacher-force several rolling commits so the frozen
theorizer is queried with the same shifted canvas state that inference will
use.

The tiny public DiffusionGemma checkpoint is useful for interface tests:

    .venv-diffusion/bin/python training/frozen_diffusion_solidifier.py probe

The full checkpoint is selected explicitly when GPU memory is available:

    .venv-diffusion/bin/python training/frozen_diffusion_solidifier.py train \
      --model-id google/diffusiongemma-26B-A4B-it \
      --data path/to/train.txt \
      --commits 4

This script deliberately uses the frozen model's self-conditioning logits as
the carried diffusion state.  It does not turn those logits into committed
text or feed decoded canvas tokens to the solidifier.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import (
        AutoProcessor,
        DiffusionGemmaForBlockDiffusion,
        EntropyBoundSampler,
        EntropyBoundSamplerConfig,
    )
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "This script needs Transformers 5.11+ in a Python 3.10+ environment. "
        "Use .venv-diffusion or install the training requirements there."
    ) from exc


DEFAULT_MODEL_ID = "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion"

SOLIDIFIER_PRESETS = {
    "smoke": {"d_model": 128, "n_heads": 4, "layers": 2, "ff_dim": 512},
    "student-400m": {"d_model": 768, "n_heads": 12, "layers": 12, "ff_dim": 3072},
    "student-750m": {"d_model": 1152, "n_heads": 18, "layers": 24, "ff_dim": 4608},
    "student-1b": {"d_model": 1536, "n_heads": 24, "layers": 20, "ff_dim": 6144},
}


@dataclass
class CanvasState:
    canvas_ids: torch.Tensor
    self_conditioning_logits: Optional[torch.Tensor] = None


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32" or device.type == "mps":
        return torch.float32
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def append_prefix_token(
    prefix_ids: torch.Tensor,
    token_id: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    combined = torch.cat([prefix_ids, token_id], dim=1)
    if combined.shape[1] <= max_length:
        return combined
    if max_length < 2:
        return combined[:, -max_length:]
    return torch.cat(
        [combined[:, :1], combined[:, -(max_length - 1) :]], dim=1
    )


def limit_prefix_length(prefix_ids: torch.Tensor, max_length: int) -> torch.Tensor:
    if prefix_ids.shape[1] <= max_length:
        return prefix_ids
    if max_length < 2:
        return prefix_ids[:, -max_length:]
    return torch.cat([prefix_ids[:, :1], prefix_ids[:, -(max_length - 1) :]], dim=1)


class FrozenDiffusionGemmaTheorizer:
    """Thin wrapper around the official frozen DiffusionGemma forward path."""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        canvas_length: Optional[int] = None,
    ):
        model_path = Path(model_id).expanduser()
        if model_path.is_dir() and any(model_path.glob("*.dgq*")):
            raise ValueError(
                f"{model_id} is a native .dgq DiffusionGemma pack. "
                "The PyTorch adapter needs a Transformers checkpoint with "
                "model weights; use the installed Rust/Metal runtime for "
                "inference or provide a compatible Transformers checkpoint."
            )
        load_kwargs = {"dtype": dtype}
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            model_id, **load_kwargs
        )
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.device = device
        self.dtype = dtype
        self.vocab_size = self.model.config.text_config.vocab_size
        self.hidden_size = self.model.config.text_config.hidden_size
        self.canvas_length = canvas_length or self.model.config.canvas_length
        self.max_prefix_length = self.model.config.text_config.max_position_embeddings
        if self.canvas_length > self.model.config.canvas_length:
            raise ValueError(
                f"canvas_length={self.canvas_length} exceeds model canvas length "
                f"{self.model.config.canvas_length}"
            )

    def tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        encoded = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
        )
        return {
            key: value.to(self.device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }

    @torch.inference_mode()
    def initialize(self, batch_size: int) -> CanvasState:
        canvas_ids = torch.randint(
            0,
            self.vocab_size,
            (batch_size, self.canvas_length),
            device=self.device,
        )
        return CanvasState(canvas_ids=canvas_ids)

    @torch.no_grad()
    def refine(
        self,
        prefix_inputs: dict[str, torch.Tensor],
        state: CanvasState,
        steps: int,
        entropy_bound: float,
        denoiser_sampling: str = "multinomial",
    ) -> Tuple[CanvasState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine a canvas and return both prefix and canvas hidden states."""

        sampler = EntropyBoundSampler(
            EntropyBoundSamplerConfig(entropy_bound=entropy_bound),
            canvas_length=self.canvas_length,
            vocab_size=self.vocab_size,
            max_denoising_steps=steps,
        )
        current_canvas = state.canvas_ids
        self_conditioning_logits = state.self_conditioning_logits
        last_hidden = None
        last_logits = None
        last_prefix_hidden = None

        for step in range(steps):
            outputs = self.model(
                **prefix_inputs,
                decoder_input_ids=current_canvas,
                self_conditioning_logits=self_conditioning_logits,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits
            last_hidden = outputs.hidden_states[-1]
            last_logits = logits
            last_prefix_hidden = outputs.encoder_last_hidden_state
            if denoiser_sampling == "argmax":
                denoiser_canvas = logits.argmax(dim=-1)
            else:
                probabilities = torch.softmax(logits.float(), dim=-1)
                denoiser_canvas = torch.multinomial(
                    probabilities.reshape(-1, self.vocab_size), num_samples=1
                ).reshape_as(current_canvas)
            accepted_canvas = sampler.accept_canvas(
                current_canvas,
                denoiser_canvas,
                logits,
                step,
            )
            current_canvas = sampler.renoise_canvas(accepted_canvas, step)
            self_conditioning_logits = logits.to(self.dtype)

        if last_hidden is None or last_logits is None or last_prefix_hidden is None:
            raise RuntimeError("theorizer refinement ran zero steps")
        return (
            CanvasState(
                canvas_ids=current_canvas,
                self_conditioning_logits=self_conditioning_logits,
            ),
            last_hidden.detach(),
            last_logits.detach(),
            last_prefix_hidden.detach(),
        )

    @torch.inference_mode()
    def roll(self, state: CanvasState) -> CanvasState:
        """Drop the committed slot and append a fresh stochastic tail."""

        random_tail = torch.randint(
            0,
            self.vocab_size,
            (state.canvas_ids.shape[0], 1),
            device=self.device,
        )
        canvas_ids = torch.cat([state.canvas_ids[:, 1:], random_tail], dim=1)

        if state.self_conditioning_logits is None:
            next_self_conditioning = None
        else:
            tail_logits = torch.zeros(
                state.self_conditioning_logits.shape[0],
                1,
                self.vocab_size,
                device=self.device,
                dtype=state.self_conditioning_logits.dtype,
            )
            tail_logits.scatter_(2, random_tail[:, :, None], 10.0)
            next_self_conditioning = torch.cat(
                [state.self_conditioning_logits[:, 1:, :], tail_logits], dim=1
            )
        return CanvasState(
            canvas_ids=canvas_ids,
            self_conditioning_logits=next_self_conditioning,
        )


class LowRankCanvasCrossAttention(nn.Module):
    """Small trainable adapter from frozen canvas states to a commit vector."""

    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        rank = max(1, min(rank, hidden_size))
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.value = nn.Linear(hidden_size, rank, bias=False)
        self.output = nn.Linear(rank, hidden_size, bias=False)
        self.scale = rank**-0.5

    def forward(self, query: torch.Tensor, canvas: torch.Tensor) -> torch.Tensor:
        q = self.query(query)
        k = self.key(canvas)
        v = self.value(canvas)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        return self.output(torch.matmul(weights, v))


class TrainableAutoregressiveSolidifier(nn.Module):
    """A causal student that reads the frozen theorizer's continuous canvas."""

    def __init__(
        self,
        theorizer: FrozenDiffusionGemmaTheorizer,
        preset: str,
        max_prefix_length: int,
        canvas_mode: str = "full",
        extra_causal_layers: int = 0,
    ):
        super().__init__()
        if preset not in SOLIDIFIER_PRESETS:
            raise ValueError(f"unknown solidifier preset: {preset}")
        if canvas_mode not in ("full", "none"):
            raise ValueError(f"unknown canvas_mode: {canvas_mode}")
        if extra_causal_layers < 0:
            raise ValueError("extra_causal_layers must be >= 0")
        config = dict(SOLIDIFIER_PRESETS[preset])
        d_model = config["d_model"]
        prefix_layers = config["layers"] + extra_causal_layers
        self.config = {
            "preset": preset,
            **config,
            "canvas_mode": canvas_mode,
            "extra_causal_layers": extra_causal_layers,
            "prefix_layers": prefix_layers,
            "vocab_size": theorizer.vocab_size,
            "theorizer_hidden_size": theorizer.hidden_size,
            "canvas_length": theorizer.canvas_length,
            "max_prefix_length": max_prefix_length,
        }
        self.token_embedding = nn.Embedding(theorizer.vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config["n_heads"],
            dim_feedforward=config["ff_dim"],
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prefix_encoder = nn.TransformerEncoder(
            layer, num_layers=prefix_layers, enable_nested_tensor=False
        )
        self.frozen_prefix_bridge = nn.Linear(theorizer.hidden_size, d_model, bias=False)
        self.prefix_norm = nn.LayerNorm(d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.canvas_mode = canvas_mode
        if canvas_mode == "full":
            self.canvas_bridge = nn.Linear(theorizer.hidden_size, d_model, bias=False)
            self.canvas_position_embedding = nn.Embedding(
                theorizer.canvas_length, d_model
            )
            self.canvas_norm = nn.LayerNorm(d_model)
            self.canvas_cross_attention = nn.MultiheadAttention(
                d_model, config["n_heads"], dropout=0.0, batch_first=True
            )
            self.gate = nn.Parameter(torch.tensor(0.1))

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1
        )

    def forward(
        self,
        prefix_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        frozen_prefix_hidden: torch.Tensor,
        canvas_hidden: torch.Tensor,
    ) -> torch.Tensor:
        prefix_states = self.prefix_encoder(
            self.token_embedding(prefix_ids),
            mask=self._causal_mask(prefix_ids.shape[1], prefix_ids.device),
            src_key_padding_mask=~prefix_attention_mask.bool(),
        )
        query = self.prefix_norm(prefix_states[:, -1:, :])
        frozen_prefix_input = frozen_prefix_hidden[:, -1:, :].to(
            dtype=self.frozen_prefix_bridge.weight.dtype
        )
        query = query + self.frozen_prefix_bridge(frozen_prefix_input)
        if self.canvas_mode == "full":
            canvas_positions = torch.arange(
                canvas_hidden.shape[1], device=canvas_hidden.device
            )
            canvas_input = canvas_hidden.to(dtype=self.canvas_bridge.weight.dtype)
            canvas = self.canvas_bridge(canvas_input)
            canvas = canvas + self.canvas_position_embedding(canvas_positions)[None, :, :]
            canvas = self.canvas_norm(canvas)
            canvas_update, _ = self.canvas_cross_attention(
                query, canvas, canvas, need_weights=False
            )
            fused = self.fusion_norm(query + self.gate * canvas_update)
        else:
            fused = self.fusion_norm(query)
        return F.linear(fused[:, 0, :], self.token_embedding.weight)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }


class FrozenCanvasSolidifier(nn.Module):
    """Trainable adapter with the frozen theorizer's vocabulary head."""

    def __init__(self, theorizer: FrozenDiffusionGemmaTheorizer, adapter_rank: int):
        super().__init__()
        hidden_size = theorizer.hidden_size
        self.canvas_adapter = LowRankCanvasCrossAttention(hidden_size, adapter_rank)
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.lm_head = theorizer.model.lm_head
        for parameter in self.lm_head.parameters():
            parameter.requires_grad_(False)

    def forward(self, prefix_hidden: torch.Tensor, canvas_hidden: torch.Tensor) -> torch.Tensor:
        query = prefix_hidden[:, -1:, :]
        adapter_dtype = self.canvas_adapter.query.weight.dtype
        canvas_update = self.canvas_adapter(
            query.to(dtype=adapter_dtype),
            canvas_hidden.to(dtype=adapter_dtype),
        )
        fused = self.norm(query.to(dtype=adapter_dtype) + self.gate * canvas_update)
        lm_head_dtype = next(self.lm_head.parameters()).dtype
        return self.lm_head(fused[:, 0, :].to(dtype=lm_head_dtype))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only adapter weights, never a second copy of the frozen LM head."""

        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }


def load_texts(data: str) -> List[str]:
    if data == "synthetic":
        return [
            "The theorizer sketches a future sentence before the solidifier commits it.",
            "A rolling canvas preserves provisional context across several commits.",
            "The student should read the frozen diffusion canvas without seeing cleartext.",
            "Position and uncertainty remain part of the hidden state interface.",
        ]
    path = Path(data)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def examples_from_texts(
    theorizer: FrozenDiffusionGemmaTheorizer,
    texts: Sequence[str],
    horizon: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    examples = []
    tokenizer = theorizer.tokenizer
    for text in texts:
        ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        if ids.numel() <= horizon + 1:
            continue
        for start in range(0, ids.numel() - horizon - 1):
            prefix = ids[: start + 1].unsqueeze(0).to(theorizer.device)
            future = ids[start + 1 : start + horizon + 1].to(theorizer.device)
            examples.append((prefix, future))
    if not examples:
        raise ValueError("data produced no prefix/future examples")
    return examples


def load_examples(
    theorizer: FrozenDiffusionGemmaTheorizer,
    data: str,
    horizon: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return examples_from_texts(theorizer, load_texts(data), horizon)


def split_examples(
    theorizer: FrozenDiffusionGemmaTheorizer,
    data: str,
    horizon: int,
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Split source texts before windowing so validation is not a duplicate window."""

    texts = load_texts(data)
    if len(texts) < 2 or validation_fraction <= 0.0:
        examples = examples_from_texts(theorizer, texts, horizon)
        cut = max(1, int(len(examples) * 0.8))
        return examples[:cut], examples[cut:] or examples[-min(8, len(examples)) :]

    shuffled = list(texts)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_texts = shuffled[:validation_count]
    train_texts = shuffled[validation_count:] or shuffled
    train_examples = examples_from_texts(theorizer, train_texts, horizon)
    try:
        validation_examples = examples_from_texts(theorizer, validation_texts, horizon)
    except ValueError:
        # Very short line-oriented files may not leave a complete validation
        # window. Fall back to a disjoint example split rather than silently
        # reporting training-set quality as held-out quality.
        all_examples = examples_from_texts(theorizer, shuffled, horizon)
        cut = max(1, int(len(all_examples) * 0.8))
        train_examples = all_examples[:cut]
        validation_examples = all_examples[cut:] or all_examples[-min(8, len(all_examples)) :]
    return train_examples, validation_examples


def prefix_inputs_from_ids(prefix_ids: torch.Tensor, pad_token_id: int) -> dict[str, torch.Tensor]:
    attention_mask = (prefix_ids != pad_token_id).long()
    return {"input_ids": prefix_ids, "attention_mask": attention_mask}


@torch.no_grad()
def evaluate_examples(
    theorizer: FrozenDiffusionGemmaTheorizer,
    solidifier: nn.Module,
    examples: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    commits: int,
    diffusion_steps: int,
    entropy_bound: float,
    denoiser_sampling: str,
    max_examples: int,
    solidifier_mode: str,
) -> List[dict[str, float]]:
    """Evaluate teacher-forced rolling accuracy on held-out examples."""

    solidifier.eval()
    rows = [
        {"loss_sum": 0.0, "correct": 0.0, "count": 0.0}
        for _ in range(commits)
    ]
    for prefix_ids, future_ids in list(examples)[:max_examples]:
        prefix_ids = prefix_ids.clone()
        state = theorizer.initialize(batch_size=1)
        for commit in range(commits):
            prefix_inputs = prefix_inputs_from_ids(
                prefix_ids, theorizer.tokenizer.pad_token_id
            )
            state, canvas_hidden, _, prefix_hidden = theorizer.refine(
                prefix_inputs,
                state,
                diffusion_steps,
                entropy_bound,
                denoiser_sampling,
            )
            if solidifier_mode == "adapter":
                logits = solidifier(prefix_hidden, canvas_hidden)
            else:
                logits = solidifier(
                    prefix_ids,
                    prefix_inputs["attention_mask"],
                    prefix_hidden,
                    canvas_hidden,
                )
            target = future_ids[commit : commit + 1]
            rows[commit]["loss_sum"] += float(F.cross_entropy(logits, target).item())
            rows[commit]["correct"] += float((logits.argmax(dim=-1) == target).sum().item())
            rows[commit]["count"] += 1.0
            prefix_ids = append_prefix_token(
                prefix_ids,
                target[:, None],
                getattr(solidifier, "config", {}).get(
                    "max_prefix_length", theorizer.max_prefix_length
                ),
            )
            state = theorizer.roll(state)

    return [
        {
            "commit": index + 1,
            "loss": row["loss_sum"] / max(row["count"], 1.0),
            "accuracy": row["correct"] / max(row["count"], 1.0),
            "examples": row["count"],
        }
        for index, row in enumerate(rows)
    ]


def probe(args) -> None:
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    theorizer = FrozenDiffusionGemmaTheorizer(
        args.model_id, device, dtype, canvas_length=args.canvas_length
    )
    inputs = theorizer.tokenize([args.prompt])
    state = theorizer.initialize(batch_size=1)
    for step in range(args.commits):
        state, hidden, logits, _ = theorizer.refine(
            inputs,
            state,
            args.diffusion_steps,
            args.entropy_bound,
            args.denoiser_sampling,
        )
        print(
            f"commit={step + 1} canvas_ids={tuple(state.canvas_ids.shape)} "
            f"canvas_hidden={tuple(hidden.shape)} logits={tuple(logits.shape)} "
            f"self_conditioning={state.self_conditioning_logits is not None}"
        )
        state = theorizer.roll(state)
        next_token = torch.argmax(logits[:, 0, :], dim=-1, keepdim=True)
        inputs["input_ids"] = append_prefix_token(
            inputs["input_ids"],
            next_token,
            theorizer.max_prefix_length,
        )
        inputs["attention_mask"] = (inputs["input_ids"] != theorizer.tokenizer.pad_token_id).long()


def train(args) -> None:
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    theorizer = FrozenDiffusionGemmaTheorizer(
        args.model_id, device, dtype, canvas_length=args.canvas_length
    )
    max_prefix_length = min(args.solidifier_max_prefix_length, theorizer.max_prefix_length)
    if args.solidifier_mode == "adapter":
        solidifier = FrozenCanvasSolidifier(theorizer, args.adapter_rank).to(device)
        solidifier_config = {"mode": "adapter", "adapter_rank": args.adapter_rank}
    else:
        solidifier = TrainableAutoregressiveSolidifier(
            theorizer,
            args.solidifier_preset,
            max_prefix_length,
            canvas_mode=args.canvas_mode,
            extra_causal_layers=args.causal_extra_layers,
        ).to(device)
        solidifier_config = {"mode": "ar", **solidifier.config}
    trainable_parameters = [
        parameter for parameter in solidifier.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(
        f"solidifier_mode={args.solidifier_mode} "
        f"solidifier_preset={args.solidifier_preset if args.solidifier_mode == 'ar' else 'adapter'} "
        f"canvas_mode={args.canvas_mode if args.solidifier_mode == 'ar' else 'full'} "
        f"trainable_params={trainable_count}"
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_examples, validation_examples = split_examples(
        theorizer,
        args.data,
        args.commits,
        args.validation_fraction,
        args.seed,
    )
    print(
        f"train_examples={len(train_examples)} "
        f"validation_examples={len(validation_examples)} "
        f"commits={args.commits}"
    )
    rng = random.Random(args.seed)
    max_commits = args.commits

    solidifier.train()
    for step in range(1, args.steps + 1):
        prefix_ids, future_ids = rng.choice(train_examples)
        prefix_ids = limit_prefix_length(prefix_ids.clone(), max_prefix_length)
        state = theorizer.initialize(batch_size=1)
        total_loss = torch.zeros((), device=device)
        used_commits = 0

        for commit in range(max_commits):
            prefix_inputs = prefix_inputs_from_ids(
                prefix_ids, theorizer.tokenizer.pad_token_id
            )
            state, canvas_hidden, _, prefix_hidden = theorizer.refine(
                prefix_inputs,
                state,
                args.diffusion_steps,
                args.entropy_bound,
                args.denoiser_sampling,
            )
            if args.solidifier_mode == "adapter":
                logits = solidifier(prefix_hidden, canvas_hidden)
            else:
                logits = solidifier(
                    prefix_ids,
                    prefix_inputs["attention_mask"],
                    prefix_hidden,
                    canvas_hidden,
                )
            target = future_ids[commit : commit + 1]
            total_loss = total_loss + F.cross_entropy(logits, target)
            used_commits += 1

            # Teacher-force the known commitment, then roll the frozen canvas.
            prefix_ids = append_prefix_token(
                prefix_ids,
                target[:, None],
                max_prefix_length,
            )
            state = theorizer.roll(state)

        optimizer.zero_grad(set_to_none=True)
        (total_loss / max(used_commits, 1)).backward()
        torch.nn.utils.clip_grad_norm_(solidifier.parameters(), args.max_grad_norm)
        optimizer.step()
        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step}/{args.steps} "
                f"loss={total_loss.item() / max(used_commits, 1):.4f} "
                f"trainable_params={trainable_count}"
            )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_id": args.model_id,
                "model_config": theorizer.model.config.to_dict(),
                "solidifier_config": solidifier_config,
                "state_dict": solidifier.trainable_state_dict(),
                "args": vars(args),
            },
            output_path,
        )
        print(f"saved={output_path}")

    validation_metrics = evaluate_examples(
        theorizer,
        solidifier,
        validation_examples,
        args.commits,
        args.diffusion_steps,
        args.entropy_bound,
        args.denoiser_sampling,
        args.eval_examples,
        args.solidifier_mode,
    )
    print("held_out_rolling_metrics=")
    for row in validation_metrics:
        print(
            f"commit={int(row['commit'])} "
            f"loss={row['loss']:.4f} "
            f"accuracy={row['accuracy']:.4f} "
            f"examples={int(row['examples'])}"
        )
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "data": args.data,
                    "solidifier_config": solidifier_config,
                    "validation_examples": len(validation_examples),
                    "metrics": validation_metrics,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"metrics_saved={metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("probe", "validate frozen DiffusionGemma canvas hidden states and rolling"),
        ("train", "train the solidifier adapter against frozen DiffusionGemma"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
        command_parser.add_argument("--device", default="auto")
        command_parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
        command_parser.add_argument("--canvas-length", type=int, default=None)
        command_parser.add_argument("--diffusion-steps", type=int, default=2)
        command_parser.add_argument("--entropy-bound", type=float, default=0.5)
        command_parser.add_argument(
            "--denoiser-sampling", choices=("multinomial", "argmax"), default="multinomial"
        )
        command_parser.add_argument("--commits", type=int, default=4)
    subparsers.choices["probe"].add_argument("--prompt", default="Hello")

    train_parser = subparsers.choices["train"]
    train_parser.add_argument("--data", default="synthetic")
    train_parser.add_argument(
        "--solidifier-mode", choices=("ar", "adapter"), default="ar",
        help="train the causal student or the lightweight adapter-only control",
    )
    train_parser.add_argument(
        "--canvas-mode",
        choices=("full", "none"),
        default="full",
        help="use the frozen canvas or run the causal no-canvas control",
    )
    train_parser.add_argument(
        "--solidifier-preset", choices=tuple(SOLIDIFIER_PRESETS), default="smoke",
    )
    train_parser.add_argument("--solidifier-max-prefix-length", type=int, default=256)
    train_parser.add_argument(
        "--causal-extra-layers",
        type=int,
        default=1,
        help="extra causal layers for the no-canvas capacity control",
    )
    train_parser.add_argument("--adapter-rank", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--steps", type=int, default=100)
    train_parser.add_argument("--seed", type=int, default=2027)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--eval-examples", type=int, default=32)
    train_parser.add_argument("--output", default="runs/frozen-solidifier/adapter.pt")
    train_parser.add_argument("--metrics-output", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.diffusion_steps < 1 or args.commits < 1:
        parser.error("--diffusion-steps and --commits must be >= 1")
    if args.command == "train":
        if args.causal_extra_layers < 0:
            parser.error("--causal-extra-layers must be >= 0")
        if not 0.0 <= args.validation_fraction < 1.0:
            parser.error("--validation-fraction must be between 0 and 1")
        if args.eval_examples < 1:
            parser.error("--eval-examples must be >= 1")
    if args.command == "probe":
        probe(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
