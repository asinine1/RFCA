#!/usr/bin/env python3
"""Train a small Gemma-style rolling diffusion/AR student.

This is an intentionally explicit research prototype rather than a wrapper
around the released DiffusionGemma checkpoint.  The student has two paths:

* a causal prefix encoder, which represents committed tokens;
* a bidirectional canvas denoiser, which receives mask tokens and cross-attends
  to the committed prefix;
* a full-canvas solidifier, which predicts the next committed token from the
  final prefix state and every canvas position.

The default canvas input is all mask tokens.  The clean future is used only as
the training target, never as an input to the canvas.  That makes the first
prototype a causal conditional denoiser and avoids accidentally creating a
teacher-forced future-information channel.

The implementation is deliberately self-contained so the architecture can be
changed as the experiments become more specific.  It uses Gemma-like RMSNorm,
SwiGLU, tied embeddings, and scaled dot-product attention, but it does not
claim to reproduce Gemma 4 or DiffusionGemma weights/configuration.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import random
import time
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torch.utils.data import DataLoader, Dataset, DistributedSampler


@dataclasses.dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    prefix_layers: int
    canvas_layers: int
    ff_dim: int
    max_prefix_length: int
    max_canvas_length: int
    max_diffusion_steps: int = 16
    dropout: float = 0.0
    solidifier_layers: int = 0


PRESETS: Dict[str, Dict[str, int]] = {
    # The smoke preset is small enough to run on CPU/MPS and validates the
    # complete forward/backward path before spending accelerator time.
    "smoke": {
        "d_model": 256,
        "n_heads": 8,
        "prefix_layers": 2,
        "canvas_layers": 2,
        "ff_dim": 1024,
    },
    # With a Gemma-sized tokenizer (~262k entries), this is roughly 400M
    # parameters after accounting for the tied embedding table.
    "student-400m": {
        "d_model": 768,
        "n_heads": 12,
        "prefix_layers": 10,
        "canvas_layers": 10,
        "ff_dim": 3072,
    },
    # A reasonable first 2xA100 candidate if the local smoke test is healthy.
    "student-750m": {
        "d_model": 1024,
        "n_heads": 16,
        "prefix_layers": 16,
        "canvas_layers": 16,
        "ff_dim": 4096,
    },
    # The upper end of the requested range.  Reduce canvas/prefix lengths
    # before reducing width if memory is tight: the weights dominate here.
    "student-1b": {
        "d_model": 1280,
        "n_heads": 20,
        "prefix_layers": 12,
        "canvas_layers": 12,
        "ff_dim": 5120,
    },
}


class ByteTokenizer:
    """Dependency-free tokenizer for smoke tests and synthetic data."""

    vocab_size = 260
    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = 258
    mask_token_id = 259

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        data = bytes(i for i in ids if 0 <= i <= 255)
        return data.decode("utf-8", errors="replace")

    def save(self, output_dir: Path) -> None:
        (output_dir / "tokenizer.json").write_text(
            json.dumps(
                {
                    "type": "byte",
                    "vocab_size": self.vocab_size,
                    "bos_token_id": self.bos_token_id,
                    "eos_token_id": self.eos_token_id,
                    "pad_token_id": self.pad_token_id,
                    "mask_token_id": self.mask_token_id,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


class HuggingFaceTokenizer:
    """Adapter for a Gemma tokenizer, loaded only when requested."""

    def __init__(self, model_id: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError(
                "The Hugging Face tokenizer requires transformers. Install "
                "training/requirements.txt or use --tokenizer byte."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|rfca_mask|>"]}
        )
        self.vocab_size = len(self.tokenizer)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.bos_token_id is None:
            self.bos_token_id = self.eos_token_id or 1
        if self.eos_token_id is None:
            self.eos_token_id = self.bos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|rfca_mask|>")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, output_dir: Path) -> None:
        self.tokenizer.save_pretrained(output_dir / "tokenizer")


def build_tokenizer(tokenizer_name: str):
    if tokenizer_name == "byte":
        return ByteTokenizer()
    return HuggingFaceTokenizer(tokenizer_name)


def synthetic_texts(count: int) -> List[str]:
    """Create deterministic structured strings for a no-download smoke test."""

    examples = []
    for i in range(count):
        items = []
        for j in range(24):
            key = (i * 7 + j * 11) % 97
            value = (key * 3 + i) % 101
            items.append(f"item_{j:02d}=key_{key:02d}:value_{value:03d}")
        examples.append(" | ".join(items) + " <eos>")
    return examples


def read_texts(data_path: Optional[str], synthetic_count: int) -> List[str]:
    if data_path is None or data_path == "synthetic":
        return synthetic_texts(synthetic_count)

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data does not exist: {path}")

    texts: List[str] = []
    if path.suffix.lower() in {".jsonl", ".json"}:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if isinstance(record, str):
                texts.append(record)
            elif isinstance(record, dict):
                if "text" in record:
                    texts.append(str(record["text"]))
                elif "prompt" in record and "completion" in record:
                    texts.append(f"{record['prompt']}\n{record['completion']}")
                else:
                    raise ValueError(
                        f"JSON record on line {line_number} needs text or prompt/completion"
                    )
            else:
                raise ValueError(f"Unsupported JSON record on line {line_number}")
    else:
        texts = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not texts:
        raise ValueError(f"No non-empty examples found in {path}")
    return texts


class WindowDataset(Dataset):
    """Tokenize examples and expose fixed prefix/future windows."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        prefix_length: int,
        canvas_length: int,
        stride: Optional[int] = None,
    ):
        self.prefix_length = prefix_length
        self.canvas_length = canvas_length
        self.window_length = prefix_length + canvas_length
        self.pad_id = tokenizer.pad_token_id
        self.examples: List[torch.Tensor] = []
        step = stride or canvas_length

        for text in texts:
            ids = [tokenizer.bos_token_id]
            ids.extend(tokenizer.encode(text))
            ids.append(tokenizer.eos_token_id)
            if len(ids) < self.window_length:
                ids.extend([self.pad_id] * (self.window_length - len(ids)))
                self.examples.append(torch.tensor(ids, dtype=torch.long))
                continue
            for start in range(0, len(ids) - self.window_length + 1, step):
                self.examples.append(
                    torch.tensor(ids[start : start + self.window_length], dtype=torch.long)
                )

        if not self.examples:
            raise ValueError("Tokenized dataset produced no windows")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        window = self.examples[index]
        return {
            "prefix_ids": window[: self.prefix_length],
            "targets": window[self.prefix_length :],
        }


class PositionProbeDataset(Dataset):
    """A canvas-interface task whose useful symbol lives at the final position.

    The theorizer target is a sequence of distractor bytes followed by a
    decision byte.  The solidifier target is that final decision byte, so the
    first canvas position cannot solve the probe by itself.  The canvas is
    still initialized blank by the model; these IDs are labels only.
    """

    def __init__(
        self,
        tokenizer,
        prefix_length: int,
        canvas_length: int,
        count: int,
        seed: int,
    ):
        if not isinstance(tokenizer, ByteTokenizer):
            raise ValueError("position_probe currently requires --tokenizer byte")
        if canvas_length < 2:
            raise ValueError("position_probe needs canvas_length >= 2")
        rng = random.Random(seed)
        self.examples: List[Dict[str, torch.Tensor]] = []
        labels = b"ABCDEFGH"

        for index in range(count):
            left = rng.randrange(32)
            right = rng.randrange(32)
            label = labels[left % len(labels)]
            prefix_text = (
                f"position_probe left={left:02d} right={right:02d} "
                "commit="
            )
            prefix_ids = [tokenizer.bos_token_id] + tokenizer.encode(prefix_text)
            if len(prefix_ids) > prefix_length:
                prefix_ids = prefix_ids[:prefix_length]
            prefix_ids.extend([tokenizer.pad_token_id] * (prefix_length - len(prefix_ids)))

            distractors = [
                ord("a") + ((left + right + position * 3) % 4)
                for position in range(canvas_length - 1)
            ]
            canvas_targets = distractors + [label]
            self.examples.append(
                {
                    "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
                    "targets": torch.tensor(canvas_targets, dtype=torch.long),
                    "solidifier_targets": torch.tensor(label, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.examples[index]


class OrderedSequenceProbeDataset(Dataset):
    """A sequence-level canvas task with identical bags and different orders.

    Every example contains four words with numeric sort keys in the
    committed prefix. The theorizer must expand that instruction into the
    same four words sorted by key. The solidifier emits a class byte
    determined by that exact sorted order, so a pooled canvas has no
    order-invariant shortcut. The task is still deliberately synthetic: it
    tests whether ordered future structure survives the interface, not
    language quality or semantic generalization.
    """

    def __init__(
        self,
        tokenizer,
        prefix_length: int,
        canvas_length: int,
        count: int,
        seed: int,
    ):
        if not isinstance(tokenizer, ByteTokenizer):
            raise ValueError("ordered_sequence_probe currently requires --tokenizer byte")
        words = (b"red", b"blue", b"green", b"gold")
        orders = list(permutations(words))
        phrase_lengths = {
            sum(len(word) for word in order) + len(order) for order in orders
        }
        phrase_length = next(iter(phrase_lengths))
        if len(phrase_lengths) != 1:
            raise RuntimeError("ordered probe permutations must have equal lengths")
        if canvas_length < phrase_length:
            raise ValueError(
                f"ordered_sequence_probe needs canvas_length >= {phrase_length}"
            )
        rng = random.Random(seed)
        labels = bytes(range(33, 33 + len(orders)))
        self.examples: List[Dict[str, torch.Tensor]] = []

        for index in range(count):
            keys = [rng.randrange(10, 100) for _ in words]
            while len(set(keys)) != len(keys):
                keys = [rng.randrange(10, 100) for _ in words]
            sorted_indices = sorted(range(len(words)), key=lambda item: keys[item])
            order = tuple(words[item] for item in sorted_indices)
            order_index = orders.index(order)
            phrase = b">".join(order) + b";"
            prefix_text = (
                "sort sequence "
                f"red={keys[0]:02d} blue={keys[1]:02d} "
                f"green={keys[2]:02d} gold={keys[3]:02d} "
                "response="
            )
            prefix_ids = [tokenizer.bos_token_id] + tokenizer.encode(prefix_text)
            if len(prefix_ids) > prefix_length:
                prefix_ids = prefix_ids[:prefix_length]
            prefix_ids.extend(
                [tokenizer.pad_token_id] * (prefix_length - len(prefix_ids))
            )
            canvas_targets = list(phrase)
            canvas_targets.extend([ord(".")] * (canvas_length - len(canvas_targets)))
            self.examples.append(
                {
                    "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
                    "targets": torch.tensor(canvas_targets, dtype=torch.long),
                    "solidifier_targets": torch.tensor(
                        labels[order_index], dtype=torch.long
                    ),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.examples[index]


class IndexedSequenceProbeDataset(Dataset):
    """Ask for one indexed item from an ordered future sequence.

    The prefix supplies sort keys and asks for the final item. The canvas must
    contain the sorted phrase, while the solidifier target is the first byte
    of the final word. Every example has the same bag of words, so the task
    tests whether the interface preserves a specific late position.
    """

    def __init__(
        self,
        tokenizer,
        prefix_length: int,
        canvas_length: int,
        count: int,
        seed: int,
    ):
        if not isinstance(tokenizer, ByteTokenizer):
            raise ValueError("indexed_sequence_probe currently requires --tokenizer byte")
        words = (b"red", b"blue", b"cyan", b"gold")
        phrase_length = sum(len(word) for word in words) + len(words)
        if canvas_length < phrase_length:
            raise ValueError(
                f"indexed_sequence_probe needs canvas_length >= {phrase_length}"
            )
        rng = random.Random(seed)
        self.examples: List[Dict[str, torch.Tensor]] = []

        for _ in range(count):
            keys = [rng.randrange(10, 100) for _ in words]
            while len(set(keys)) != len(keys):
                keys = [rng.randrange(10, 100) for _ in words]
            sorted_indices = sorted(range(len(words)), key=lambda item: keys[item])
            order = tuple(words[item] for item in sorted_indices)
            query_index = len(order) - 1
            prefix_text = (
                "index sorted sequence "
                f"red={keys[0]:02d} blue={keys[1]:02d} "
                f"cyan={keys[2]:02d} gold={keys[3]:02d} "
                "query=last response="
            )
            prefix_ids = [tokenizer.bos_token_id] + tokenizer.encode(prefix_text)
            if len(prefix_ids) > prefix_length:
                prefix_ids = prefix_ids[:prefix_length]
            prefix_ids.extend(
                [tokenizer.pad_token_id] * (prefix_length - len(prefix_ids))
            )
            phrase = b">".join(order) + b";"
            canvas_targets = list(phrase)
            canvas_targets.extend([ord(".")] * (canvas_length - len(canvas_targets)))
            self.examples.append(
                {
                    "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
                    "targets": torch.tensor(canvas_targets, dtype=torch.long),
                    "solidifier_targets": torch.tensor(
                        order[query_index][0], dtype=torch.long
                    ),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.examples[index]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype=self.weight.dtype)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"d_model={dim} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"d_model={dim} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, query_length, dim = query.shape
        context_length = context.shape[1]
        q = self.q(query).view(batch, query_length, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.view(batch, context_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, context_length, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, query_length, dim)
        return self.out(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, ff_dim, bias=False)
        self.up = nn.Linear(dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, use_cross_attention: bool):
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model)
        self.self_attn = SelfAttention(config.d_model, config.n_heads, config.dropout)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.norm_cross = RMSNorm(config.d_model)
            self.cross_attn = CrossAttention(
                config.d_model, config.n_heads, config.dropout
            )
        self.norm_ff = RMSNorm(config.d_model)
        self.ff = SwiGLU(config.d_model, config.ff_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm_attn(x), causal=causal)
        if self.use_cross_attention:
            if context is None:
                raise ValueError("Canvas blocks require prefix context")
            x = x + self.cross_attn(self.norm_cross(x), context)
        x = x + self.ff(self.norm_ff(x))
        return x


class RFCAStudent(nn.Module):
    """A compact, trainable approximation of the RFCA interface."""

    def __init__(self, config: ModelConfig, gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # A blank canvas must not be initialized with the tied embedding of a
        # vocabulary token.  Doing so lets the output head recover that token
        # directly and creates artificial "confidence" before denoising.
        self.canvas_init = nn.Parameter(torch.zeros(config.d_model))
        self.prefix_position = nn.Embedding(config.max_prefix_length, config.d_model)
        self.canvas_position = nn.Embedding(config.max_canvas_length, config.d_model)
        self.diffusion_time = nn.Embedding(config.max_diffusion_steps, config.d_model)

        self.prefix_blocks = nn.ModuleList(
            TransformerBlock(config, use_cross_attention=False)
            for _ in range(config.prefix_layers)
        )
        self.canvas_blocks = nn.ModuleList(
            TransformerBlock(config, use_cross_attention=True)
            for _ in range(config.canvas_layers)
        )
        self.solidifier_norm = RMSNorm(config.d_model)
        self.solidifier_cross = (
            CrossAttention(config.d_model, config.n_heads, config.dropout)
            if config.canvas_layers > 0
            else None
        )
        self.solidifier_blocks = nn.ModuleList(
            TransformerBlock(config, use_cross_attention=False)
            for _ in range(config.solidifier_layers)
        )
        self.output_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tying keeps the student compact and matches a common Gemma design.
        self.lm_head.weight = self.token_embedding.weight

    def _run_block(
        self,
        block: TransformerBlock,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return gradient_checkpoint(
                lambda value: block(value, context=context, causal=causal),
                x,
                use_reentrant=False,
            )
        return block(x, context=context, causal=causal)

    def _run_blocks(
        self,
        blocks: Iterable[TransformerBlock],
        x: torch.Tensor,
        *,
        context: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        for block in blocks:
            x = self._run_block(block, x, context, causal)
        return x

    def encode_prefix(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Encode only committed tokens with causal self-attention."""

        prefix_length = prefix_ids.shape[1]
        if prefix_length > self.config.max_prefix_length:
            raise ValueError("prefix length exceeds model config")
        prefix_positions = torch.arange(prefix_length, device=prefix_ids.device)
        prefix = self.token_embedding(prefix_ids) + self.prefix_position(prefix_positions)
        return self._run_blocks(
            self.prefix_blocks, prefix, context=None, causal=True
        )

    def initialize_canvas(self, canvas_ids: torch.Tensor) -> torch.Tensor:
        """Create a continuous blank canvas state.

        ``canvas_ids`` determines the shape only.  The baseline does not feed
        a readable mask-token embedding into the tied output head.
        """

        canvas_length = canvas_ids.shape[1]
        if canvas_length > self.config.max_canvas_length:
            raise ValueError("canvas length exceeds model config")
        canvas_positions = torch.arange(canvas_length, device=canvas_ids.device)
        canvas = self.canvas_init.view(1, 1, -1).expand(
            canvas_ids.shape[0], canvas_length, -1
        )
        return canvas + self.canvas_position(canvas_positions)

    def theorize_once(
        self,
        prefix_states: torch.Tensor,
        canvas_states: torch.Tensor,
        diffusion_step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one noncausal theorizer pass over a continuous canvas.

        The canvas remains in hidden-state form between calls.  This is
        important for RFCA: an intermediate canvas should carry provisional
        structure without handing the solidifier a sequence of argmax tokens.
        """

        if not 0 <= diffusion_step < self.config.max_diffusion_steps:
            raise ValueError("diffusion_step is outside model config")
        canvas_states = canvas_states + self.diffusion_time.weight[diffusion_step].view(
            1, 1, -1
        )
        canvas_states = self._run_blocks(
            self.canvas_blocks,
            canvas_states,
            context=prefix_states,
            causal=False,
        )
        logits = self.canvas_logits(canvas_states)
        return canvas_states, logits

    def canvas_logits(self, canvas_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.output_norm(canvas_states))

    def solidify(
        self,
        prefix_states: torch.Tensor,
        canvas_states: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one next committed token from the whole canvas."""

        solidifier_states = prefix_states
        if self.solidifier_cross is not None:
            canvas_length = canvas_states.shape[1]
            canvas_positions = torch.arange(canvas_length, device=canvas_states.device)
            # Re-inject slot identity at the commitment boundary.  The denoiser
            # is allowed to exchange information bidirectionally, but the
            # solidifier must still distinguish provisional destination slots.
            canvas_for_read = canvas_states + self.canvas_position(canvas_positions)
            query = self.solidifier_norm(prefix_states[:, -1:, :])
            canvas_update = self.solidifier_cross(query, canvas_for_read)
            solidifier_states = prefix_states.clone()
            solidifier_states[:, -1:, :] = solidifier_states[:, -1:, :] + canvas_update
        solidifier_states = self._run_blocks(
            self.solidifier_blocks,
            solidifier_states,
            context=None,
            causal=True,
        )
        return self.lm_head(self.output_norm(solidifier_states[:, -1, :]))

    def roll_canvas(
        self,
        canvas_states: torch.Tensor,
        mask_token_id: int,
    ) -> torch.Tensor:
        """Shift a continuous canvas and initialize its newly exposed tail.

        The position embeddings are re-based after the shift.  The operation
        is deliberately simple and measurable; more elaborate partial-refresh
        policies belong in later ablations.
        """

        batch, canvas_length, _ = canvas_states.shape
        if canvas_length < 2:
            mask_ids = torch.full(
                (batch, 1), mask_token_id, device=canvas_states.device, dtype=torch.long
            )
            return self.initialize_canvas(mask_ids)

        positions = self.canvas_position.weight[:canvas_length]
        rolled = canvas_states[:, 1:, :] - positions[1:canvas_length].view(
            1, canvas_length - 1, -1
        )
        rolled = rolled + positions[: canvas_length - 1].view(
            1, canvas_length - 1, -1
        )
        tail_ids = torch.full(
            (batch, 1), mask_token_id, device=canvas_states.device, dtype=torch.long
        )
        tail = self.initialize_canvas(tail_ids)
        return torch.cat([rolled, tail], dim=1)

    def forward(
        self,
        prefix_ids: torch.Tensor,
        canvas_ids: torch.Tensor,
        diffusion_steps: int,
    ) -> Dict[str, torch.Tensor]:
        prefix_length = prefix_ids.shape[1]
        canvas_length = canvas_ids.shape[1]
        if prefix_length > self.config.max_prefix_length:
            raise ValueError("prefix length exceeds model config")
        if canvas_length > self.config.max_canvas_length:
            raise ValueError("canvas length exceeds model config")
        if not 1 <= diffusion_steps <= self.config.max_diffusion_steps:
            raise ValueError("diffusion_steps is outside model config")

        prefix = self.encode_prefix(prefix_ids)
        canvas = self.initialize_canvas(canvas_ids)
        for step in range(diffusion_steps):
            canvas, diffusion_logits = self.theorize_once(prefix, canvas, step)

        # The last committed position queries every provisional canvas
        # position.  This is the RFCA candidate interface; replacing this with
        # canvas[:, :1] or canvas.mean(dim=1, keepdim=True) gives controls.
        solidifier_logits = self.solidify(prefix, canvas)

        return {
            "diffusion_logits": diffusion_logits,
            "solidifier_logits": solidifier_logits,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed() -> Tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-process training currently requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return distributed, rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean(value: float, device: torch.device, distributed: bool) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_model_config(args, vocab_size: int) -> ModelConfig:
    preset = PRESETS[args.model_preset]
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=preset["d_model"],
        n_heads=preset["n_heads"],
        prefix_layers=(
            args.prefix_layers
            if getattr(args, "prefix_layers", None) is not None
            else preset["prefix_layers"]
        ),
        canvas_layers=(
            args.canvas_layers
            if getattr(args, "canvas_layers", None) is not None
            else preset["canvas_layers"]
        ),
        ff_dim=preset["ff_dim"],
        max_prefix_length=args.prefix_length,
        max_canvas_length=args.canvas_length,
        max_diffusion_steps=max(args.diffusion_steps, 1),
        dropout=args.dropout,
        solidifier_layers=max(int(getattr(args, "solidifier_layers", 0)), 0),
    )


def next_batch(loader: DataLoader, iterator: Optional[Iterator], sampler, epoch: int):
    try:
        batch = next(iterator)
    except (StopIteration, TypeError):
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator, epoch


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    pad_id: int,
    diffusion_weight: float,
    solidifier_weight: float,
    solidifier_targets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    diffusion_logits = outputs["diffusion_logits"]
    solidifier_logits = outputs["solidifier_logits"]
    diffusion_loss = F.cross_entropy(
        diffusion_logits.reshape(-1, diffusion_logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=pad_id,
    )
    if solidifier_targets is None:
        solidifier_targets = targets[:, 0]
    solidifier_loss = F.cross_entropy(
        solidifier_logits,
        solidifier_targets,
        ignore_index=pad_id,
    )
    total = diffusion_weight * diffusion_loss + solidifier_weight * solidifier_loss
    return total, {
        "loss": float(total.detach().item()),
        "diffusion_loss": float(diffusion_loss.detach().item()),
        "solidifier_loss": float(solidifier_loss.detach().item()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args,
    pad_id: int,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "diffusion_loss": 0.0, "solidifier_loss": 0.0}
    count = 0
    for batch in loader:
        if count >= max_batches:
            break
        prefix_ids = batch["prefix_ids"].to(device)
        targets = batch["targets"].to(device)
        solidifier_targets = batch.get("solidifier_targets")
        if solidifier_targets is not None:
            solidifier_targets = solidifier_targets.to(device)
        canvas_ids = torch.full_like(targets, args.mask_id)
        with autocast_context(device, args.precision):
            outputs = model(prefix_ids, canvas_ids, args.diffusion_steps)
            _, metrics = compute_losses(
                outputs,
                targets,
                pad_id,
                args.diffusion_loss_weight,
                args.solidifier_loss_weight,
                solidifier_targets=solidifier_targets,
            )
        for key in totals:
            totals[key] += metrics[key]
        count += 1
    model.train()
    if count == 0:
        return totals
    return {key: value / count for key, value in totals.items()}


def distribution_metrics(
    logits: torch.Tensor,
    ignore_token_ids: Optional[Sequence[int]] = None,
) -> Tuple[float, float, float]:
    """Return mean confidence, entropy in nats, and normalized entropy."""

    logits = logits.float()
    if ignore_token_ids:
        logits = logits.clone()
        logits[..., list(ignore_token_ids)] = torch.finfo(logits.dtype).min
    probabilities = torch.softmax(logits, dim=-1)
    confidence = float(probabilities.max(dim=-1).values.mean().item())
    entropy = float(
        (-(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1))
        .mean()
        .item()
    )
    normalized_entropy = entropy / max(math.log(logits.shape[-1]), 1e-9)
    return confidence, entropy, normalized_entropy


def stop_metric_value(
    logits: torch.Tensor,
    stop_metric: str,
    ignore_token_ids: Optional[Sequence[int]] = None,
) -> Tuple[float, float, float]:
    confidence, entropy, normalized_entropy = distribution_metrics(
        logits, ignore_token_ids=ignore_token_ids
    )
    if stop_metric == "confidence":
        return confidence, entropy, normalized_entropy
    if stop_metric == "entropy":
        return entropy, entropy, normalized_entropy
    if stop_metric == "normalized_entropy":
        return normalized_entropy, entropy, normalized_entropy
    raise ValueError(f"Unknown theorizer stop metric: {stop_metric}")


def stop_metric_satisfied(value: float, stop_metric: str, threshold: float) -> bool:
    if stop_metric == "confidence":
        return value >= threshold
    return value <= threshold


def append_committed_token(
    prefix_ids: torch.Tensor,
    next_token: torch.Tensor,
    max_prefix_length: int,
) -> torch.Tensor:
    """Append a committed token while respecting the prefix context window."""

    combined = torch.cat([prefix_ids, next_token], dim=1)
    if combined.shape[1] <= max_prefix_length:
        return combined
    if max_prefix_length < 2:
        return combined[:, -max_prefix_length:]
    # Preserve BOS when possible and keep the newest committed context. This
    # is separate from canvas rolling: the canvas retains provisional future
    # state, while the causal prefix retains the most recent committed tokens.
    return torch.cat(
        [combined[:, :1], combined[:, -(max_prefix_length - 1) :]], dim=1
    )


def forward_teacher_forced_rolling(
    model: RFCAStudent,
    prefix_ids: torch.Tensor,
    targets: torch.Tensor,
    mask_token_id: int,
    diffusion_steps: int,
    commits: int,
) -> Dict[str, object]:
    """Run a short teacher-forced rolling trajectory for curriculum training.

    The target tokens are appended to the causal prefix exactly as they are in
    rolling evaluation. The continuous canvas is then shifted one slot and
    reused for the next commit instead of being reset. This exposes the final
    solidifier loss to the same stale, partially refreshed canvas states that
    inference will produce.
    """

    if commits < 1:
        raise ValueError("rolling curriculum requires at least one commit")
    if commits >= targets.shape[1]:
        raise ValueError("rolling curriculum commits must leave a target token")

    core_model = model.module if isinstance(model, DDP) else model
    mask_ids = torch.full_like(targets, mask_token_id)
    retained_canvas = core_model.initialize_canvas(mask_ids)
    diffusion_records: List[Tuple[torch.Tensor, int]] = []
    solidifier_logits = None

    for commit_index in range(commits + 1):
        prefix_states = core_model.encode_prefix(prefix_ids)
        diffusion_logits = None
        for diffusion_step in range(diffusion_steps):
            retained_canvas, diffusion_logits = core_model.theorize_once(
                prefix_states,
                retained_canvas,
                diffusion_step % core_model.config.max_diffusion_steps,
            )
        if diffusion_logits is None:
            raise RuntimeError("rolling curriculum produced no diffusion logits")
        diffusion_records.append((diffusion_logits, commit_index))
        solidifier_logits = core_model.solidify(prefix_states, retained_canvas)

        if commit_index == commits:
            break
        teacher_token = targets[:, commit_index : commit_index + 1]
        prefix_ids = append_committed_token(
            prefix_ids,
            teacher_token,
            core_model.config.max_prefix_length,
        )
        retained_canvas = core_model.roll_canvas(retained_canvas, mask_token_id)

    if solidifier_logits is None:
        raise RuntimeError("rolling curriculum produced no solidifier logits")
    return {
        "diffusion_logits": diffusion_records[-1][0],
        "solidifier_logits": solidifier_logits,
        "rolling_diffusion_records": diffusion_records,
        "rolling_commit": commits,
    }


def compute_rolling_losses(
    outputs: Dict[str, object],
    targets: torch.Tensor,
    pad_id: int,
    diffusion_weight: float,
    solidifier_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Score each available future slice and the final rolling commitment."""

    records = outputs["rolling_diffusion_records"]
    diffusion_losses = []
    for diffusion_logits, offset in records:
        usable = min(diffusion_logits.shape[1], targets.shape[1] - offset)
        if usable < 1:
            continue
        diffusion_losses.append(
            F.cross_entropy(
                diffusion_logits[:, :usable, :].reshape(-1, diffusion_logits.shape[-1]),
                targets[:, offset : offset + usable].reshape(-1),
                ignore_index=pad_id,
            )
        )
    if not diffusion_losses:
        raise RuntimeError("rolling curriculum had no valid diffusion targets")
    diffusion_loss = torch.stack(diffusion_losses).mean()
    solidifier_target = targets[:, int(outputs["rolling_commit"])]
    solidifier_loss = F.cross_entropy(
        outputs["solidifier_logits"],
        solidifier_target,
        ignore_index=pad_id,
    )
    total = diffusion_weight * diffusion_loss + solidifier_weight * solidifier_loss
    return total, {
        "loss": float(total.detach().item()),
        "diffusion_loss": float(diffusion_loss.detach().item()),
        "solidifier_loss": float(solidifier_loss.detach().item()),
    }


@torch.no_grad()
def interpolate_theorizer_boundary(
    model: RFCAStudent,
    before: torch.Tensor,
    after: torch.Tensor,
    stop_metric: str,
    threshold: float,
    device: torch.device,
    precision: str,
    ignore_token_ids: Optional[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Keep an overshooting pass from turning the canvas into cleartext.

    A full denoiser pass can jump from very uncertain to very confident.  We
    retain a point on the hidden-state trajectory near the requested boundary
    rather than handing the solidifier the overshot endpoint.  This is an
    interpolation control, not an extra teacher signal.
    """

    with autocast_context(device, precision):
        before_logits = model.canvas_logits(before)
        before_value, _, _ = stop_metric_value(
            before_logits, stop_metric, ignore_token_ids
        )
    if stop_metric_satisfied(before_value, stop_metric, threshold):
        return before, before_logits, False

    low = 0.0
    high = 1.0
    for _ in range(8):
        alpha = (low + high) / 2.0
        candidate = before + alpha * (after - before)
        with autocast_context(device, precision):
            candidate_logits = model.canvas_logits(candidate)
            candidate_value, _, _ = stop_metric_value(
                candidate_logits, stop_metric, ignore_token_ids
            )
        if stop_metric_satisfied(candidate_value, stop_metric, threshold):
            high = alpha
        else:
            low = alpha

    boundary = before + high * (after - before)
    with autocast_context(device, precision):
        boundary_logits = model.canvas_logits(boundary)
    return boundary, boundary_logits, True


def choose_inference_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def rolling_generate(
    model: RFCAStudent,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int,
    canvas_length: int,
    max_theorizer_steps: int,
    stop_metric: str,
    confidence_threshold: float,
    entropy_threshold: float,
    temperature: float,
    precision: str,
    interpolate_boundary: bool,
) -> Tuple[str, List[Dict[str, float]]]:
    """Run the one-token solidifier / rolling theorizer loop.

    The canvas is refined as continuous hidden states.  No intermediate
    argmax canvas is fed back into the theorizer or exposed as readable token
    IDs to the solidifier.
    """

    model.eval()
    initial_ids = [tokenizer.bos_token_id]
    initial_ids.extend(tokenizer.encode(prompt))
    prefix_ids = torch.tensor([initial_ids], dtype=torch.long, device=device)
    if prefix_ids.shape[1] > model.config.max_prefix_length:
        prefix_ids = prefix_ids[:, -model.config.max_prefix_length :]
    mask_ids = torch.full(
        (1, canvas_length),
        tokenizer.mask_token_id,
        dtype=torch.long,
        device=device,
    )
    ignore_canvas_token_ids = [tokenizer.mask_token_id, tokenizer.pad_token_id]
    canvas_states = model.initialize_canvas(mask_ids)
    generated_ids: List[int] = []
    trace: List[Dict[str, float]] = []

    for commit_index in range(max_new_tokens):
        with autocast_context(device, precision):
            prefix_states = model.encode_prefix(prefix_ids)
        passes = 0
        confidence = 0.0
        entropy = float("inf")
        normalized_entropy = float("inf")
        stopped_by_threshold = False
        boundary_interpolated = False

        for theorizer_step in range(max_theorizer_steps):
            previous_canvas_states = canvas_states
            with autocast_context(device, precision):
                canvas_states, canvas_logits = model.theorize_once(
                    prefix_states,
                    canvas_states,
                    theorizer_step % model.config.max_diffusion_steps,
            )
            confidence, entropy, normalized_entropy = distribution_metrics(
                canvas_logits, ignore_token_ids=ignore_canvas_token_ids
            )
            passes = theorizer_step + 1
            threshold = (
                confidence_threshold if stop_metric == "confidence" else entropy_threshold
            )
            metric_value, _, _ = stop_metric_value(
                canvas_logits,
                stop_metric,
                ignore_token_ids=ignore_canvas_token_ids,
            )
            stopped_by_threshold = stop_metric_satisfied(
                metric_value, stop_metric, threshold
            )
            if stopped_by_threshold:
                if interpolate_boundary:
                    (
                        canvas_states,
                        canvas_logits,
                        did_interpolate,
                    ) = interpolate_theorizer_boundary(
                        model,
                        previous_canvas_states,
                        canvas_states,
                        stop_metric,
                        threshold,
                        device,
                        precision,
                        ignore_canvas_token_ids,
                    )
                    confidence, entropy, normalized_entropy = distribution_metrics(
                        canvas_logits, ignore_token_ids=ignore_canvas_token_ids
                    )
                    boundary_interpolated = did_interpolate
                break

        with autocast_context(device, precision):
            solidifier_logits = model.solidify(prefix_states, canvas_states)
        if temperature > 0:
            probabilities = torch.softmax(solidifier_logits / temperature, dim=-1)
            next_token = int(torch.multinomial(probabilities, num_samples=1).item())
        else:
            next_token = int(solidifier_logits.argmax(dim=-1).item())
        generated_ids.append(next_token)
        prefix_ids = append_committed_token(
            prefix_ids,
            torch.tensor([[next_token]], device=device),
            model.config.max_prefix_length,
        )

        trace.append(
            {
                "commit_index": float(commit_index),
                "theorizer_passes": float(passes),
                "mean_confidence": confidence,
                "mean_entropy_nats": entropy,
                "mean_normalized_entropy": normalized_entropy,
                "stopped_by_threshold": float(stopped_by_threshold),
                "boundary_interpolated": float(boundary_interpolated),
            }
        )
        if next_token == tokenizer.eos_token_id:
            break
        canvas_states = model.roll_canvas(canvas_states, tokenizer.mask_token_id)

    return tokenizer.decode(generated_ids), trace


def generate(args) -> None:
    device = choose_inference_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    tokenizer = build_tokenizer(args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({tokenizer.vocab_size}) does not match checkpoint "
            f"vocab ({model_config.vocab_size})"
        )
    model = RFCAStudent(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    canvas_length = args.canvas_length or model_config.max_canvas_length
    completion, trace = rolling_generate(
        model,
        tokenizer,
        args.prompt,
        device,
        max_new_tokens=args.max_new_tokens,
        canvas_length=canvas_length,
        max_theorizer_steps=args.max_theorizer_steps,
        stop_metric=args.theorizer_stop,
        confidence_threshold=args.theorizer_confidence_threshold,
        entropy_threshold=args.theorizer_entropy_threshold,
        temperature=args.temperature,
        precision=args.precision,
        interpolate_boundary=not args.no_boundary_interpolation,
    )
    print(f"device={device} checkpoint_step={checkpoint.get('step', 'unknown')}")
    print(
        f"theorizer_stop={args.theorizer_stop} "
        f"confidence_threshold={args.theorizer_confidence_threshold} "
        f"entropy_threshold={args.theorizer_entropy_threshold} "
        f"max_theorizer_steps={args.max_theorizer_steps} "
        f"boundary_interpolation={not args.no_boundary_interpolation}"
    )
    print(f"completion={completion!r}")
    for item in trace:
        print(
            f"commit={int(item['commit_index'])} "
            f"passes={int(item['theorizer_passes'])} "
            f"confidence={item['mean_confidence']:.4f} "
            f"entropy_nats={item['mean_entropy_nats']:.4f} "
            f"normalized_entropy={item['mean_normalized_entropy']:.4f} "
            f"threshold_stop={bool(item['stopped_by_threshold'])} "
            f"boundary_interpolated={bool(item['boundary_interpolated'])}"
        )
    if args.trace_output:
        Path(args.trace_output).write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "prompt": args.prompt,
                    "completion": completion,
                    "trace": trace,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _diagnostic_loss_and_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
) -> Tuple[float, float, int]:
    valid = targets != pad_id
    if not bool(valid.any()):
        return 0.0, 0.0, 0
    valid_logits = logits[valid]
    valid_targets = targets[valid]
    loss = F.cross_entropy(valid_logits, valid_targets, reduction="sum")
    accuracy = (valid_logits.argmax(dim=-1) == valid_targets).float().sum()
    count = int(valid_targets.numel())
    return float(loss.item()), float(accuracy.item()), count


def diagnose(args) -> None:
    """Measure whether the solidifier's decision changes when the canvas changes."""

    device = choose_inference_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    saved_args = checkpoint.get("args", {})
    tokenizer = build_tokenizer(args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({tokenizer.vocab_size}) does not match checkpoint "
            f"vocab ({model_config.vocab_size})"
        )

    prefix_length = args.prefix_length or int(
        saved_args.get("prefix_length", model_config.max_prefix_length)
    )
    canvas_length = args.canvas_length or int(
        saved_args.get("canvas_length", model_config.max_canvas_length)
    )
    diffusion_steps = args.diffusion_steps or int(saved_args.get("diffusion_steps", 1))
    task = args.task or saved_args.get("task", "standard")
    if task == "position_probe":
        probe_examples = args.probe_examples or int(
            saved_args.get("probe_examples", 1024)
        )
        dataset = PositionProbeDataset(
            tokenizer,
            prefix_length,
            canvas_length,
            probe_examples,
            args.seed,
        )
    elif task == "ordered_sequence_probe":
        probe_examples = args.probe_examples or int(
            saved_args.get("probe_examples", 1024)
        )
        dataset = OrderedSequenceProbeDataset(
            tokenizer,
            prefix_length,
            canvas_length,
            probe_examples,
            args.seed,
        )
    elif task == "indexed_sequence_probe":
        probe_examples = args.probe_examples or int(
            saved_args.get("probe_examples", 1024)
        )
        dataset = IndexedSequenceProbeDataset(
            tokenizer,
            prefix_length,
            canvas_length,
            probe_examples,
            args.seed,
        )
    else:
        data_name = args.data or saved_args.get("data", "synthetic")
        synthetic_count = args.synthetic_examples or int(
            saved_args.get("synthetic_examples", 256)
        )
        texts = read_texts(data_name, synthetic_count)
        dataset = WindowDataset(
            texts,
            tokenizer,
            prefix_length,
            canvas_length,
            stride=args.window_stride,
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = RFCAStudent(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.manual_seed(args.seed)

    variant_names = (
        "full",
        "no_canvas",
        "first_position",
        "pooled",
        "shuffled",
        "random",
    )
    totals = {
        name: {"loss_sum": 0.0, "correct": 0.0, "count": 0} for name in variant_names
    }
    comparisons = {
        name: {"kl_sum": 0.0, "agreement": 0.0, "count": 0}
        for name in variant_names
        if name != "full"
    }
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= args.samples:
                break
            remaining = args.samples - seen
            if batch["prefix_ids"].shape[0] > remaining:
                batch = {key: value[:remaining] for key, value in batch.items()}
            prefix_ids = batch["prefix_ids"].to(device)
            targets = batch.get("solidifier_targets", batch["targets"][:, 0]).to(device)
            with autocast_context(device, args.precision):
                prefix_states = model.encode_prefix(prefix_ids)
                mask_ids = torch.full(
                    (prefix_ids.shape[0], canvas_length),
                    tokenizer.mask_token_id,
                    dtype=torch.long,
                    device=device,
                )
                canvas_states = model.initialize_canvas(mask_ids)
                canvas_logits = None
                for theorizer_step in range(diffusion_steps):
                    canvas_states, canvas_logits = model.theorize_once(
                        prefix_states,
                        canvas_states,
                        theorizer_step % model.config.max_diffusion_steps,
                    )
                if canvas_logits is None:
                    raise RuntimeError("diagnostic theorizer produced no logits")

                variants = {
                    "full": canvas_states,
                    "no_canvas": torch.zeros_like(canvas_states),
                    "first_position": canvas_states[:, :1, :],
                    "pooled": canvas_states.mean(dim=1, keepdim=True),
                    "random": torch.randn(
                        canvas_states.shape,
                        device=device,
                        dtype=canvas_states.dtype,
                    )
                    * canvas_states.detach().float().std().clamp_min(1e-6).to(
                        canvas_states.dtype
                    ),
                }
                # Shuffle canvas content while keeping each destination slot
                # fixed. The states already contain learned position tags, so
                # shuffling complete states would preserve content-position
                # pairs and would not test whether slot order matters.
                canvas_positions = model.canvas_position.weight[:canvas_length]
                canvas_content = canvas_states - canvas_positions.view(
                    1, canvas_length, -1
                )
                permutation = torch.randperm(canvas_length, device=device)
                variants["shuffled"] = (
                    canvas_content[:, permutation, :]
                    + canvas_positions.view(1, canvas_length, -1)
                )
                logits_by_variant = {
                    name: model.solidify(prefix_states, states)
                    for name, states in variants.items()
                }

            full_logits = logits_by_variant["full"].float()
            full_log_probs = F.log_softmax(full_logits, dim=-1)
            full_probs = full_log_probs.exp()
            valid = targets != tokenizer.pad_token_id
            for name, logits in logits_by_variant.items():
                loss_sum, correct, count = _diagnostic_loss_and_accuracy(
                    logits.float(), targets, tokenizer.pad_token_id
                )
                totals[name]["loss_sum"] += loss_sum
                totals[name]["correct"] += correct
                totals[name]["count"] += count
                if name != "full":
                    variant_log_probs = F.log_softmax(logits.float(), dim=-1)
                    kl = (full_probs * (full_log_probs - variant_log_probs)).sum(dim=-1)
                    agreement = (
                        full_logits.argmax(dim=-1) == logits.float().argmax(dim=-1)
                    )
                    if bool(valid.any()):
                        comparisons[name]["kl_sum"] += float(kl[valid].sum().item())
                        comparisons[name]["agreement"] += float(agreement[valid].float().sum().item())
                        comparisons[name]["count"] += int(valid.sum().item())
            seen += int(prefix_ids.shape[0])

    report = {
        "checkpoint": args.checkpoint,
        "samples": seen,
        "prefix_length": prefix_length,
        "canvas_length": canvas_length,
        "diffusion_steps": diffusion_steps,
        "task": task,
        "variants": {},
    }
    for name in variant_names:
        count = totals[name]["count"]
        row = {
            "next_token_loss": totals[name]["loss_sum"] / max(count, 1),
            "next_token_accuracy": totals[name]["correct"] / max(count, 1),
        }
        if name != "full":
            comparison_count = comparisons[name]["count"]
            row.update(
                {
                    "kl_full_to_variant": comparisons[name]["kl_sum"]
                    / max(comparison_count, 1),
                    "top1_agreement_with_full": comparisons[name]["agreement"]
                    / max(comparison_count, 1),
                }
            )
        report["variants"][name] = row

    print(
        f"device={device} samples={seen} prefix={prefix_length} "
        f"canvas={canvas_length} diffusion_steps={diffusion_steps} task={task}"
    )
    print("variant       loss       accuracy   KL(full||variant)   top1_agreement")
    for name in variant_names:
        row = report["variants"][name]
        print(
            f"{name:<13} {row['next_token_loss']:.4f}     "
            f"{row['next_token_accuracy']:.4f}     "
            f"{row.get('kl_full_to_variant', 0.0):.4f}              "
            f"{row.get('top1_agreement_with_full', 1.0):.4f}"
        )
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


@torch.no_grad()
def rolling_diagnose(args) -> None:
    """Evaluate multi-token teacher-forced rolling canvas retention.

    The retained path follows the inference transition exactly: refine the
    current canvas, commit one target token, append that committed token to
    the prefix, shift the hidden canvas left by one slot, and initialize only
    the newly exposed tail. The reset control reinitializes a blank canvas at
    every commit, isolating the value of retaining previous canvas state.
    """

    device = choose_inference_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    saved_args = checkpoint.get("args", {})
    tokenizer = build_tokenizer(args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({tokenizer.vocab_size}) does not match checkpoint "
            f"vocab ({model_config.vocab_size})"
        )

    prefix_length = args.prefix_length or int(
        saved_args.get("prefix_length", model_config.max_prefix_length)
    )
    canvas_length = args.canvas_length or int(
        saved_args.get("canvas_length", model_config.max_canvas_length)
    )
    diffusion_steps = args.diffusion_steps or int(saved_args.get("diffusion_steps", 1))
    data_name = args.data or saved_args.get("data", "synthetic")
    synthetic_count = args.synthetic_examples or int(
        saved_args.get("synthetic_examples", 256)
    )
    texts = read_texts(data_name, synthetic_count)
    dataset = WindowDataset(
        texts,
        tokenizer,
        prefix_length,
        canvas_length,
        stride=args.window_stride,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = RFCAStudent(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.manual_seed(args.seed)

    commits = min(args.commits, canvas_length)
    if commits < 1:
        raise ValueError("--commits must be >= 1")
    if args.theorizer_passes < 1:
        raise ValueError("--theorizer-passes must be >= 1")

    metrics = {
        "retained": [
            {"loss_sum": 0.0, "correct": 0.0, "count": 0} for _ in range(commits)
        ],
        "reset": [
            {"loss_sum": 0.0, "correct": 0.0, "count": 0} for _ in range(commits)
        ],
    }
    comparisons = [
        {"kl_sum": 0.0, "agreement": 0.0, "count": 0}
        for _ in range(commits)
    ]
    seen = 0
    for batch in loader:
        if seen >= args.samples:
            break
        remaining = args.samples - seen
        if batch["prefix_ids"].shape[0] > remaining:
            batch = {key: value[:remaining] for key, value in batch.items()}
        prefix_ids = batch["prefix_ids"].to(device)
        targets = batch["targets"].to(device)
        batch_size = prefix_ids.shape[0]
        mask_ids = torch.full(
            (batch_size, canvas_length),
            tokenizer.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        retained_canvas = model.initialize_canvas(mask_ids)

        for commit_index in range(commits):
            with autocast_context(device, args.precision):
                prefix_states = model.encode_prefix(prefix_ids)
                for theorizer_step in range(args.theorizer_passes):
                    retained_canvas, _ = model.theorize_once(
                        prefix_states,
                        retained_canvas,
                        theorizer_step % model.config.max_diffusion_steps,
                    )

                reset_canvas = model.initialize_canvas(mask_ids)
                for theorizer_step in range(args.theorizer_passes):
                    reset_canvas, _ = model.theorize_once(
                        prefix_states,
                        reset_canvas,
                        theorizer_step % model.config.max_diffusion_steps,
                    )
                retained_logits = model.solidify(prefix_states, retained_canvas).float()
                reset_logits = model.solidify(prefix_states, reset_canvas).float()

            target = targets[:, commit_index]
            for name, logits in (
                ("retained", retained_logits),
                ("reset", reset_logits),
            ):
                loss_sum, correct, count = _diagnostic_loss_and_accuracy(
                    logits, target, tokenizer.pad_token_id
                )
                metrics[name][commit_index]["loss_sum"] += loss_sum
                metrics[name][commit_index]["correct"] += correct
                metrics[name][commit_index]["count"] += count

            valid = target != tokenizer.pad_token_id
            retained_log_probs = F.log_softmax(retained_logits, dim=-1)
            reset_log_probs = F.log_softmax(reset_logits, dim=-1)
            retained_probs = retained_log_probs.exp()
            kl = (retained_probs * (retained_log_probs - reset_log_probs)).sum(dim=-1)
            agreement = retained_logits.argmax(dim=-1) == reset_logits.argmax(dim=-1)
            if bool(valid.any()):
                comparisons[commit_index]["kl_sum"] += float(kl[valid].sum().item())
                comparisons[commit_index]["agreement"] += float(
                    agreement[valid].float().sum().item()
                )
                comparisons[commit_index]["count"] += int(valid.sum().item())

            # Teacher-force the correct commit so later positions measure
            # rolling-state behavior rather than compounding token mistakes.
            prefix_ids = append_committed_token(
                prefix_ids,
                target[:, None],
                model.config.max_prefix_length,
            )
            retained_canvas = model.roll_canvas(
                retained_canvas, tokenizer.mask_token_id
            )
        seen += batch_size

    report = {
        "checkpoint": args.checkpoint,
        "samples": seen,
        "prefix_length": prefix_length,
        "canvas_length": canvas_length,
        "diffusion_steps": diffusion_steps,
        "theorizer_passes": args.theorizer_passes,
        "commits": commits,
        "variants": {},
    }
    for name in ("retained", "reset"):
        report["variants"][name] = []
        for commit_index, row in enumerate(metrics[name]):
            count = row["count"]
            report["variants"][name].append(
                {
                    "commit": commit_index + 1,
                    "next_token_loss": row["loss_sum"] / max(count, 1),
                    "next_token_accuracy": row["correct"] / max(count, 1),
                }
            )

    report["retained_vs_reset"] = []
    for commit_index, row in enumerate(comparisons):
        count = row["count"]
        report["retained_vs_reset"].append(
            {
                "commit": commit_index + 1,
                "kl_retained_to_reset": row["kl_sum"] / max(count, 1),
                "top1_agreement": row["agreement"] / max(count, 1),
            }
        )

    print(
        f"device={device} samples={seen} prefix={prefix_length} "
        f"canvas={canvas_length} commits={commits} "
        f"theorizer_passes={args.theorizer_passes}"
    )
    print(
        "commit  retained_loss  retained_acc  reset_loss  reset_acc  "
        "KL(retained||reset)  top1_agreement"
    )
    for commit_index in range(commits):
        retained_row = report["variants"]["retained"][commit_index]
        reset_row = report["variants"]["reset"][commit_index]
        comparison = report["retained_vs_reset"][commit_index]
        print(
            f"{commit_index + 1:>6}  "
            f"{retained_row['next_token_loss']:.4f}        "
            f"{retained_row['next_token_accuracy']:.4f}       "
            f"{reset_row['next_token_loss']:.4f}     "
            f"{reset_row['next_token_accuracy']:.4f}     "
            f"{comparison['kl_retained_to_reset']:.4f}              "
            f"{comparison['top1_agreement']:.4f}"
        )
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    model_config: ModelConfig,
    args,
    tokenizer,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(
        {
            "step": step,
            "model": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": dataclasses.asdict(model_config),
            "args": vars(args),
        },
        output_dir / f"checkpoint-{step:07d}.pt",
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(dataclasses.asdict(model_config), indent=2) + "\n",
        encoding="utf-8",
    )
    tokenizer.save(output_dir)


def load_resume(
    resume_path: Optional[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> int:
    if not resume_path:
        return 0
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("step", 0))


def train(args) -> None:
    distributed, rank, world_size, device = init_distributed()
    set_seed(args.seed + rank)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = build_tokenizer(args.tokenizer)
    args.mask_id = tokenizer.mask_token_id
    if args.task in (
        "position_probe",
        "ordered_sequence_probe",
        "indexed_sequence_probe",
    ):
        probe_dataset = {
            "position_probe": PositionProbeDataset,
            "ordered_sequence_probe": OrderedSequenceProbeDataset,
            "indexed_sequence_probe": IndexedSequenceProbeDataset,
        }[args.task]
        train_dataset = probe_dataset(
            tokenizer,
            args.prefix_length,
            args.canvas_length,
            args.probe_examples,
            args.seed,
        )
        validation_dataset = probe_dataset(
            tokenizer,
            args.prefix_length,
            args.canvas_length,
            max(64, args.probe_examples // 8),
            args.seed + 1,
        )
    else:
        texts = read_texts(args.data, args.synthetic_examples)
        split = max(1, int(len(texts) * (1.0 - args.validation_fraction)))
        train_texts = texts[:split]
        validation_texts = texts[split:] or texts[: min(8, len(texts))]
        train_dataset = WindowDataset(
            train_texts,
            tokenizer,
            args.prefix_length,
            args.canvas_length,
            stride=args.window_stride,
        )
        validation_dataset = WindowDataset(
            validation_texts,
            tokenizer,
            args.prefix_length,
            args.canvas_length,
            stride=args.window_stride,
        )

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if distributed else None
    )
    validation_sampler = (
        DistributedSampler(validation_dataset, shuffle=False) if distributed else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        sampler=validation_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_config = build_model_config(args, tokenizer.vocab_size)
    model = RFCAStudent(
        model_config,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    if is_main_process(rank):
        print(
            f"device={device} world_size={world_size} "
            f"parameters={parameter_count(model):,} "
            f"preset={args.model_preset} vocab={tokenizer.vocab_size:,}"
        )
        print(
            f"train_windows={len(train_dataset):,} "
            f"validation_windows={len(validation_dataset):,} "
            f"task={args.task} "
            f"prefix={args.prefix_length} canvas={args.canvas_length} "
            f"diffusion_steps={args.diffusion_steps} "
            f"rolling_curriculum_prob={args.rolling_curriculum_prob:.2f} "
            f"rolling_curriculum_max_commits={args.rolling_curriculum_max_commits}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    total_updates = max(args.steps, 1)
    warmup_updates = min(args.warmup_steps, total_updates)

    def lr_lambda(update: int) -> float:
        if update < warmup_updates:
            return max(float(update + 1) / max(warmup_updates, 1), 1e-8)
        progress = (update - warmup_updates) / max(total_updates - warmup_updates, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = load_resume(args.resume, model, optimizer, scheduler, device)
    # Restore into the ordinary module before optional compilation/DDP
    # wrappers add state-dict prefixes.
    if args.compile and device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)

    scaler_enabled = args.precision == "fp16" and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # PyTorch < 2.4 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_args.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if distributed:
        dist.barrier()

    model.train()
    iterator = iter(train_loader)
    epoch = 0
    running = {"loss": 0.0, "diffusion_loss": 0.0, "solidifier_loss": 0.0}
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, args.steps):
        batch, iterator, epoch = next_batch(
            train_loader, iterator, train_sampler, epoch
        )
        prefix_ids = batch["prefix_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        solidifier_targets = batch.get("solidifier_targets")
        if solidifier_targets is not None:
            solidifier_targets = solidifier_targets.to(device, non_blocking=True)
        canvas_ids = torch.full_like(targets, args.mask_id)
        rolling_commits = 0
        if (
            args.task == "standard"
            and args.rolling_curriculum_prob > 0.0
            and random.random() < args.rolling_curriculum_prob
        ):
            max_commits = min(
                args.rolling_curriculum_max_commits,
                args.canvas_length - 1,
            )
            if max_commits >= 1:
                rolling_commits = random.randint(1, max_commits)

        with autocast_context(device, args.precision):
            if rolling_commits:
                outputs = forward_teacher_forced_rolling(
                    model,
                    prefix_ids,
                    targets,
                    args.mask_id,
                    args.diffusion_steps,
                    rolling_commits,
                )
                loss, metrics = compute_rolling_losses(
                    outputs,
                    targets,
                    tokenizer.pad_token_id,
                    args.diffusion_loss_weight,
                    args.solidifier_loss_weight,
                )
            else:
                outputs = model(prefix_ids, canvas_ids, args.diffusion_steps)
                loss, metrics = compute_losses(
                    outputs,
                    targets,
                    tokenizer.pad_token_id,
                    args.diffusion_loss_weight,
                    args.solidifier_loss_weight,
                    solidifier_targets=solidifier_targets,
                )
            scaled_loss = loss / args.gradient_accumulation

        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (step + 1) % args.gradient_accumulation == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        for key in running:
            running[key] += metrics[key]

        completed_step = step + 1
        if is_main_process(rank) and completed_step % args.log_every == 0:
            elapsed = time.perf_counter() - started
            denominator = args.log_every
            values = {key: value / denominator for key, value in running.items()}
            running = {key: 0.0 for key in running}
            print(
                f"step={completed_step}/{args.steps} "
                f"loss={values['loss']:.4f} "
                f"diff={values['diffusion_loss']:.4f} "
                f"solid={values['solidifier_loss']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} "
                f"steps/s={args.log_every / max(elapsed, 1e-6):.2f}"
            )
            started = time.perf_counter()

        if completed_step % args.eval_every == 0:
            metrics = evaluate(
                model,
                validation_loader,
                device,
                args,
                tokenizer.pad_token_id,
                args.eval_batches,
            )
            if is_main_process(rank):
                print(
                    f"eval step={completed_step} loss={metrics['loss']:.4f} "
                    f"diff={metrics['diffusion_loss']:.4f} "
                    f"solid={metrics['solidifier_loss']:.4f}"
                )
            model.train()

        if (
            completed_step % args.save_every == 0 or completed_step == args.steps
        ) and is_main_process(rank):
            save_checkpoint(
                output_dir,
                model,
                optimizer,
                scheduler,
                completed_step,
                model_config,
                args,
                tokenizer,
            )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="train the student")
    train_parser.add_argument(
        "--data",
        default="synthetic",
        help="text/JSONL path, or synthetic (default: synthetic)",
    )
    train_parser.add_argument(
        "--tokenizer",
        default="byte",
        help="byte for smoke tests, or a Hugging Face tokenizer/model id",
    )
    train_parser.add_argument(
        "--model-preset", choices=sorted(PRESETS), default="smoke"
    )
    train_parser.add_argument(
        "--task",
        choices=(
            "standard",
            "position_probe",
            "ordered_sequence_probe",
            "indexed_sequence_probe",
        ),
        default="standard",
        help="standard text windows or a synthetic canvas-interface probe",
    )
    train_parser.add_argument("--output-dir", default="runs/rfca-student")
    train_parser.add_argument("--prefix-length", type=int, default=128)
    train_parser.add_argument(
        "--prefix-layers",
        type=int,
        default=None,
        help="override causal prefix depth for a matched-capacity control",
    )
    train_parser.add_argument("--canvas-length", type=int, default=32)
    train_parser.add_argument(
        "--canvas-layers",
        type=int,
        default=None,
        help="override canvas denoiser depth; use 0 for the causal-only control",
    )
    train_parser.add_argument(
        "--solidifier-layers",
        type=int,
        default=0,
        help="add dense causal blocks after the canvas read; use for a larger solidifier",
    )
    train_parser.add_argument(
        "--rolling-curriculum-prob",
        type=float,
        default=0.0,
        help=(
            "probability of using a teacher-forced retained-canvas trajectory "
            "on each batch; standard task only"
        ),
    )
    train_parser.add_argument(
        "--rolling-curriculum-max-commits",
        type=int,
        default=4,
        help="maximum number of teacher-forced commits before the trained loss",
    )
    train_parser.add_argument("--diffusion-steps", type=int, default=2)
    train_parser.add_argument("--max-diffusion-steps", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--gradient-accumulation", type=int, default=1)
    train_parser.add_argument("--steps", type=int, default=1000)
    train_parser.add_argument("--warmup-steps", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--beta1", type=float, default=0.9)
    train_parser.add_argument("--beta2", type=float, default=0.95)
    train_parser.add_argument("--weight-decay", type=float, default=0.1)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--diffusion-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--solidifier-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--validation-fraction", type=float, default=0.1)
    train_parser.add_argument("--window-stride", type=int, default=None)
    train_parser.add_argument("--synthetic-examples", type=int, default=256)
    train_parser.add_argument("--probe-examples", type=int, default=2048)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--eval-every", type=int, default=100)
    train_parser.add_argument("--eval-batches", type=int, default=10)
    train_parser.add_argument("--save-every", type=int, default=500)
    train_parser.add_argument("--seed", type=int, default=2027)
    train_parser.add_argument("--resume", default=None)
    train_parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    train_parser.add_argument("--gradient-checkpointing", action="store_true")
    train_parser.add_argument("--compile", action="store_true")
    generate_parser = subparsers.add_parser(
        "generate", help="test the rolling theorizer/solidifier loop"
    )
    generate_parser.add_argument("--checkpoint", required=True)
    generate_parser.add_argument(
        "--tokenizer",
        default="byte",
        help="must match the tokenizer used to create the checkpoint",
    )
    generate_parser.add_argument("--prompt", default="")
    generate_parser.add_argument("--device", default="auto")
    generate_parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    generate_parser.add_argument("--max-new-tokens", type=int, default=32)
    generate_parser.add_argument("--canvas-length", type=int, default=None)
    generate_parser.add_argument("--max-theorizer-steps", type=int, default=8)
    generate_parser.add_argument(
        "--theorizer-stop",
        choices=("confidence", "entropy", "normalized_entropy"),
        default="entropy",
        help="confidence is mean max probability; entropy is in nats",
    )
    generate_parser.add_argument(
        "--theorizer-confidence-threshold",
        type=float,
        default=0.50,
        help="stop when mean per-position max probability reaches this value",
    )
    generate_parser.add_argument(
        "--theorizer-entropy-threshold",
        type=float,
        default=0.50,
        help="stop when mean entropy reaches this value (lower is more certain)",
    )
    generate_parser.add_argument("--temperature", type=float, default=0.0)
    generate_parser.add_argument(
        "--no-boundary-interpolation",
        action="store_true",
        help="allow a denoising pass to overshoot the requested uncertainty",
    )
    generate_parser.add_argument("--trace-output", default=None)
    rolling_parser = subparsers.add_parser(
        "rolling_diagnose",
        help="measure multi-token retention of the rolling hidden canvas",
    )
    rolling_parser.add_argument("--checkpoint", required=True)
    rolling_parser.add_argument(
        "--tokenizer",
        default="byte",
        help="must match the tokenizer used to create the checkpoint",
    )
    rolling_parser.add_argument("--data", default=None)
    rolling_parser.add_argument("--device", default="auto")
    rolling_parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    rolling_parser.add_argument("--samples", type=int, default=128)
    rolling_parser.add_argument("--batch-size", type=int, default=8)
    rolling_parser.add_argument("--prefix-length", type=int, default=None)
    rolling_parser.add_argument("--canvas-length", type=int, default=None)
    rolling_parser.add_argument("--diffusion-steps", type=int, default=None)
    rolling_parser.add_argument("--window-stride", type=int, default=None)
    rolling_parser.add_argument("--synthetic-examples", type=int, default=None)
    rolling_parser.add_argument("--commits", type=int, default=8)
    rolling_parser.add_argument("--theorizer-passes", type=int, default=2)
    rolling_parser.add_argument("--seed", type=int, default=2027)
    rolling_parser.add_argument("--output", default=None)
    diagnose_parser = subparsers.add_parser(
        "diagnose", help="measure solidifier sensitivity to the theorizer canvas"
    )
    diagnose_parser.add_argument("--checkpoint", required=True)
    diagnose_parser.add_argument(
        "--tokenizer",
        default="byte",
        help="must match the tokenizer used to create the checkpoint",
    )
    diagnose_parser.add_argument("--data", default=None)
    diagnose_parser.add_argument(
        "--task",
        choices=(
            "standard",
            "position_probe",
            "ordered_sequence_probe",
            "indexed_sequence_probe",
        ),
        default=None,
    )
    diagnose_parser.add_argument("--device", default="auto")
    diagnose_parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    diagnose_parser.add_argument("--samples", type=int, default=128)
    diagnose_parser.add_argument("--batch-size", type=int, default=8)
    diagnose_parser.add_argument("--prefix-length", type=int, default=None)
    diagnose_parser.add_argument("--canvas-length", type=int, default=None)
    diagnose_parser.add_argument("--diffusion-steps", type=int, default=None)
    diagnose_parser.add_argument("--window-stride", type=int, default=None)
    diagnose_parser.add_argument("--synthetic-examples", type=int, default=None)
    diagnose_parser.add_argument("--probe-examples", type=int, default=None)
    diagnose_parser.add_argument("--seed", type=int, default=2027)
    diagnose_parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        if args.max_diffusion_steps is not None:
            args.diffusion_steps = min(args.diffusion_steps, args.max_diffusion_steps)
        if args.diffusion_steps < 1:
            parser.error("--diffusion-steps must be >= 1")
        if args.prefix_length < 2 or args.canvas_length < 1:
            parser.error("prefix/canvas lengths are too small")
        if args.canvas_layers is not None and args.canvas_layers < 0:
            parser.error("--canvas-layers must be >= 0")
        if args.prefix_layers is not None and args.prefix_layers < 0:
            parser.error("--prefix-layers must be >= 0")
        if args.solidifier_layers < 0:
            parser.error("--solidifier-layers must be >= 0")
        if not 0.0 <= args.rolling_curriculum_prob <= 1.0:
            parser.error("--rolling-curriculum-prob must be between 0 and 1")
        if args.rolling_curriculum_max_commits < 1:
            parser.error("--rolling-curriculum-max-commits must be >= 1")
        if args.rolling_curriculum_prob > 0.0 and args.task != "standard":
            parser.error("rolling curriculum currently supports --task standard only")
        train(args)
    elif args.command == "generate":
        if args.max_new_tokens < 1 or args.max_theorizer_steps < 1:
            parser.error("generation lengths must be >= 1")
        if not 0.0 <= args.theorizer_confidence_threshold <= 1.0:
            parser.error("confidence threshold must be between 0 and 1")
        if args.theorizer_entropy_threshold <= 0.0:
            parser.error("entropy threshold must be > 0")
        if args.temperature < 0.0:
            parser.error("temperature must be >= 0")
        generate(args)
    elif args.command == "diagnose":
        if args.samples < 1 or args.batch_size < 1:
            parser.error("--samples and --batch-size must be >= 1")
        diagnose(args)
    elif args.command == "rolling_diagnose":
        if args.samples < 1 or args.batch_size < 1:
            parser.error("--samples and --batch-size must be >= 1")
        if args.commits < 1 or args.theorizer_passes < 1:
            parser.error("--commits and --theorizer-passes must be >= 1")
        rolling_diagnose(args)


if __name__ == "__main__":
    main()
