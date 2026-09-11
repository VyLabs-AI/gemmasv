"""Real-corpus data + held-out perplexity for the SV-graft distillation.

Respectable, standard sources (streamed -- nothing fills the 512 GB):
  * TRAIN: FineWeb-Edu (``HuggingFaceFW/fineweb-edu``, ``sample-10BT``) -- the current
    high-quality web-text standard, as used for LoLCATs-style distillation.
  * EVAL : WikiText-103 (``Salesforce/wikitext``, ``wikitext-103-raw-v1``) test -- the
    canonical LM perplexity benchmark, for quality-parity (original vs grafted+recovered).
Both tokenized with the model's own (Gemma) tokenizer and packed into fixed-length blocks.
"""
from __future__ import annotations

import itertools
import math
from typing import Iterable, Iterator, List

import torch

from gemma_sv.recovery_protocol import FINEWEB_REVISION, WIKITEXT_REVISION

FINEWEB = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
WIKITEXT = ("Salesforce/wikitext", "wikitext-103-raw-v1")


def packed_block_stream(texts: Iterable[str], tok, seq_len: int,
                        add_bos: bool = True) -> Iterator[List[int]]:
    """Tokenize a stream of documents and yield contiguous ``seq_len`` token blocks
    (packed, GPT-style; each doc prefixed with BOS). No padding -- every block is full."""
    bos = [tok.bos_token_id] if (add_bos and tok.bos_token_id is not None) else []
    buf: List[int] = []
    for t in texts:
        if not t or not t.strip():
            continue
        buf.extend(bos + tok(t, add_special_tokens=False).input_ids)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]


def fineweb_edu_batch_iter(
    tok,
    seq_len: int,
    batch: int,
    *,
    device: str = "cpu",
    seed: int = 0,
    revision: str | None = FINEWEB_REVISION,
) -> Iterator[torch.Tensor]:
    """Endlessly stream FineWeb-Edu, pack, and yield (batch, seq_len) id tensors on the
    fly -- overlaps streaming/tokenisation with compute for long runs."""
    from datasets import load_dataset

    ds = load_dataset(
        *FINEWEB,
        split="train",
        streaming=True,
        revision=revision,
    ).shuffle(seed=seed, buffer_size=10000)
    blocks = packed_block_stream((ex["text"] for ex in ds), tok, seq_len)
    while True:
        rows = list(itertools.islice(blocks, batch))
        if len(rows) < batch:
            return
        yield torch.tensor(rows, dtype=torch.long, device=device)


def fineweb_edu_batches(
    tok,
    seq_len: int,
    batch: int,
    n_steps: int,
    *,
    device: str = "cpu",
    seed: int = 0,
    revision: str | None = FINEWEB_REVISION,
) -> List[torch.Tensor]:
    """Materialise ``n_steps`` FineWeb-Edu (batch, seq_len) id tensors (small runs/tests)."""
    it = fineweb_edu_batch_iter(
        tok,
        seq_len,
        batch,
        device=device,
        seed=seed,
        revision=revision,
    )
    return list(itertools.islice(it, n_steps))


def wikitext103_blocks(
    tok,
    seq_len: int,
    *,
    max_blocks: int | None = None,
    split: str = "test",
    revision: str | None = WIKITEXT_REVISION,
) -> List[List[int]]:
    """WikiText-103 ``split`` packed into ``seq_len`` token blocks for perplexity."""
    from datasets import load_dataset

    ds = load_dataset(*WIKITEXT, split=split, revision=revision)
    blocks = list(packed_block_stream((t for t in ds["text"]), tok, seq_len))
    return blocks[:max_blocks] if max_blocks else blocks


# Second-corpus perplexity panel (eval-only parity hardening): different domains from the
# FineWeb-Edu training -- Lambada (literary/narrative), C4 (web) -- plus WikiText (wikipedia).
CORPORA = {
    "wikitext": (("Salesforce/wikitext", "wikitext-103-raw-v1"), "test", False),
    "lambada": (("EleutherAI/lambada_openai",), "test", True),
    "c4": (("allenai/c4", "en"), "validation", True),
}
CORPUS_REVISIONS = {
    "wikitext": WIKITEXT_REVISION,
    "lambada": None,
    "c4": None,
}


def corpus_blocks(
    tok,
    key: str,
    seq_len: int,
    *,
    max_blocks: int | None = None,
    revision: str | None = None,
) -> List[List[int]]:
    """Packed seq_len token blocks from a named corpus (streamed for lambada/c4)."""
    from datasets import load_dataset

    (path, *cfg), split, streaming = CORPORA[key]
    resolved_revision = revision if revision is not None else CORPUS_REVISIONS[key]
    ds = load_dataset(
        path,
        *cfg,
        split=split,
        streaming=streaming,
        revision=resolved_revision,
    )
    texts = (ex["text"] for ex in ds) if streaming else (t for t in ds["text"])
    out: List[List[int]] = []
    for b in packed_block_stream(texts, tok, seq_len):
        out.append(b)
        if max_blocks and len(out) >= max_blocks:
            break
    return out


@torch.no_grad()
def perplexity(model, blocks: List[List[int]], *, device: str = "cpu",
               batch: int = 1) -> float:
    """Token-level held-out perplexity over packed blocks (causal LM loss, exp of the
    token-weighted mean). Restores the model's train/eval mode on exit."""
    was_training = model.training
    model.eval()
    total_nll, n_tok = 0.0, 0
    for i in range(0, len(blocks), batch):
        rows = blocks[i:i + batch]
        ids = torch.tensor(rows, dtype=torch.long, device=device)
        loss = model(input_ids=ids, labels=ids).loss        # mean CE over B*(T-1) tokens
        t = ids.shape[0] * (ids.shape[1] - 1)
        total_nll += float(loss) * t
        n_tok += t
    if was_training:
        model.train()
    return math.exp(total_nll / max(n_tok, 1))
