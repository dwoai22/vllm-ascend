#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a running Mistral Large 4 1P1D deployment.

Two checks, both against live servers (standard library only):

1. **The KV really came from P.** D's ``vllm:prompt_tokens_by_source`` counter
   is scraped before and after the PD requests. With the transfer working,
   nearly every prompt token lands under ``external_kv_transfer``; a D that
   recomputed the prompt locally shows them under ``local_compute``. This holds
   for dummy weights too.

2. **D decodes what P would.** The same greedy request is sent once through the
   proxy (P prefills, D decodes from the transferred KV) and once straight to P
   (no ``kv_transfer_params``, so P serves it alone). Only meaningful with real
   weights (``shrink_ml4_config.py --truncate-only`` + ``LOAD_FORMAT=mistral``):
   vLLM's dummy weights are all ~1e-3, so the dummy model's logits barely
   depend on the KV cache and a corrupted transfer would still "match". Pass
   ``--expect-match`` to make it a pass/fail check.

The long prompt is longer than the model's ``index_topk`` (2048), so D's sparse
attention really selects from the transferred indexer cache rather than
attending to every token.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any

METRIC = "vllm:prompt_tokens_by_source"
SOURCE_LABEL = 'source="'
# Fraction of prompt tokens that must arrive by transfer: D recomputes at most
# the last token, so anything well below 1.0 means a local recompute.
MIN_TRANSFER_FRACTION = 0.9
LONG_PROMPT_LINES = 600  # ~10k tokens: well past index_topk = 2048


def http_json(url: str, body: dict | None = None, timeout: float = 600) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def prompt_tokens_by_source(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=60) as response:
        text = response.read().decode()
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith(METRIC) or line.startswith(f"{METRIC}_created"):
            continue
        # One Prometheus sample per line:
        #   vllm:prompt_tokens_by_source{...,source="local_compute"} 1234.0
        start = line.find(SOURCE_LABEL)
        if start < 0:
            continue
        start += len(SOURCE_LABEL)
        end = line.find('"', start)
        if end < 0:
            continue
        source = line[start:end]
        totals[source] = totals.get(source, 0.0) + float(line.rsplit(" ", 1)[1])
    if not totals:
        raise RuntimeError(f"{base_url}/metrics has no {METRIC}; is this the Decode engine?")
    return totals


def complete(base_url: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "logprobs": 1,
    }
    response = http_json(f"{base_url}/v1/completions", body)
    choice = response["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return {
        "text": choice["text"],
        "tokens": logprobs.get("tokens") or [],
        "logprobs": logprobs.get("token_logprobs") or [],
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "finish_reason": choice.get("finish_reason"),
    }


def build_prompts() -> list[tuple[str, str]]:
    short = "Write a short story about a lighthouse keeper who finds a message in a bottle. " * 8
    lines = [
        f"Record {i}: sensor {i % 37} reported value {(i * 7919) % 1000} at step {i}." for i in range(LONG_PROMPT_LINES)
    ]
    long = "\n".join(lines) + "\nSummarize the records above."
    return [("short", short), ("long", long)]


def compare(reference: dict[str, Any], disaggregated: dict[str, Any]) -> tuple[int, float]:
    """Tokens matched before the first divergence, and the largest logprob gap there."""
    matched, max_gap = 0, 0.0
    for ref_token, pd_token, ref_lp, pd_lp in zip(
        reference["tokens"], disaggregated["tokens"], reference["logprobs"], disaggregated["logprobs"]
    ):
        if ref_token != pd_token:
            break
        matched += 1
        if ref_lp is not None and pd_lp is not None:
            max_gap = max(max_gap, abs(ref_lp - pd_lp))
    return matched, max_gap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proxy", required=True, help="Proxy base URL, e.g. http://P_IP:9000")
    parser.add_argument("--prefill", required=True, help="Prefill engine base URL, e.g. http://P_IP:8000")
    parser.add_argument("--decode", required=True, help="Decode engine base URL, e.g. http://D_IP:8000")
    parser.add_argument("--model", default="ml4-mini")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--expect-match", action="store_true", help="Fail unless PD matches P-only (real weights).")
    parser.add_argument("--min-match", type=int, default=8, help="Leading tokens that must match with --expect-match.")
    args = parser.parse_args()

    before = prompt_tokens_by_source(args.decode)
    failures: list[str] = []
    total_prompt_tokens = 0
    for name, prompt in build_prompts():
        reference = complete(args.prefill, args.model, prompt, args.max_tokens)
        disaggregated = complete(args.proxy, args.model, prompt, args.max_tokens)
        total_prompt_tokens += disaggregated["prompt_tokens"]
        matched, gap = compare(reference, disaggregated)
        print(
            f"[{name}] prompt_tokens={disaggregated['prompt_tokens']} "
            f"pd_tokens={len(disaggregated['tokens'])} finish={disaggregated['finish_reason']} "
            f"matched_prefix={matched}/{args.max_tokens} max|dlogprob| on prefix={gap:.4f}"
        )
        print(f"    P only: {reference['text'][:120]!r}")
        print(f"    P -> D: {disaggregated['text'][:120]!r}")
        if len(disaggregated["tokens"]) != args.max_tokens:
            failures.append(f"{name}: PD returned {len(disaggregated['tokens'])} tokens, expected {args.max_tokens}")
        if args.expect_match and matched < min(args.min_match, args.max_tokens):
            failures.append(f"{name}: PD diverged from P-only after {matched} tokens")

    after = prompt_tokens_by_source(args.decode)
    delta = {source: after.get(source, 0.0) - before.get(source, 0.0) for source in after}
    transferred = delta.get("external_kv_transfer", 0.0)
    print(f"D prompt tokens by source (delta): {json.dumps(delta)}; PD prompt tokens sent: {total_prompt_tokens}")
    if transferred < MIN_TRANSFER_FRACTION * total_prompt_tokens:
        failures.append(
            f"only {transferred:.0f} of {total_prompt_tokens} prompt tokens reached D by KV transfer; "
            "D recomputed the rest (or another client used D meanwhile)"
        )
    if not args.expect_match:
        print("note: token match not enforced; with dummy weights it does not reflect KV correctness.")

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
