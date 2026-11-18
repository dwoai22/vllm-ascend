#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correctness check for a KV-sharing model over MooncakeConnectorV2 1P1D.

For every prompt this compares two routes on the same pair of engines:

  baseline  the prefill engine generates the whole completion by itself
  pd        the prefill engine emits 1 token and hands its KV to the decode
            engine, which generates the rest

Greedy decoding makes the two byte-identical when the transfer is complete. If
the KV of a *borrower* layer is missing or stale on the decode side, the two
diverge -- usually within the first few tokens -- so a mismatch here is the
signal that the owner/borrower split is wrong, not a flaky run.

    python verify_pd.py --prefill-url http://127.0.0.1:13700 \
                        --decode-url  http://127.0.0.1:13701

Takes no NPU of its own; run it once both engines report healthy.
"""

from __future__ import annotations

import argparse
import sys

import requests

# Long enough that the prefill spans several blocks in both the full-attention
# and the sliding-window group, so borrowers in each group are covered.
PROMPTS = [
    "The capital of France is",
    "This is background information. " * 200 + "The capital of France is",
    "Count from one to twenty, in words, separated by commas:",
]


def complete(url: str, model: str, prompt: str, **kwargs) -> dict:
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 64,
            "ignore_eos": True,
            **kwargs,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def texts(result: dict) -> list[str]:
    return [choice["text"] for choice in result["choices"]]


def served_model_name(url: str) -> str:
    response = requests.get(f"{url}/v1/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefill-url", default="http://127.0.0.1:13700")
    parser.add_argument("--decode-url", default="http://127.0.0.1:13701")
    parser.add_argument("--model", default=None, help="Served model name (default: ask the prefill engine)")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    model = args.model or served_model_name(args.prefill_url)
    print(f"model: {model}\nprefill: {args.prefill_url}\ndecode:  {args.decode_url}\n")

    failures = 0
    for index, prompt in enumerate(PROMPTS):
        label = f"prompt {index} ({len(prompt)} chars)"

        baseline = texts(complete(args.prefill_url, model, prompt, max_tokens=args.max_tokens))

        # Drop anything the baseline left cached, so the PD route really
        # recomputes the prefill and really transfers it.
        try:
            requests.post(f"{args.prefill_url}/reset_prefix_cache", timeout=60).raise_for_status()
        except requests.RequestException:
            pass  # endpoint needs VLLM_SERVER_DEV_MODE=1; harmless without prefix caching

        prefill = complete(
            args.prefill_url,
            model,
            prompt,
            max_tokens=1,
            min_tokens=1,
            kv_transfer_params={"do_remote_decode": True},
        )
        transfer_params = prefill.get("kv_transfer_params")
        if not transfer_params:
            print(f"FAIL {label}: the prefill engine returned no kv_transfer_params")
            failures += 1
            continue

        pd = texts(
            complete(
                args.decode_url,
                model,
                prompt,
                max_tokens=args.max_tokens,
                kv_transfer_params=transfer_params,
            )
        )

        if pd == baseline:
            print(f"PASS {label}: {baseline[0][:60]!r}")
        else:
            failures += 1
            print(f"FAIL {label}")
            print(f"  baseline: {baseline[0][:200]!r}")
            print(f"  pd      : {pd[0][:200]!r}")

    print()
    if failures:
        print(f"{failures}/{len(PROMPTS)} prompt(s) diverged between the baseline and the PD route.")
        return 1
    print(f"All {len(PROMPTS)} prompts matched the baseline over the PD route.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.RequestException as exc:
        sys.exit(f"HTTP error talking to an engine: {exc}")
