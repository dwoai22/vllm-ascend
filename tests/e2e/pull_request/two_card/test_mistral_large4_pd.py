# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Mistral Large 4 prefill/decode disaggregation over MooncakeConnectorV2.

One prefill node and one decode node, each on its own NPU, running the reduced
dummy-weight model of the one-card test. That model keeps a layer that reads
another layer's latent KV and layers that reuse a neighbour's top-k, which are
the two things a KV transfer has to get right for this architecture: the
borrowing layer owns no blocks to move, and it still owns an indexer cache
that must arrive.

The reference is the decode node's own run of the same prompt, not the prefill
node's. Both nodes prefill in chunks that leave the last prompt token on its
own, so a decode that starts from transferred KV recomputes exactly the step
the local run ends its prefill with, from the same cached keys. The two then
take the same path over the same inputs and must agree token for token; a
single-pass prefill would only agree up to the rounding of a different batch
shape, which dummy weights turn into different tokens.
"""

import json

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemotePDServer, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.test_mistral_large4 import (
    CONFIG,
    INDEXER_LAYERS,
    OUTPUT_TOKENS,
    SHARING_LAYERS,
    VOCAB_SIZE,
    _write_model,
)
from vllm_ascend.models import _upstream_has_module

pytestmark = pytest.mark.skipif(
    not _upstream_has_module("vllm.models.mistral.mistral_large4"),
    reason="vLLM without Mistral Large 4 (vllm-project/vllm#64534)",
)

SERVED_MODEL_NAME = "mistral-large4-pd"
# Prefill chunk. Prompts are one token longer than a whole number of chunks, so
# the last chunk is the single token a decode node recomputes after a transfer.
PREFILL_CHUNK_TOKENS = 64
PROMPT_LENGTHS = (PREFILL_CHUNK_TOKENS + 1, 3 * PREFILL_CHUNK_TOKENS + 1)
# Each node takes kv_port for its worker handshake and the next port for its
# scheduler; keep the two nodes' ranges apart.
PREFILL_KV_PORT = 27170
DECODE_KV_PORT = 27370
REQUEST_TIMEOUT_SECONDS = 180


def _server_args(model: str, port: int, kv_role: str, kv_port: int) -> list[str]:
    return [
        model,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--load-format",
        "dummy",
        "--skip-tokenizer-init",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        str(PREFILL_CHUNK_TOKENS),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.5",
        "--seed",
        "0",
        "--kv-transfer-config",
        json.dumps({"kv_connector": "MooncakeConnectorV2", "kv_role": kv_role, "kv_port": kv_port}),
    ]


def _prompt(length: int, offset: int) -> list[int]:
    return [(7 * i + offset) % VOCAB_SIZE for i in range(length)]


def _complete(url: str, prompt: list[int], *, max_tokens: int = OUTPUT_TOKENS, **kwargs) -> dict:
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": SERVED_MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
            **kwargs,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    output = response.json()
    token_ids = output["choices"][0]["token_ids"]
    assert len(token_ids) == max_tokens
    assert all(0 <= token < VOCAB_SIZE for token in token_ids)
    return output


@wait_until_npu_memory_free()
def test_mistral_large4_decodes_from_transferred_kv(tmp_path):
    assert SHARING_LAYERS, "the reduced config must keep a KV-sharing layer"
    assert set(SHARING_LAYERS) & set(INDEXER_LAYERS), "a KV-sharing layer must still own an indexer cache"
    assert len(INDEXER_LAYERS) < CONFIG["num_hidden_layers"], "the reduced config must keep a shared-index layer"

    model = tmp_path / "model"
    _write_model(model)
    prefill_port, decode_port = get_open_port(), get_open_port()
    prefill_url, decode_url = f"http://127.0.0.1:{prefill_port}", f"http://127.0.0.1:{decode_port}"
    servers = [
        _server_args(str(model), prefill_port, "kv_producer", PREFILL_KV_PORT),
        _server_args(str(model), decode_port, "kv_consumer", DECODE_KV_PORT),
    ]
    with RemotePDServer(servers):
        for offset, length in enumerate(PROMPT_LENGTHS, start=1):
            prompt = _prompt(length, offset)
            # Transfer first: the decode node has never seen this prompt, so
            # nothing but the pulled blocks can hold its KV. A failed pull
            # fails the request, which is the connector's default policy.
            prefill = _complete(prefill_url, prompt, max_tokens=1, kv_transfer_params={"do_remote_decode": True})
            transfer = prefill["kv_transfer_params"]
            assert transfer["do_remote_prefill"]
            assert any(transfer["remote_block_ids"])
            decoded = _complete(decode_url, prompt, kv_transfer_params=transfer)

            reference = _complete(decode_url, prompt)
            assert decoded["choices"][0]["token_ids"] == reference["choices"][0]["token_ids"], (
                f"decode from transferred KV diverged from a local run for the {length}-token prompt"
            )
