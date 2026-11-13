# SPDX-License-Identifier: Apache-2.0
"""A KV-sharing sparse-MLA layout through the pull connector, end to end.

The layout is Mistral Large 4's, reduced: latent-KV owners, one layer that
reads another layer's latent KV yet owns an indexer cache, one layer that
reuses a neighbour's top-k and so owns no indexer cache, and a dense draft
layer in a cache group of its own.

Registration on both peers, the producer's block selection and metadata merge,
and the consumer's layer matching and address generation all run for real. Only
the transfer engine is replaced, by one that copies the byte ranges it is given,
so the assertions are on what lands in the consumer's caches.
"""

import ctypes
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker, pull_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import (
    MooncakePullConnectorScheduler,
    MooncakeSchedulerSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread

from .helpers import make_kv_cache_tensor, make_sfa_indexer_spec

BLOCK_SIZE = 16
KV_LORA_RANK = 6
ROPE_DIM = 2
INDEX_HEAD_DIM = 8
PROMPT_TOKENS = 40  # Two full blocks and a partial third.
SENTINEL = -1

# Layer i reads the latent KV of layer KV_SOURCE_LAYERS[i].
KV_SOURCE_LAYERS = [0, 1, 2, 1]
# "shared" layers reuse the preceding indexer's top-k and cache no keys.
INDEX_SHARING = ["full", "full", "shared", "full"]
NUM_TARGET_LAYERS = len(KV_SOURCE_LAYERS)


def attn_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.attn"


def indexer_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.indexer.k_cache"


OWNERS = [attn_name(i) for i, source in enumerate(KV_SOURCE_LAYERS) if source == i]
SHARING_TARGETS = {attn_name(i): attn_name(source) for i, source in enumerate(KV_SOURCE_LAYERS) if source != i}
INDEXERS = [indexer_name(i) for i, kind in enumerate(INDEX_SHARING) if kind == "full"]
DRAFT = f"draft_model.model.layers.{NUM_TARGET_LAYERS}.self_attn.attn"
TARGET_GROUP, DRAFT_GROUP = 0, 1


def _latent_spec() -> AscendMLAAttentionSpec:
    return AscendMLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=KV_LORA_RANK + ROPE_DIM,
        dtype=torch.bfloat16,
    )


def _make_caches(num_blocks: int, *, fill: bool) -> dict[str, object]:
    """Allocate one peer's caches the way the runner binds them."""

    def tensor(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        if not fill:
            return torch.full((num_blocks, *shape), SENTINEL, dtype=dtype)
        # Small integers are exact in bf16/fp16, so equality checks are exact.
        return torch.randint(0, 64, (num_blocks, *shape)).to(dtype)

    caches: dict[str, object] = {}
    for name in OWNERS:
        # Ascend binds the latent cache as separate NoPE and RoPE views.
        caches[name] = (
            tensor(BLOCK_SIZE, 1, KV_LORA_RANK, dtype=torch.bfloat16),
            tensor(BLOCK_SIZE, 1, ROPE_DIM, dtype=torch.bfloat16),
        )
    for name in INDEXERS:
        caches[name] = (tensor(BLOCK_SIZE, 1, INDEX_HEAD_DIM, dtype=torch.float16),)
    caches[DRAFT] = tensor(BLOCK_SIZE, KV_LORA_RANK + ROPE_DIM, dtype=torch.bfloat16)
    return caches


def _cache_tensors(cache: object) -> tuple[torch.Tensor, ...]:
    return cache if isinstance(cache, tuple) else (cache,)  # type: ignore[return-value]


def _make_config(num_blocks: int, caches: dict[str, object]) -> KVCacheConfig:
    """The scheduler's view: owners only, KV-sharing layers appear nowhere."""
    latent = _latent_spec()
    indexer = make_sfa_indexer_spec(block_size=BLOCK_SIZE, replication_size=1)
    target_specs = {**{name: latent for name in OWNERS}, **{name: indexer for name in INDEXERS}}
    tensors = []
    for name, cache in caches.items():
        size = sum(view.nbytes for view in _cache_tensors(cache))
        tensors.append(
            make_kv_cache_tensor(size=size, layers=[name], layer_stride=size, block_stride=size // num_blocks)
        )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=list(target_specs),
                kv_cache_spec=UniformTypeKVCacheSpecs(block_size=BLOCK_SIZE, kv_cache_specs=target_specs),
            ),
            KVCacheGroupSpec(layer_names=[DRAFT], kv_cache_spec=latent),
        ],
    )


def _make_vllm_config() -> SimpleNamespace:
    layers = {name: SimpleNamespace(kv_sharing_target_layer_name=None) for name in [*OWNERS, *INDEXERS, DRAFT]}
    layers.update(
        {name: SimpleNamespace(kv_sharing_target_layer_name=target) for name, target in SHARING_TARGETS.items()}
    )
    return SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=layers),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        model_config=SimpleNamespace(is_deepseek_mla=True, hf_text_config=SimpleNamespace()),
        speculative_config=None,
    )


def _register(engine_id: str, num_blocks: int, caches: dict[str, object]) -> MooncakeBaseConnectorWorker:
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config = _make_vllm_config()
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 1
    worker.kv_cache_config = _make_config(num_blocks, caches)
    worker.engine_id, worker.te_rpc_port = engine_id, 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = BLOCK_SIZE, "127.0.0.1", 5000
    # The runner binds every KV-sharing layer to its source before registering.
    bound = dict(caches)
    for name, target in SHARING_TARGETS.items():
        bound[name] = bound[target]
    worker.register_kv_caches(bound)
    return worker


class _CopyingTransferEngine:
    """Stands in for a Mooncake READ: copy each remote range into its local one."""

    def __init__(self) -> None:
        self.num_bytes = 0
        self.num_reads = 0
        self._lock = threading.Lock()

    def batch_transfer_sync_read(
        self, session_id: str, local_addrs: list[int], remote_addrs: list[int], lengths: list[int]
    ) -> int:
        for local_addr, remote_addr, length in zip(local_addrs, remote_addrs, lengths):
            ctypes.memmove(local_addr, remote_addr, length)
        with self._lock:
            self.num_bytes += sum(lengths)
            self.num_reads += 1
        return 0


def _make_consumer_thread(worker: MooncakeBaseConnectorWorker, engine: object) -> MooncakePullRecvingThread:
    # The arguments MooncakePullConnectorWorker.register_kv_caches passes.
    return MooncakePullRecvingThread(
        engine=engine,
        vllm_config=worker.vllm_config,
        kv_cache_config=worker.kv_cache_config,
        kv_cache_specs=worker.kv_cache_specs,
        layer_name_to_group_index=worker.layer_name_to_group_index,
        layer_name_to_spec_index=worker.layer_name_to_spec_index,
        local_metadata=worker.xfer_handshake_metadata,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_rank=0,
        dp_size=1,
        pcp_rank=0,
        pcp_size=1,
        dcp_rank=0,
        dcp_size=1,
        device=0,
        ready_event=threading.Event(),
    )


def _serve_producer_metadata(monkeypatch: pytest.MonkeyPatch, producer: MooncakeBaseConnectorWorker) -> None:
    """Answer the consumer's metadata request with what the producer publishes."""
    sending = MooncakeSchedulerSendingThread(
        host="127.0.0.1",
        port=6000,
        engine_id="engine-p",
        metadata={0: producer.xfer_handshake_metadata},
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        use_kv_pp=False,
        ready_event=threading.Event(),
    )
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    monkeypatch.setattr(pull_worker, "zmq_ctx", MagicMock(return_value=context))
    monkeypatch.setattr(pull_worker, "ensure_zmq_send", MagicMock())
    monkeypatch.setattr(pull_worker, "ensure_zmq_recv", MagicMock(return_value=sending.encoded_metadata))


def _producer_transfer_blocks(config: KVCacheConfig, allocated: tuple[list[int], ...]) -> tuple[list[int], ...]:
    """What the producer's scheduler exposes for a finished prefill request."""
    scheduler = MooncakePullConnectorScheduler.__new__(MooncakePullConnectorScheduler)
    scheduler.dcp_size = 1
    scheduler.group_block_size = [group.kv_cache_spec.block_size for group in config.kv_cache_groups]
    scheduler.group_unique_specs = [scheduler._get_group_unique_specs(group) for group in config.kv_cache_groups]
    return scheduler._get_transfer_block_ids(allocated, PROMPT_TOKENS)


@pytest.fixture
def peers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(base_worker, "global_te", MagicMock())
    monkeypatch.setattr(base_worker, "validate_register_region_count", MagicMock())
    torch.manual_seed(0)
    producer_caches = _make_caches(8, fill=True)
    consumer_caches = _make_caches(6, fill=False)
    producer = _register("engine-p", 8, producer_caches)
    consumer = _register("engine-d", 6, consumer_caches)
    return producer, producer_caches, consumer, consumer_caches


def test_kv_sharing_layers_are_not_published_by_either_peer(peers) -> None:
    producer, _, consumer, _ = peers
    transferred = [*OWNERS, *INDEXERS, DRAFT]

    assert SHARING_TARGETS, "the layout must keep a KV-sharing layer"
    for worker in (producer, consumer):
        metadata = worker.xfer_handshake_metadata
        assert metadata.layer_names == transferred
        assert not set(metadata.layer_names) & SHARING_TARGETS.keys()
        assert metadata.group_indices == [TARGET_GROUP] * (len(OWNERS) + len(INDEXERS)) + [DRAFT_GROUP]
    # The borrowing layer keeps the indexer cache it owns.
    borrower = next(iter(SHARING_TARGETS))
    assert borrower.replace(".attn", ".indexer.k_cache") in transferred


def test_pull_moves_every_owned_block_once_and_the_borrower_reads_its_source(
    monkeypatch: pytest.MonkeyPatch, peers
) -> None:
    producer, producer_caches, consumer, consumer_caches = peers
    _serve_producer_metadata(monkeypatch, producer)
    engine = _CopyingTransferEngine()
    thread = _make_consumer_thread(consumer, engine)

    # The producer allocated one lookahead block past the prompt in each group.
    producer_blocks = _producer_transfer_blocks(producer.kv_cache_config, ([5, 2, 6, 0], [3, 0, 7, 1]))
    assert producer_blocks == ([5, 2, 6], [3, 0, 7])
    consumer_blocks = ([1, 4, 3], [5, 2, 0])
    request = ReqMeta(
        local_block_ids=consumer_blocks,
        local_num_prompt_tokens=PROMPT_TOKENS,
        num_external_tokens=PROMPT_TOKENS,
        num_computed_tokens=0,
        remote_block_ids=producer_blocks,
        remote_host="127.0.0.1",
        remote_port=6000,
        remote_engine_id="engine-p",
        remote_request_id="request-p",
        remote_num_prompt_tokens=PROMPT_TOKENS,
        local_full_block_ids=consumer_blocks,
    )

    try:
        failed = thread._handle_requests("engine-p", "127.0.0.1", 6000, {"request-d": request})
    finally:
        thread.executor.shutdown(wait=True)
    assert failed == set()

    group_of = {**{name: TARGET_GROUP for name in [*OWNERS, *INDEXERS]}, DRAFT: DRAFT_GROUP}
    expected_bytes = 0
    for name, group in group_of.items():
        loaded = dict(zip(consumer_blocks[group], producer_blocks[group]))
        for local, remote in zip(_cache_tensors(consumer_caches[name]), _cache_tensors(producer_caches[name])):
            for local_block in range(local.shape[0]):
                if local_block in loaded:
                    assert torch.equal(local[local_block], remote[loaded[local_block]]), (name, local_block)
                    expected_bytes += local[local_block].nbytes
                else:
                    assert torch.all(local[local_block] == SENTINEL), (name, local_block)
    # Nothing was read twice for the layer that shares its source's cache.
    assert engine.num_bytes == expected_bytes
    assert engine.num_reads == 1

    # Read the prompt through the borrower's own binding: it is the cache its
    # source owns, so the KV it attends to is the producer's source-layer KV.
    loaded = dict(zip(consumer_blocks[TARGET_GROUP], producer_blocks[TARGET_GROUP]))
    for borrower, source in SHARING_TARGETS.items():
        assert borrower not in consumer_caches
        bound = _cache_tensors(consumer.kv_caches[borrower])
        for local, remote in zip(bound, _cache_tensors(producer_caches[source]), strict=True):
            assert all(torch.equal(local[d], remote[p]) for d, p in loaded.items()), borrower


def test_consumer_rejects_a_producer_without_its_draft_layers(monkeypatch: pytest.MonkeyPatch, peers) -> None:
    """Draft KV is prefilled by the producer, so both peers must load the drafter."""
    producer, _, consumer, _ = peers
    metadata = producer.xfer_handshake_metadata
    draft_index = metadata.layer_names.index(DRAFT)
    for field in (
        "layer_names",
        "layer_block_sizes",
        "group_indices",
        "kv_caches_base_addr",
        "block_strides",
        "block_lens",
        "block_shapes",
        "block_size_scales",
        "indexer_replication_sizes",
        "dcp_sharded",
    ):
        del getattr(metadata, field)[draft_index]
    _serve_producer_metadata(monkeypatch, producer)
    thread = _make_consumer_thread(consumer, _CopyingTransferEngine())

    try:
        with pytest.raises(ValueError, match="missing layers required by this worker") as error:
            thread._get_remote_metadata("engine-p", "127.0.0.1", 6000)
    finally:
        thread.executor.shutdown(wait=True)
    assert DRAFT in str(error.value)
