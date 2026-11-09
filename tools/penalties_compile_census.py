#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile census for the Model Runner V2 penalties Triton kernels.

Black-box validation evidence: how many times do the MRV2 sampling penalty
kernels compile, with which launch shapes and which specialization metadata,
and does any of that compilation happen once the engine is already serving.

Why this needs measuring
------------------------
``vllm_ascend/worker/v2/sample/penalties.py`` is reached through
``patch/worker/patch_v2/patch_triton.py``, which replaces
``vllm.v1.worker.gpu.sample.penalties.apply_penalties``. vLLM's MRV2 logits
processor launches it only when a request in the batch actually carries a
penalty, so dummy runs and ACL graph capture never touch it, and
``kernel_warmup`` warms the *V1* kernel (``ops/triton/penalty.py``), not this
one. Every compiled variant therefore lands on a real request.

Launch shapes follow the number of sampled rows (requests x (1 + draft
positions)), not prompt length, so the sweep varies batch size and lets waves
drain raggedly instead of varying prompt text.

Single card, in process (the simple case)
-----------------------------------------
    python3 tools/penalties_compile_census.py \
        --model /path/to/Qwen3.6-35B-A3B-w8a8 --trust-remote-code \
        --batches 1,7,16,64,128 --out /tmp/penalties-census

Tensor parallel (compiles happen in the worker processes)
---------------------------------------------------------
The probe must be installed inside each worker, so inject it with
``sitecustomize.py``, which every Python process imports at startup:

    mkdir -p /tmp/census && cp tools/penalties_compile_census.py /tmp/census/
    cat > /tmp/census/sitecustomize.py <<'EOF'
    import os
    if os.environ.get("PENALTIES_CENSUS_AUTO") == "1":
        try:
            import penalties_compile_census as census
            census.install(auto_report=True)
        except Exception as exc:  # never break the host process
            print("penalties census install failed:", exc)
    EOF

    PYTHONPATH=/tmp/census:${PYTHONPATH} \
    PENALTIES_CENSUS_AUTO=1 \
    PENALTIES_CENSUS_MARKER=/tmp/census/serving.marker \
    PENALTIES_CENSUS_OUT=/tmp/census/report \
    python3 /tmp/census/penalties_compile_census.py --model ... --tp 4 --out /tmp/census

Each process then writes ``report.rank<N>.pid<N>.json``; the per-rank files are
the evidence, since the driver process never launches the kernel itself.

Real GPQA prompts
-----------------
Pass ``--dataset-json file.json`` with a JSON list of rendered prompt strings
(or a list of objects plus ``--dataset-field``). The dataset itself is gated on
Hugging Face, so rendering it is left to the caller; ``--batches`` matters far
more than prompt content for the shape census.
"""

from __future__ import annotations

import os
import sys

# Running this as ``python3 tools/penalties_compile_census.py`` puts ``tools/``
# first on sys.path, and this repo keeps a package called ``bisect`` there that
# shadows the standard library module of the same name -- which ``random``
# imports, so the interpreter dies (or worse, executes the wrong package)
# before main() is reached. Nothing here is imported from the script's own
# directory, so drop it before importing anything else.
_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0] or os.getcwd()) == _HERE:
    sys.path.pop(0)

import argparse  # noqa: E402
import atexit  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

DEFAULT_KERNELS = (
    "vllm_ascend.worker.v2.sample.penalties:_penalties_kernel",
    "vllm_ascend.worker.v2.sample.penalties:_bincount_kernel",
)

# Positional tail of the launcher's call, so the runtime ints Triton can
# specialize on get real names. Mirrors tests/ut/worker/v2/test_penalties.py.
NAMED_TAIL = {"_penalties_kernel": ("vocab_size", "total_tiles", "tiles_per_worker")}

_PHASE_LOCK = threading.Lock()
_PHASE = "startup"
_MARKER = os.environ.get("PENALTIES_CENSUS_MARKER")
_MARKER_LAST_CHECK = 0.0
_CENSUS: Census | None = None


def mark_phase(phase: str) -> None:
    """Record the phase here, and signal other processes through the marker file."""
    global _PHASE
    with _PHASE_LOCK:
        _PHASE = phase
    if _MARKER and phase == "serving":
        try:
            with open(_MARKER, "w") as handle:
                handle.write(str(time.time()))
        except OSError:
            pass


def current_phase() -> str:
    """Phase of the run; latches to serving once the marker file appears."""
    global _PHASE, _MARKER_LAST_CHECK
    if _PHASE == "serving" or not _MARKER:
        return _PHASE
    now = time.monotonic()
    if now - _MARKER_LAST_CHECK > 0.5:
        _MARKER_LAST_CHECK = now
        if os.path.exists(_MARKER):
            with _PHASE_LOCK:
                _PHASE = "serving"
    return _PHASE


def _cache_keys(kernel: Any) -> set[str] | None:
    """Every compiled variant of a JITFunction, as stable strings.

    Triton's cache layout differs between versions (``{device: {key: kernel}}``
    in most, flat ``{key: kernel}`` in others), so both are handled. ``None``
    means the layout was not recognised, and compile detection falls back to
    timing outliers.
    """
    cache = getattr(kernel, "cache", None)
    if cache is None:
        return None
    keys: set[str] = set()
    try:
        for outer, value in list(cache.items()):
            if isinstance(value, dict):
                keys.update(f"{outer}|{inner!r}" for inner in list(value.keys()))
            else:
                keys.add(f"{outer!r}")
    except Exception:
        return None
    return keys


def _int_class(value: Any) -> str | None:
    """How Triton keys a non-constexpr int argument: divisible by 16, equal to 1."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return f"div16={value % 16 == 0},eq1={value == 1}"


def _launch_signature(kernel_name: str, grid: Any, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Shape plus predicted specialization metadata for one launch."""
    sig: dict[str, Any] = {}
    sig["grid"] = tuple(int(g) for g in grid) if isinstance(grid, tuple) else repr(grid)
    shape = getattr(args[0], "shape", None) if args else None
    if shape is not None:
        sig["rows"] = int(shape[0])
        sig["cols"] = int(shape[-1])
        sig["dtype"] = str(getattr(args[0], "dtype", ""))
    named = NAMED_TAIL.get(kernel_name)
    ints: dict[str, int] = {}
    if named and len(args) >= len(named):
        tail = args[-len(named) :]
        if all(isinstance(v, int) and not isinstance(v, bool) for v in tail):
            ints = {name: int(value) for name, value in zip(named, tail)}
    if not ints:
        ints = {f"arg{i}": int(v) for i, v in enumerate(args) if isinstance(v, int) and not isinstance(v, bool)}
    sig["int_args"] = ints
    sig["constexpr"] = {
        key: (int(value) if isinstance(value, int) and not isinstance(value, bool) else repr(value))
        for key, value in sorted(kwargs.items())
    }
    classes = [f"{name}:{_int_class(value)}" for name, value in sorted(ints.items())]
    classes += [f"{key}={value}" for key, value in sig["constexpr"].items()]
    sig["spec_class"] = ";".join(classes)
    return sig


class Census:
    """Collects compile events, launch shapes and specialization classes."""

    def __init__(self, slow_launch_s: float = 0.5) -> None:
        self._lock = threading.RLock()  # report() calls compiles_by_phase() while holding it
        self.slow_launch_s = slow_launch_s
        self.start_wall = time.time()
        self.start_mono = time.monotonic()
        self.kernels: dict[str, dict[str, Any]] = {}
        self.compiles: list[dict[str, Any]] = []
        self.shapes: dict[str, dict[str, Any]] = {}
        self.global_compiles: list[dict[str, Any]] = []
        self.global_hook: str | None = None
        self.detection = "unknown"

    def register(self, name: str, kernel: Any) -> None:
        keys = _cache_keys(kernel)
        with self._lock:
            self.kernels[name] = {
                "do_not_specialize": sorted(str(x) for x in (getattr(kernel, "do_not_specialize", ()) or ())),
                "variants_at_install": len(keys) if keys is not None else None,
                "cache_layout_understood": keys is not None,
            }

    def observe(
        self,
        name: str,
        kernel: Any,
        grid: Any,
        args: tuple,
        kwargs: dict,
        keys_before: set[str] | None,
        elapsed: float,
    ) -> None:
        phase = current_phase()
        keys_after = _cache_keys(kernel)
        sig = _launch_signature(name, grid, args, kwargs)
        new_keys: list[str] = []
        if keys_before is not None and keys_after is not None:
            new_keys = sorted(keys_after - keys_before)
            detection = "cache-keys"
        else:
            detection = "slow-launch"
        compiled = bool(new_keys) or (detection == "slow-launch" and elapsed >= self.slow_launch_s)
        shape_key = f"{name}|rows={sig.get('rows')}|{sig['spec_class']}"
        with self._lock:
            self.detection = detection
            entry = self.shapes.setdefault(
                shape_key,
                {
                    "kernel": name,
                    "rows": sig.get("rows"),
                    "spec_class": sig["spec_class"],
                    "int_args": sig["int_args"],
                    "grid": sig["grid"],
                    "first_phase": phase,
                    "launches": 0,
                    "max_launch_s": 0.0,
                },
            )
            entry["launches"] += 1
            entry["max_launch_s"] = max(entry["max_launch_s"], round(elapsed, 4))
            if compiled:
                self.compiles.append(
                    {
                        "index": len(self.compiles) + 1,
                        "kernel": name,
                        "phase": phase,
                        "t_since_install_s": round(time.monotonic() - self.start_mono, 3),
                        "launch_s": round(elapsed, 3),
                        "detected_by": "new cache key" if new_keys else "slow launch (cache layout unknown)",
                        "new_cache_keys": new_keys,
                        "signature": sig,
                    }
                )

    def note_global_compile(self, name: str, already_compiled: bool) -> None:
        with self._lock:
            self.global_compiles.append(
                {
                    "kernel": name,
                    "phase": current_phase(),
                    "already_compiled": already_compiled,
                    "t_since_install_s": round(time.monotonic() - self.start_mono, 3),
                }
            )

    def compiles_by_phase(self) -> dict[str, int]:
        counts: dict[str, int] = {"startup": 0, "serving": 0}
        with self._lock:
            for event in self.compiles:
                counts[event["phase"]] = counts.get(event["phase"], 0) + 1
        return counts

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pid": os.getpid(),
                "rank": _rank(),
                "started_at": self.start_wall,
                "detection": self.detection,
                "global_hook": self.global_hook,
                "kernels": dict(self.kernels),
                "compiles_by_phase": self.compiles_by_phase(),
                "compiles": list(self.compiles),
                "shapes": list(self.shapes.values()),
                "global_compiles": list(self.global_compiles),
            }

    def render(self) -> str:
        data = self.report()
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append(f"MRV2 penalties kernel compile census  (pid {data['pid']}, rank {data['rank']})")
        lines.append("=" * 78)
        if not data["kernels"]:
            lines.append("NO KERNEL WAS INSTRUMENTED -- the target module was never imported in this process.")
            lines.append("For tensor parallel runs the probe must be installed inside the workers; see the")
            lines.append("sitecustomize recipe in this file's docstring.")
            return "\n".join(lines)
        for name, info in sorted(data["kernels"].items()):
            dns = ", ".join(info["do_not_specialize"]) or "(none)"
            lines.append(f"kernel {name}: do_not_specialize = {dns}")
            if not info["cache_layout_understood"]:
                lines.append("  warning: triton cache layout not recognised; compiles inferred from launch time")
        lines.append("")
        counts = data["compiles_by_phase"]
        lines.append(f"compiles during startup : {counts.get('startup', 0)}")
        lines.append(f"compiles during serving : {counts.get('serving', 0)}")
        if data["compiles"]:
            lines.append("")
            lines.append("compile events")
            for event in data["compiles"]:
                sig = event["signature"]
                ints = " ".join(f"{k}={v}" for k, v in sig["int_args"].items())
                lines.append(
                    f"  #{event['index']:<3} {event['phase']:<8} t=+{event['t_since_install_s']:>9.3f}s"
                    f"  launch={event['launch_s']:>7.3f}s  {event['kernel']}"
                )
                lines.append(f"       rows={sig.get('rows')} grid={sig['grid']} {ints}")
                lines.append(f"       constexpr={sig['constexpr']}")
                for key in event["new_cache_keys"]:
                    lines.append(f"       new cache key: {key[:160]}")
        shapes = sorted(data["shapes"], key=lambda item: (item["kernel"], -item["launches"]))
        lines.append("")
        lines.append(f"distinct (shape, specialization) classes observed: {len(shapes)}")
        for entry in shapes:
            ints = " ".join(f"{k}={v}" for k, v in entry["int_args"].items())
            lines.append(
                f"  {entry['kernel']:<20} rows={str(entry['rows']):<6} launches={entry['launches']:<7}"
                f" first={entry['first_phase']:<8} max_launch={entry['max_launch_s']:.4f}s"
            )
            lines.append(f"      {ints}")
            lines.append(f"      spec_class={entry['spec_class'][:150]}")
        if data["global_compiles"]:
            compiled = [item for item in data["global_compiles"] if not item["already_compiled"]]
            serving = [item for item in compiled if item["phase"] == "serving"]
            lines.append("")
            lines.append(
                f"process-wide triton compiles via {data['global_hook']}: {len(compiled)}"
                f" ({len(serving)} during serving)"
            )
            for item in serving[:20]:
                lines.append(f"  serving compile: {item['kernel']} at t=+{item['t_since_install_s']}s")
        lines.append("")
        verdict = counts.get("serving", 0)
        lines.append(f"VERDICT: {verdict} penalties-kernel compile(s) occurred during the SERVING phase.")
        return "\n".join(lines)


class _KernelProbe:
    """Stands in for a JITFunction, recording each launch and each new variant."""

    def __init__(self, kernel: Any, name: str, census: Census) -> None:
        self._kernel = kernel
        self._name = name
        self._census = census

    def __getitem__(self, grid: Any):
        inner = self._kernel[grid]

        def launch(*args: Any, **kwargs: Any):
            keys_before = _cache_keys(self._kernel)
            start = time.perf_counter()
            try:
                return inner(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                with contextlib.suppress(Exception):  # instrumentation must never break the run
                    self._census.observe(self._name, self._kernel, grid, args, kwargs, keys_before, elapsed)

        return launch

    def __call__(self, *args: Any, **kwargs: Any):
        return self._kernel(*args, **kwargs)

    def __getattr__(self, item: str):
        if item in {"_kernel", "_name", "_census"}:
            raise AttributeError(item)
        return getattr(self._kernel, item)


def _rank() -> str:
    for key in ("RANK", "LOCAL_RANK", "VLLM_DP_RANK", "ASCEND_RT_VISIBLE_DEVICES"):
        value = os.environ.get(key)
        if value:
            return value
    return "0"


def _install_global_compile_hook(census: Census) -> str | None:
    """Best-effort process-wide compile log; API differs between Triton versions."""

    def hook(*_args: Any, **kwargs: Any) -> None:
        try:
            fn = kwargs.get("fn")
            target = getattr(fn, "jit_function", fn)
            name = getattr(target, "__name__", None) or str(target)
            census.note_global_compile(name, bool(kwargs.get("already_compiled")))
        except Exception:
            pass

    # Only hook what the host process has already imported. Importing triton
    # from here would drag in the whole toolchain (and can block for minutes)
    # before the engine has even started, so the waiter thread retries instead.
    triton = sys.modules.get("triton")
    if triton is None:
        return None
    try:
        runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
        if runtime is not None and hasattr(runtime, "jit_cache_hook"):
            runtime.jit_cache_hook = hook
            return "triton.knobs.runtime.jit_cache_hook"
    except Exception:
        pass
    try:
        jit_module = sys.modules.get("triton.runtime.jit")
        jit_function = getattr(jit_module, "JITFunction", None) if jit_module is not None else None
        if jit_function is not None and hasattr(jit_function, "cache_hook"):
            jit_function.cache_hook = hook
            return "JITFunction.cache_hook"
    except Exception:
        pass
    return None


def _patch(spec: str, census: Census) -> bool:
    module_name, _, attr = spec.partition(":")
    module = sys.modules.get(module_name)
    if module is None:
        return False
    kernel = getattr(module, attr, None)
    if kernel is None:
        return False
    if isinstance(kernel, _KernelProbe):
        return True
    census.register(attr, kernel)
    setattr(module, attr, _KernelProbe(kernel, attr, census))
    return True


def install(
    kernels: tuple[str, ...] = DEFAULT_KERNELS,
    auto_report: bool = False,
    slow_launch_s: float = 0.5,
    timeout_s: float = 1800.0,
) -> Census:
    """Attach the probe, waiting in a background thread for the module to import."""
    global _CENSUS
    census = _CENSUS or Census(slow_launch_s=slow_launch_s)
    _CENSUS = census
    census.global_hook = census.global_hook or _install_global_compile_hook(census)

    pending = [spec for spec in kernels if not _patch(spec, census)]
    if pending or census.global_hook is None:

        def waiter() -> None:
            """Attach to the kernels, and to triton, as soon as they are imported."""
            deadline = time.monotonic() + timeout_s
            remaining = list(pending)
            while time.monotonic() < deadline:
                remaining = [spec for spec in remaining if not _patch(spec, census)]
                if census.global_hook is None:
                    census.global_hook = _install_global_compile_hook(census)
                if not remaining and census.global_hook is not None:
                    return
                time.sleep(0.05 if remaining else 1.0)

        threading.Thread(target=waiter, name="penalties-census", daemon=True).start()

    if auto_report:
        atexit.register(_dump_at_exit, census)
    return census


def _dump_at_exit(census: Census) -> None:
    out = os.environ.get("PENALTIES_CENSUS_OUT")
    text = census.render()
    print(text, flush=True)
    if not out:
        return
    path = out if out.endswith(".json") else f"{out}.rank{_rank()}.pid{os.getpid()}.json"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            json.dump(census.report(), handle, indent=2, default=str)
        print(f"penalties census written to {path}", flush=True)
    except OSError as exc:
        print(f"penalties census could not be written to {path}: {exc}", flush=True)


_FIELDS = (
    "organic chemistry",
    "molecular biology",
    "quantum mechanics",
    "astrophysics",
    "genetics",
    "physical chemistry",
)
_SENTENCES = (
    "A reaction is run under anhydrous conditions with a stoichiometric excess of the reagent.",
    "The measured spectrum shows a single broad resonance that sharpens on cooling.",
    "Assume the perturbation is small enough that first-order theory applies.",
    "The observed decay rate is independent of the initial concentration of the substrate.",
    "Two independent assays disagree by slightly more than their combined uncertainty.",
    "The sample was prepared by sequential dilution from a stock of known molarity.",
)


def gpqa_style_prompts(count: int, seed: int, body_sentences: int = 6) -> list[str]:
    """Deterministic multiple-choice prompts shaped like GPQA items."""
    rng = random.Random(seed)
    prompts: list[str] = []
    for index in range(count):
        field = _FIELDS[index % len(_FIELDS)]
        body = " ".join(rng.choice(_SENTENCES) for _ in range(body_sentences))
        prompts.append(
            f"The following is a graduate-level multiple choice question in {field}.\n\n"
            f"{body}\n\nWhich statement about the system is correct?\n"
            f"(A) The rate-limiting step changes when the temperature is raised.\n"
            f"(B) The observable is unchanged because the symmetry forbids the transition.\n"
            f"(C) The effect scales linearly with the applied field.\n"
            f"(D) None of the above explains the measurement.\n\n"
            f"Think step by step, then give the letter of the correct option."
        )
    return prompts


def _load_prompts(args: argparse.Namespace) -> list[str]:
    needed = max(args.batches)
    if not args.dataset_json:
        return gpqa_style_prompts(needed, args.seed)
    with open(args.dataset_json) as handle:
        raw = json.load(handle)
    prompts: list[str] = []
    for item in raw:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and args.dataset_field in item:
            prompts.append(str(item[args.dataset_field]))
    if not prompts:
        raise SystemExit(f"no prompts found in {args.dataset_json} (field {args.dataset_field!r})")
    while len(prompts) < needed:
        prompts.extend(prompts[: needed - len(prompts)])
    return prompts[:needed]


def _worker_recipe(script: str, out: str) -> str:
    return (
        "Tensor parallel detected: the penalties kernel compiles inside the worker\n"
        "processes, which this driver cannot see. Install the probe there with\n"
        "sitecustomize.py (see this script's docstring), for example:\n\n"
        f"  mkdir -p /tmp/census && cp {script} /tmp/census/\n"
        "  printf '%s\\n' \"import os\" \\\n"
        "    \"if os.environ.get('PENALTIES_CENSUS_AUTO') == '1':\" \\\n"
        '    "    import penalties_compile_census as c; c.install(auto_report=True)" \\\n'
        "    > /tmp/census/sitecustomize.py\n\n"
        f"  PYTHONPATH=/tmp/census:${{PYTHONPATH}} PENALTIES_CENSUS_AUTO=1 \\\n"
        f"  PENALTIES_CENSUS_MARKER={out}/serving.marker PENALTIES_CENSUS_OUT={out}/report \\\n"
        f"  python3 /tmp/census/penalties_compile_census.py <same args>\n"
    )


def _self_test() -> int:
    """Prove the probe detects variants and phases, without a device."""
    import types

    class _FakeKernel:
        """Mimics a JITFunction: one cache entry per distinct specialization key."""

        do_not_specialize = ()

        def __init__(self) -> None:
            self.cache = {"npu:0": {}}

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                tail = args[-3:]
                key = (tuple(v % 16 == 0 for v in tail), tuple(v == 1 for v in tail), tuple(sorted(kwargs.items())))
                self.cache["npu:0"].setdefault(key, object())

            return launch

    module = types.ModuleType("fake_penalties_module")
    module._penalties_kernel = _FakeKernel()
    sys.modules["fake_penalties_module"] = module

    census = install(kernels=("fake_penalties_module:_penalties_kernel",))
    probe = module._penalties_kernel
    logits = types.SimpleNamespace(shape=(4, 151936), dtype="torch.float32")
    probe[(48,)](logits, 151936, 151936, 152, 4, NUM_VOCAB_BLOCKS=38, BLOCK_SIZE=4096)
    probe[(48,)](logits, 151936, 151936, 152, 4, NUM_VOCAB_BLOCKS=38, BLOCK_SIZE=4096)
    mark_phase("serving")
    probe[(48,)](logits, 151936, 151936, 19456, 406, NUM_VOCAB_BLOCKS=38, BLOCK_SIZE=4096)

    counts = census.compiles_by_phase()
    report = census.report()
    print(census.render())
    failures = []
    if counts.get("startup") != 1:
        failures.append(f"expected 1 startup compile, got {counts.get('startup')}")
    if counts.get("serving") != 1:
        failures.append(f"expected 1 serving compile, got {counts.get('serving')}")
    if len(report["shapes"]) != 2:
        failures.append(f"expected 2 shape classes, got {len(report['shapes'])}")
    if report["shapes"][0]["launches"] != 2:
        failures.append("expected the repeated launch to be counted without a new compile")
    print("")
    if failures:
        for failure in failures:
            print(f"SELF-TEST FAILED: {failure}")
        return 1
    print("SELF-TEST PASSED: compile detection, phase split and shape census all behave.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="model path or id")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batches", default="1,7,16,64", help="comma separated wave sizes")
    parser.add_argument("--output-len", type=int, default=64, help="max_tokens per request")
    parser.add_argument("--ragged", action="store_true", default=True, help="jitter max_tokens so waves drain unevenly")
    parser.add_argument("--no-ragged", dest="ragged", action="store_false")
    parser.add_argument("--frequency-penalty", type=float, default=0.1)
    parser.add_argument("--presence-penalty", type=float, default=0.1)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dataset-json", default=None, help="JSON list of prompts (e.g. rendered GPQA)")
    parser.add_argument("--dataset-field", default="prompt")
    parser.add_argument("--engine-kwargs-json", default=None, help="extra LLM() kwargs as a JSON object")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="directory for the JSON report and phase marker")
    parser.add_argument("--slow-launch-s", type=float, default=0.5)
    parser.add_argument("--fail-on-serving-compiles", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="verify the probe on CPU, no model needed")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.model:
        parser.error("--model is required unless --self-test is given")

    args.batches = [int(value) for value in args.batches.split(",") if value.strip()]
    out_dir = args.out or os.path.join(os.getcwd(), "penalties-census")
    os.makedirs(out_dir, exist_ok=True)

    global _MARKER
    _MARKER = os.environ.setdefault("PENALTIES_CENSUS_MARKER", os.path.join(out_dir, "serving.marker"))
    if os.path.exists(_MARKER):
        os.unlink(_MARKER)
    os.environ.setdefault("PENALTIES_CENSUS_OUT", os.path.join(out_dir, "report"))
    if args.tp == 1:
        # Keep the engine core in this process so the kernel cache is visible here.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    else:
        print(_worker_recipe(os.path.abspath(__file__), out_dir), flush=True)

    census = install(auto_report=False, slow_launch_s=args.slow_launch_s)
    prompts = _load_prompts(args)

    from vllm import LLM, SamplingParams

    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
    }
    if args.tokenizer:
        engine_kwargs["tokenizer"] = args.tokenizer
    if args.max_model_len:
        engine_kwargs["max_model_len"] = args.max_model_len
    if args.engine_kwargs_json:
        engine_kwargs.update(json.loads(args.engine_kwargs_json))

    print(f"[census] building engine; phase = {current_phase()}", flush=True)
    llm = LLM(**engine_kwargs)
    startup = census.compiles_by_phase().get("startup", 0)
    print(f"[census] engine ready; startup-phase penalties compiles so far = {startup}", flush=True)
    mark_phase("serving")

    rng = random.Random(args.seed)
    for batch in args.batches:
        wave = prompts[:batch]
        params = []
        for _ in wave:
            jitter = rng.randint(-args.output_len // 4, args.output_len // 4) if args.ragged else 0
            params.append(
                SamplingParams(
                    temperature=args.temperature,
                    max_tokens=max(8, args.output_len + jitter),
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    repetition_penalty=args.repetition_penalty,
                    ignore_eos=True,
                )
            )
        print(f"[census] wave of {batch} request(s)", flush=True)
        llm.generate(wave, params)

    print("")
    print(census.render(), flush=True)
    path = os.path.join(out_dir, f"report.rank{_rank()}.pid{os.getpid()}.json")
    with open(path, "w") as handle:
        json.dump(census.report(), handle, indent=2, default=str)
    print(f"\nJSON report: {path}", flush=True)

    serving = census.compiles_by_phase().get("serving", 0)
    if args.fail_on_serving_compiles and serving:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
