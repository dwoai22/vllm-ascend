#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Report whether the installed vLLM can serve Mistral Large 4.

Mistral Large 4 support is not in any vLLM release: it lives in
vllm-project/vllm#64534, which is still unmerged. Without it a serve attempt
fails while parsing ``params.json``, because the Mistral config adapter does
not accept the checkpoint's 128x128 block-FP8 scheme::

    AssertionError: Only NO_SCALES and TENSOR (default) are supported for
    qscheme_act

That message comes from the installed vLLM, not from vllm-ascend (which does
not touch this code path). A vLLM that *does* carry #64534 can still reject a
checkpoint, with a different message, so pass ``--model`` to have the
checkpoint's own quantization block checked against the post-PR rules and see
which of the two is at fault.

This script inspects the vLLM package *files on disk* and imports neither vLLM
nor torch, so it is fast, safe to run on a head node, and works even when the
install is broken. Run it with the same interpreter that runs ``vllm serve``:

    python3 check_ml4_support.py --model /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

# Added by #64534 to _remap_mistral_quantization_args.
BLOCK_MARKER = "BLOCK quantization must be used for both weights and activations"
ML4_PACKAGE = Path("models") / "mistral" / "mistral_large4"
ML4_ARCH = "MistralLarge4ForCausalLM"
FIX = """
    test -d /vllm-workspace/vllm || git clone --depth=40 \\
        https://github.com/vllm-project/vllm.git /vllm-workspace/vllm
    cd /vllm-workspace/vllm
    git fetch --depth=40 https://github.com/vllm-project/vllm.git pull/64534/head:ml4
    git checkout ml4
    python3 -m pip uninstall -y vllm
    VLLM_TARGET_DEVICE=empty python3 -m pip install -e . --no-build-isolation
"""


def vllm_root() -> Path | None:
    """The installed vllm package directory, without importing it."""
    try:
        spec = importlib.util.find_spec("vllm")
    except Exception:  # a broken install can raise rather than return None
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def dist_info() -> tuple[str, str, bool]:
    """(version, location, editable) from the installed distribution metadata."""
    try:
        dist = importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError:
        return "not installed", "-", False
    version = dist.version
    location = str(getattr(dist, "_path", "") or "-")
    editable = False
    try:
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            editable = bool((json.loads(direct_url).get("dir_info") or {}).get("editable"))
    except Exception:
        pass
    return version, location, editable


def read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def shadowing_copies() -> list[Path]:
    """Plain ``vllm`` package directories on sys.path, in resolution order.

    An editable install resolves through a finder rather than a path entry, so
    it does not appear here; a leftover directory copy that shadows it does.
    """
    found: list[Path] = []
    for entry in sys.path:
        candidate = Path(entry or os.getcwd()) / "vllm" / "__init__.py"
        if candidate.is_file():
            found.append(candidate.parent.resolve())
    return found


def check_checkpoint(model_dir: Path) -> bool:
    """Apply #64534's own quantization rules to this checkpoint's params.json.

    Mirrors _remap_mistral_quantization_args: the same accepted values and the
    same two assertions, so a rejection here is the one a post-PR vLLM raises.
    """
    print(f"\ncheckpoint  : {model_dir}")
    params_path = model_dir / "params.json"
    if not params_path.is_file():
        print("  [FAIL] no params.json (vLLM reads this model through --config-format mistral)")
        return False
    try:
        params = json.loads(params_path.read_text())
    except (OSError, ValueError) as exc:
        print(f"  [FAIL] cannot parse params.json: {exc}")
        return False

    ok = True
    if (model_dir / "config.json").is_file():
        print("  [FAIL] a config.json sits beside params.json; vLLM merges it in as")
        print("         defaults, which can drag in a stale layer count. Remove it.")
        ok = False

    quantization = params.get("quantization")
    if not quantization:
        print("  [ok  ] no quantization block (BF16 checkpoint): nothing to validate")
        return ok
    if not isinstance(quantization, dict):
        print(f"  [FAIL] quantization is {type(quantization).__name__}, expected an object")
        return False

    qformat = quantization.get("qformat_weight")
    act = quantization.get("qscheme_act")
    weight = quantization.get("qscheme_weight")
    print(f"  quantization: qformat_weight={qformat!r} qscheme_act={act!r} qscheme_weight={weight!r}")

    if qformat != "fp8_e4m3":
        method = str(quantization.get("quant_method", "")).lower().replace("_", "-")
        if method == "compressed-tensors":
            print("  [ok  ] compressed-tensors checkpoint: passed through unvalidated here")
            return ok
        print(f"  [FAIL] post-PR vLLM raises: Found unknown quantization={quantization!r} in config")
        return False

    # The two assertions #64534 applies, in order.
    if act not in ("NO_SCALES", "TENSOR", "BLOCK", None):
        print("  [FAIL] post-PR vLLM raises: Only NO_SCALES, TENSOR (default) and BLOCK")
        print(f"         are supported for qscheme_act (yours is {act!r})")
        print("         This is a checkpoint vLLM cannot read even WITH #64534.")
        return False
    if (act == "BLOCK") != (weight == "BLOCK"):
        print("  [FAIL] post-PR vLLM raises: BLOCK quantization must be used for both")
        print(f"         weights and activations (act={act!r}, weight={weight!r})")
        print("         This is a checkpoint vLLM cannot read even WITH #64534.")
        return False

    if weight == "BLOCK":
        missing = [k for k in ("block_size_out", "block_size_in") if k not in quantization]
        if missing:
            print(f"  [FAIL] BLOCK weights but no {', '.join(missing)}: post-PR vLLM raises KeyError")
            return False
        block = [quantization["block_size_out"], quantization["block_size_in"]]
        print(f"  [ok  ] block-FP8 accepted by #64534 -> weight_block_size {block}, dynamic activations")
    else:
        scheme = "dynamic" if act == "NO_SCALES" else "static"
        print(f"  [ok  ] per-tensor FP8 accepted -> {scheme} activations (pre-PR vLLM accepts this too)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Check this environment for Mistral Large 4 support.")
    parser.add_argument("--model", type=Path, help="checkpoint directory, to validate its params.json too")
    args = parser.parse_args()

    print(f"interpreter : {sys.executable}")
    version, location, editable = dist_info()
    print(f"vllm version: {version}")
    print(f"dist path   : {location}{'  (editable)' if editable else ''}")

    root = vllm_root()
    if root is None:
        print("\nvLLM is not importable by this interpreter.")
        print(f"Install it with #64534:{FIX}")
        return 1
    print(f"package     : {root}")
    copies = shadowing_copies()
    if len(copies) > 1 or (copies and copies[0] != root.resolve()):
        print("\nMore than one vllm package is reachable; the FIRST one wins:")
        for index, copy in enumerate(copies):
            print(f"  {index}: {copy}{'   <-- imported' if copy == root.resolve() else ''}")
        print("A leftover directory copy shadows an editable install. Remove it, or")
        print("`python3 -m pip uninstall -y vllm` until none remains, then reinstall.")

    checks: dict[str, bool] = {}

    adapter = root / "transformers_utils" / "configs" / "mistral.py"
    adapter_source = read(adapter)
    if adapter_source is None:
        print(f"\ncannot read {adapter}")
        return 1
    checks["mistral config adapter accepts block-FP8 (qscheme_act=BLOCK)"] = BLOCK_MARKER in adapter_source

    checks["vllm.models.mistral.mistral_large4 package present"] = (root / ML4_PACKAGE).is_dir()

    registry_source = read(root / "model_executor" / "models" / "registry.py") or ""
    checks[f"{ML4_ARCH} in the model registry"] = ML4_ARCH in registry_source

    print()
    for label, ok in checks.items():
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label}")

    checkpoint_ok = check_checkpoint(args.model) if args.model else True

    if all(checks.values()) and not checkpoint_ok:
        print("\nThis vLLM carries #64534, so the install is fine: the checkpoint above")
        print("is what vLLM rejects. Fix it, or point --model at the right directory.")
        return 1

    if all(checks.values()):
        print("\nThis vLLM carries #64534: Mistral Large 4 can be served.")
        print("If serving still fails, the next thing to check is that vllm-ascend")
        print("registered its adapter, which is gated on the package above:")
        print(
            '  python3 -c "from vllm_ascend.models import _upstream_has_module as f;'
            " print(f('vllm.models.mistral.mistral_large4'))\""
        )
        return 0

    print("\nThis vLLM does NOT carry #64534, so Mistral Large 4 cannot be served.")
    if not checks["mistral config adapter accepts block-FP8 (qscheme_act=BLOCK)"]:
        print("A serve attempt fails while parsing params.json, with:")
        print("  AssertionError: Only NO_SCALES and TENSOR (default) are supported for qscheme_act")
    print("Patching that assertion does not help: without the model package above,")
    print(f"{ML4_ARCH} is unregistered and vllm-ascend's adapter stays inert.")
    print(f"Replace the install:{FIX}")
    if not editable and version != "not installed":
        print("Note: this install is not editable, so it lives in site-packages and a")
        print("source checkout elsewhere is NOT what gets imported. Uninstall first,")
        print("as above, or the old copy keeps winning.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
