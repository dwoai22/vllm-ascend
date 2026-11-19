#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Report whether the installed vLLM can serve Mistral Large 4.

Mistral Large 4 support is not in any vLLM release: it lives in
vllm-project/vllm#64534, which is still unmerged. Without it a serve attempt
fails while parsing ``params.json``, because the Mistral config adapter does
not accept the checkpoint's 128x128 block-FP8 scheme::

    AssertionError: Only NO_SCALES and TENSOR (default) are supported for
    qscheme_act

That message comes from the installed vLLM, not from the checkpoint and not
from vllm-ascend (which does not touch this code path), so the fix is always
to replace the vLLM install.

This script inspects the vLLM package *files on disk* and imports neither vLLM
nor torch, so it is fast, safe to run on a head node, and works even when the
install is broken. Run it with the same interpreter that runs ``vllm serve``:

    python3 check_ml4_support.py
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
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


def main() -> int:
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
