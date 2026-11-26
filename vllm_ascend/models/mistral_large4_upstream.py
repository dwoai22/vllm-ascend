#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Import the Mistral Large 4 implementation reused by the Ascend adapter."""

import sys
from importlib import import_module
from importlib.util import find_spec, module_from_spec

_PACKAGE = "vllm.models.mistral.mistral_large4"


def _ensure_upstream_package() -> None:
    try:
        import_module(_PACKAGE)
    except NotImplementedError as exc:
        if str(exc) != "Mistral Large 4 is only supported on NVIDIA and AMD GPUs.":
            raise
        # The platform entry point rejects out-of-tree platforms before Python
        # can import the shared AMD model classes. The model registry already
        # points to the Ascend adapter, so only the package namespace is needed.
        # Remove this fallback once upstream allows out-of-tree submodule imports.
        spec = find_spec(_PACKAGE)
        if spec is None or not spec.submodule_search_locations:
            raise
        parent, _, name = _PACKAGE.rpartition(".")
        package = sys.modules.setdefault(_PACKAGE, module_from_spec(spec))
        setattr(sys.modules[parent], name, package)


_ensure_upstream_package()

from vllm.models.mistral.mistral_large4.amd.dspark import MistralLarge4DSparkForCausalLM  # noqa: E402
from vllm.models.mistral.mistral_large4.amd.model import (  # noqa: E402
    MistralLarge4ForCausalLM,
    MistralLarge4Model,
)

__all__ = ["MistralLarge4DSparkForCausalLM", "MistralLarge4ForCausalLM", "MistralLarge4Model"]
