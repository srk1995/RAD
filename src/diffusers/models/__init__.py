# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    _LazyModule,
    is_flax_available,
    is_torch_available,
)


_import_structure = {}

if is_torch_available():
    _import_structure["unets.unet_2d"] = ["UNet2DModel"]
    _import_structure["modeling_utils"] = ["ModelMixin"]
    _import_structure["unets.unet_2d_local"] = ["LocalUNet2DModel"]
    _import_structure["unets.unet_2d_condition"] = ["UNet2DConditionModel"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from .adapter import MultiAdapter, T2IAdapter
        from .unets import (
            LocalUNet2DModel,
            LocalUNet2DConditionModel,
            ModelMixin,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
