from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_onnx_available,
    is_torch_available,
    is_torch_npu_available,
    is_transformers_available,
)


# These modules contain pipelines from multiple libraries/frameworks
_dummy_objects = {}
_import_structure = {
    "controlnet": [],
    "controlnet_hunyuandit": [],
    "controlnet_sd3": [],
    "controlnet_xs": [],
    "deprecated": [],
    "latent_diffusion": [],
    "ledits_pp": [],
    "marigold": [],
    "pag": [],
    "stable_diffusion": [],
    "stable_diffusion_xl": [],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_pt_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
else:
    _import_structure["auto_pipeline"] = [
        "AutoPipelineForImage2Image",
        "AutoPipelineForInpainting",
        "AutoPipelineForText2Image",
    ]
    _import_structure["ddim"] = ["DDIMPipeline"]
    _import_structure["ddpm"] = ["DDPMPipeline"]



if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_pt_objects import *  # noqa F403

    else:
        from .auto_pipeline import (
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
            AutoPipelineForText2Image,
        )
        from .ddim import DDIMPipeline
        from .ddpm import DDPMPipeline


else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
