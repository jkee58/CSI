# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2

from .inference import inference_model, init_model, show_result_pyplot
from .mmseg_inferencer import MMSegInferencer
from .remote_sense_inferencer import RSImage, RSInferencer

__all__ = [
    'init_model', 'inference_model', 'show_result_pyplot', 'MMSegInferencer',
    'RSInferencer', 'RSImage'
]