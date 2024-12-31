# Obtained from https://github.com/lhoyer/MIC

from .aspp_head import ASPPHead
from .da_head import DAHead
from .daformer_head import DAFormerHead
from .dlv2_head import DLV2Head
from .fcn_head import FCNHead
from .hrda_head import HRDAHead
from .isa_head import ISAHead
from .psp_head import PSPHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead
from .aspp_headv2 import ASPPHeadV2
from .dlv3p_head import DLV3PHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'UPerHead', 'DepthwiseSeparableASPPHead',
    'DAHead', 'DLV2Head', 'SegFormerHead', 'DAFormerHead', 'ISAHead',
    'HRDAHead', 'ASPPHeadV2', 'DLV3PHead']
