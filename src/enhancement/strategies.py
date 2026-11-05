"""Imports every built-in enhancement strategy so registry decorators run.

Adding a new strategy: create the class with ``@register`` in any module and
import it here.
"""

from .clahe import CLAHEEnhancer              # noqa: F401
from .histogram_eq import HistogramEqualizationEnhancer  # noqa: F401
from .gamma import GammaCorrectionEnhancer    # noqa: F401
from .white_balance import WhiteBalanceEnhancer  # noqa: F401
from .dark_channel import DarkChannelPriorEnhancer  # noqa: F401
from .udcp import UDCPEnhancer                # noqa: F401
from .color_restoration import ColorRestorationEnhancer  # noqa: F401
from .composite_fusion import FusionWCGEnhancer  # noqa: F401
