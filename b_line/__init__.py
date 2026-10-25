"""B-line: streaming ASR -> segmentation -> MT -> subtitle events.

Initial version. The recognizer and translator are deterministic stand-ins so the
whole pipeline runs end-to-end without a GPU; the real backends are documented
adapters that fail loudly until their environment is configured.
"""

__version__ = "0.1.0"

from .config import BLineConfig  # noqa: F401
