from importlib.metadata import version
from pathlib import Path

import eds_pseudo.data
import eds_pseudo.metrics
import eds_pseudo.pipes

BASE_DIR = Path(__file__).parent
__version__ = version("eds-pseudo")
