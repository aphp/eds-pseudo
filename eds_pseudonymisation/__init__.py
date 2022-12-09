from pathlib import Path

BASE_DIR = Path(__file__).parent

from . import evaluate
from .pipelines.addresses import PseudonymisationAddresses
from .pipelines.dates import PseudonymisationDates
from .pipelines.pseudonymisation import Pseudonymisation
from .pipelines.clean_entities import CleanEntities
from .pipelines.structured import StructuredDataMatcher
