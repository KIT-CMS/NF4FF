"""Public law task registry.

Task implementations live in :mod:`tasks.workflow`; this module stays as the
stable entry point used by ``law run`` and ``run_workflow.py``.
"""

from tasks.corrections import *  # noqa: F401,F403
from tasks.dataset import *  # noqa: F401,F403
from tasks.enrichment import *  # noqa: F401,F403
from tasks.fake_factors import *  # noqa: F401,F403
from tasks.fractions import *  # noqa: F401,F403
from tasks.interpretation import *  # noqa: F401,F403
from tasks.normalizing_flow import *  # noqa: F401,F403
from tasks.single_dnn import *  # noqa: F401,F403
from tasks.uncertainty import *  # noqa: F401,F403
