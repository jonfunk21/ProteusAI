# This source code is part of the proteusAI package and is distributed
# under the MIT License.


from importlib import metadata

__version__ = metadata.version("proteusAI")
__name__ = "proteusAI"
__author__ = "Jonathan Funk"

from .Protein import *  # noqa: F403
from .Library import *  # noqa: F403
from .Model import *  # noqa: F403
