# This source code is part of the proteusAI package and is distributed
# under the MIT License.


from importlib import metadata

__version__ = metadata.version("proteusAI")
__name__ = "proteusAI"
__author__ = "Jonathan Funk"

from .Protein.protein import Protein
from .Library.library import Library
from .Model.model import Model