"""
Count-valued dyad-independent configuration models.

These share a mean structure -- dyad (i, j) has mean ``x_i * y_j`` over an
arbitrary dyad set -- and differ only in the dyad distribution wrapped around
it. See :mod:`maxent_graph.counts.layout` for the bipartite / undirected /
directed bookkeeping and :mod:`maxent_graph.counts.base` for the common
interface.
"""

from .base import DyadModel as DyadModel
from .layout import DyadLayout as DyadLayout
