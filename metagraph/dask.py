import metagraph as mg
from .core.dask.resolver import DaskResolver

# Wrap the default resolver as a DaskResolver
resolver = DaskResolver(mg.resolver)


del mg
