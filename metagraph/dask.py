import metagraph as mg
from .core.dask import DaskResolver

# Wrap the default resolver as a DaskResolver
resolver = DaskResolver(mg.resolver)


del mg
