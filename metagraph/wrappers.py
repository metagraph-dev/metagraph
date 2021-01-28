from metagraph.plugins.core.wrappers import *

# Note: this file exists solely to allow this:
# >>> from metagraph.wrappers import NodeMapWrapper
#
# However, once the default resolver has been created, metagraph.wrappers
# is overwritten to point to metagraph.resolver.wrappers
# The result is that `metagraph.wrappers.NodeMapWrapper` will no longer exist.
# Instead, something like `metagraph.wrappers.NodeMap.NumpyNodeMap` will work.
