from metagraph.plugins.core.types import *

# Note: this file exists solely to allow this:
# >>> from metagraph.types import NodeMap
#
# However, once the default resolver has been created, metagraph.types
# is overwritten to point to metagraph.resolver.types
# The result is that `metagraph.types.NodeMap` will refer to a Namespace object,
# not the AbstractType
