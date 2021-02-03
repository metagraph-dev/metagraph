from metagraph.core.exceptions import MetagraphError


class MetagraphPluginError(MetagraphError):
    pass


class ConvergenceError(MetagraphPluginError):
    """
    Indicates an iterative algorithm failed to converge within the
    required convergence limit.
    """

    pass
