from metagraph.core.plugin import ConcreteAlgorithm, ConcreteType


class DelayedAlgo:
    def __init__(
        self, algo: ConcreteAlgorithm, result_type: ConcreteType, resolver: "Resolver"
    ):
        self.algo = algo
        self.resolver = resolver
        self.result_type = result_type

    def __call__(self, args, kwargs):
        algo = self.algo
        if algo._include_resolver or algo._compiler:
            # do not mutate the kwargs
            kwargs = kwargs.copy()
            kwargs["resolver"] = self.resolver
        return self.algo(*args, **kwargs)
