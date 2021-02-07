from metagraph.core.plugin import ConcreteAlgorithm, ConcreteType, Translator
from typing import Callable


class MetagraphTask:
    def __init__(self, callable: Callable, result_type: ConcreteType):
        self.callable = callable
        self.result_type = result_type

    def __call__(self, *args, **kwargs):
        return self.callable(*args, **kwargs)

    @property
    def func_label(self):
        return self.callable.__name__

    @property
    def data_label(self):
        return self.result_type.__name__


class DelayedAlgo(MetagraphTask):
    def __init__(
        self, algo: ConcreteAlgorithm, result_type: ConcreteType, resolver: "Resolver"
    ):
        self.algo = algo
        self.resolver = resolver

        def call(args, kwargs):
            if algo._include_resolver or algo._compiler:
                # do not mutate the kwargs
                kwargs = kwargs.copy()
                kwargs["resolver"] = resolver
            return algo(*args, **kwargs)

        super().__init__(callable=call, result_type=result_type)

    @property
    def func_label(self):
        return f"{self.algo.abstract_name}\n({self.algo.__name__})"


class DelayedTranslate(MetagraphTask):
    def __init__(
        self, translator: Translator, source_type, result_type, resolver: "Resolver"
    ):
        self.translator = translator
        self.resolver = resolver
        self.source_type = source_type

        def call(args, kwargs):
            if translator._include_resolver:
                # do not mutate the kwargs
                kwargs = kwargs.copy()
                kwargs["resolver"] = resolver
            return translator(*args, **kwargs)

        super().__init__(callable=call, result_type=result_type)

    @property
    def func_label(self):
        return f"{self.source_type.__name__}->{self.result_type.__name__}\n({self.translator.__name__})"
