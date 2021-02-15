from metagraph.core.plugin import ConcreteAlgorithm, ConcreteType, Translator
from typing import Callable, List


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
    def __init__(self, algo: ConcreteAlgorithm, result_type: ConcreteType):
        self.algo = algo

        def call(args, kwargs):
            return algo(*args, **kwargs)

        super().__init__(callable=call, result_type=result_type)

    @property
    def func_label(self):
        return f"{self.algo.abstract_name}\n({self.algo.__name__})"


class DelayedJITAlgo(MetagraphTask):
    def __init__(
        self,
        func: Callable,
        compiler: str,
        source_algos: List[ConcreteAlgorithm],
        result_type: ConcreteType,
    ):
        self.source_algos = source_algos
        self.compiler = compiler
        super().__init__(callable=func, result_type=result_type)

    @property
    def func_label(self):
        label = f"{self.compiler} fused:\n"
        for algo in self.source_algos:
            label += f" {algo.__name__}\n"
        return label


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
