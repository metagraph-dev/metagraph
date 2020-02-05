from metagraph.plugin import AbstractType, ConcreteType, Translator


def abstract_type():
    return [AbstractType(name="hypergraph")]


def concrete_type():
    return [
        ConcreteType(abstract="hypergraph", name="hypergraph_cpu"),
        ConcreteType(abstract="hypergraph", name="hypergraph_gpu"),
    ]


def translator():
    return [Translator("hypergraph_cpu", "hypergraph_gpu")]
