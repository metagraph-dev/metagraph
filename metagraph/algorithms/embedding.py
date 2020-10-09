from metagraph import abstract_algorithm
from metagraph.types import Graph, NodeEmbedding


@abstract_algorithm("embedding.train.node2vec")
def node2vec_train(
    graph: Graph,
    p: float,
    q: float,
    walks_per_node: int,
    walk_length: int,
    embedding_size: int,
) -> NodeEmbedding:
    pass  # pragma: no cover
