from metagraph import abstract_algorithm
from metagraph.types import Graph, NodeEmbedding, Vector


@abstract_algorithm("embedding.train.node2vec")
def node2vec_train(
    graph: Graph,
    p: float,
    q: float,
    walks_per_node: int,
    walk_length: int,
    embedding_size: int,
    epochs: int,
    learning_rate: float,
) -> NodeEmbedding:
    pass  # pragma: no cover


@abstract_algorithm("embedding.train.graphwave")
def graphwave_train(
    graph: Graph(edge_type="set", is_directed=False),
    scales: Vector,
    sample_point_count: int,
    sample_point_max: float,
    chebyshev_degree: int,
) -> NodeEmbedding:
    pass  # pragma: no cover
