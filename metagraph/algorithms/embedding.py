import metagraph as mg
from metagraph import abstract_algorithm
from metagraph.types import Graph, Matrix, Vector, NodeMap, GraphSageNodeEmbedding
from typing import Tuple


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
) -> Tuple[Matrix, NodeMap]:
    pass  # pragma: no cover


@abstract_algorithm("embedding.train.graph2vec")
def graph2vec_train(
    graphs: mg.List[Graph(edge_type="set", is_directed=False)],
    subgraph_degree: int,
    embedding_size: int,
    epochs: int,
    learning_rate: float,
) -> Matrix:
    pass  # pragma: no cover


@abstract_algorithm("embedding.train.graphwave")
def graphwave_train(
    graph: Graph(edge_type="set", is_directed=False),
    scales: Vector,
    sample_point_count: int,
    sample_point_max: float,
    chebyshev_degree: int,
) -> Tuple[Matrix, NodeMap]:
    pass  # pragma: no cover


@abstract_algorithm("embedding.train.hope.katz")
def hope_katz_train(
    graph: Graph(edge_type="map", is_directed=True), embedding_size: int, beta: float
) -> Tuple[Matrix, NodeMap]:
    # embedding_size is ideally even since HOPE learns 2 embedding vectors of size embedding_size // 2 and concatenates them
    pass  # pragma: no cover


@abstract_algorithm("embedding.train.graph_sage.mean")
def graph_sage_mean_train(
    graph: Graph(edge_type="map", is_directed=True),
    node_features: Matrix,
    node2row: NodeMap,
    walk_length: int,
    walks_per_node: int,
    layer_sizes: Vector,
    samples_per_layer: Vector,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> GraphSageNodeEmbedding:
    # returned embeddings will have size equal to layer_sizes[-1]
    pass  # pragma: no cover


@abstract_algorithm("embedding.train.line")
def line_train(
    graph: Graph,
    walks_per_node: int,
    negative_sample_count: int,
    embedding_size: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> Tuple[Matrix, NodeMap]:
    pass  # pragma: no cover
