from metagraph.tests.util import default_plugin_resolver
import networkx as nx
import numpy as np

from . import MultiVerify


def test_node2vec(default_plugin_resolver):
    dpr = default_plugin_resolver

    # make uneven barbell graph
    a_nodes = np.arange(10)
    b_nodes = np.arange(80, 100)
    complete_graph_a = nx.complete_graph(a_nodes)
    complete_graph_b = nx.complete_graph(b_nodes)
    nx_graph = nx.compose(complete_graph_a, complete_graph_b)
    for node in range(9, 50):
        nx_graph.add_edge(node, node + 1)
    nx_graph.add_edge(50, 80)  # have non-consecutive node ids
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)

    mv = MultiVerify(dpr)

    p = 1.0
    q = 0.5
    walks_per_node = 8
    walk_length = 8
    embedding_size = 25
    epochs = 10_000
    learning_rate = 1e-3
    embedding = mv.compute(
        "embedding.train.node2vec",
        graph,
        p,
        q,
        walks_per_node,
        walk_length,
        embedding_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    euclidean_dist = lambda a, b: np.linalg.norm(a - b)

    def cmp_func(embedding):
        a_indices = list(range(10))
        b_indices = list(range(51, 71))
        np_matrix = embedding.matrix.as_dense(copy=False)
        a_centroid = np_matrix[a_indices].mean(0)
        b_centroid = np_matrix[b_indices].mean(0)

        for a_index in a_indices:
            for b_index in b_indices:
                a_vector = embedding.matrix[a_index]
                b_vector = embedding.matrix[b_index]

                a_to_a_center = euclidean_dist(a_vector, a_centroid)
                b_to_b_center = euclidean_dist(b_vector, b_centroid)
                a_to_b = euclidean_dist(a_vector, b_vector)

                assert a_to_a_center < a_to_b
                assert a_to_a_center < a_to_b

    embedding.normalize(dpr.types.NodeEmbedding.NumpyNodeEmbeddingType).custom_compare(
        cmp_func
    )
