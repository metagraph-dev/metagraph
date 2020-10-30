from metagraph.tests.util import default_plugin_resolver
import networkx as nx
import numpy as np
import math

from . import MultiVerify

euclidean_dist = lambda a, b: np.linalg.norm(a - b)


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
    p = 1.0
    q = 0.5
    walks_per_node = 12
    walk_length = 12
    embedding_size = 25
    epochs = 15
    learning_rate = 5e-2

    def cmp_func(matrix_node_map_pair):
        matrix, node_map = matrix_node_map_pair
        a_indices = node_map[a_nodes]
        b_indices = node_map[b_nodes]
        np_matrix = matrix.as_dense(copy=False)
        a_centroid = np_matrix[a_indices].mean(0)
        b_centroid = np_matrix[b_indices].mean(0)
        for a_index in a_indices:
            for b_index in b_indices:
                a_vector = np_matrix[a_index]
                b_vector = np_matrix[b_index]
                a_to_a_center = euclidean_dist(a_vector, a_centroid)
                b_to_b_center = euclidean_dist(b_vector, b_centroid)
                a_to_b = euclidean_dist(a_vector, b_vector)

                # TODO consider using a hard-coded constant here for a more robust test
                assert a_to_a_center < a_to_b
                assert b_to_b_center < a_to_b

    MultiVerify(dpr).compute(
        "embedding.train.node2vec",
        graph,
        p,
        q,
        walks_per_node,
        walk_length,
        embedding_size,
        epochs,
        learning_rate,
    ).normalize(
        (dpr.types.Matrix.NumpyMatrixType, dpr.types.NodeMap.NumpyNodeMapType)
    ).custom_compare(
        cmp_func
    )


def test_graphwave(default_plugin_resolver):
    dpr = default_plugin_resolver

    complete_graph_size = 10
    path_length = 11
    nx_graph = nx.barbell_graph(m1=complete_graph_size, m2=path_length)

    a_start_node = 0
    path_start_node = complete_graph_size
    b_start_node = path_length + complete_graph_size

    a_nodes = np.arange(a_start_node, path_start_node)
    path_nodes = np.arange(path_start_node, b_start_node)
    b_nodes = np.arange(b_start_node, len(nx_graph.nodes))

    node_to_class = np.empty(len(nx_graph.nodes))
    node_to_class[a_nodes] = 0
    node_to_class[b_nodes] = 0
    path_middle_node = path_start_node + math.ceil((b_start_node - path_start_node) / 2)
    for i, path_node_id in enumerate(range(path_start_node, path_middle_node), start=1):
        class_label = i
        node_to_class[path_node_id] = class_label
        node_to_class[b_start_node - i] = class_label
    classes = np.unique(node_to_class)

    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)

    def cmp_func(matrix_node_map_pair):
        matrix, node_map = matrix_node_map_pair
        np_matrix = matrix.as_dense(copy=False)

        class_to_vectors = {c: [] for c in classes}
        for node_id in node_map.pos2id:
            c = node_to_class[node_id]
            class_to_vectors[c].append(np_matrix[node_id])

        class_to_mean_vector = {
            c: np.stack(vectors).mean(axis=0) for c, vectors in class_to_vectors.items()
        }

        class_to_max_dist_from_mean = {
            c: max(
                map(
                    lambda vector: euclidean_dist(vector, class_to_mean_vector[c]),
                    vectors,
                )
            )
            for c, vectors in class_to_vectors.items()
        }

        for c, max_dist in class_to_max_dist_from_mean.items():
            assert max_dist < 0.325

    scales = dpr.wrappers.Vector.NumpyVector(np.array([5, 10]))
    sample_point_count = 50
    sample_point_max = 100.0
    chebyshev_degree = 20
    MultiVerify(dpr).compute(
        "embedding.train.graphwave",
        graph,
        scales,
        sample_point_count,
        sample_point_max,
        chebyshev_degree,
    ).normalize(
        (dpr.types.Matrix.NumpyMatrixType, dpr.types.NodeMap.NumpyNodeMapType)
    ).custom_compare(
        cmp_func
    )


# def test_hope_katz(default_plugin_resolver):
#     dpr = default_plugin_resolver

#     # make two unidirectional circle graphs connected by one node
#     a_nodes = np.arange(10)
#     b_nodes = np.arange(10, 20)

#     nx_graph = nx.DiGraph()

#     for i in range(9):
#         a_src_node = a_nodes[i]
#         a_dst_node = a_nodes[i+1]
#         b_src_node = b_nodes[i]
#         b_dst_node = b_nodes[i+1]
#         nx_graph.add_edge(a_src_node, a_dst_node, weight = 25)
#         nx_graph.add_edge(b_src_node, b_dst_node, weight = 25)

#     nx_graph.add_edge(0, 1_000, weight = 1)
#     nx_graph.add_edge(10, 1_000, weight = 1)

#     graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)

#     mv = MultiVerify(dpr)

#     embedding_size = 10
#     embedding = mv.compute(
#         "embedding.train.hope.katz",
#         graph,
#         embedding_size,
#     )

#     def cmp_func(embedding):
#         np_matrix = embedding.matrix.as_dense(copy=False)

#         mean_a_vector = np.stack(np_matrix[embedding.nodes[a_node]] for a_node in a_nodes).mean(axis=0)
#         mean_b_vector = np.stack(np_matrix[embedding.nodes[b_node]] for b_node in b_nodes).mean(axis=0)

#         max_dist_from_mean_a = max(euclidean_dist(mean_a_vector, np_matrix[embedding.nodes[a_node]]) for a_node in a_nodes)
#         max_dist_from_mean_b = max(euclidean_dist(mean_b_vector, np_matrix[embedding.nodes[b_node]]) for b_node in b_nodes)

#         print(f"max_dist_from_mean_a {repr(max_dist_from_mean_a)}")
#         print(f"max_dist_from_mean_b {repr(max_dist_from_mean_b)}")

#         assert False

#     embedding.normalize(dpr.types.NodeEmbedding.NumpyNodeEmbeddingType).custom_compare(
#         cmp_func
#     )
