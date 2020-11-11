from metagraph.tests.util import default_plugin_resolver
import pytest
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
        a_centroid = matrix[a_indices].mean(0)
        b_centroid = matrix[b_indices].mean(0)
        for a_index in a_indices:
            for b_index in b_indices:
                a_vector = matrix[a_index]
                b_vector = matrix[b_index]
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


def test_graph2vec(default_plugin_resolver):
    try:
        from sklearn.mixture import GaussianMixture
    except ModuleNotFoundError:
        pytest.skip("scikit-learn not installed.")

    dpr = default_plugin_resolver
    complete_graphs = [
        nx.complete_graph(np.arange(i * 30, i * 30 + 20)) for i in range(12)
    ]
    path_graphs = [nx.path_graph(10 * (i + 1)) for i in range(12)]
    random_graphs = [nx.connected_watts_strogatz_graph(30, 10, 0.5) for i in range(12)]
    nx_graphs = complete_graphs + path_graphs + random_graphs
    graphs = list(map(dpr.wrappers.Graph.NetworkXGraph, nx_graphs))
    subgraph_degree = 10
    embedding_size = 12
    epochs = 10
    learning_rate = 5e-2

    def cmp_func(matrix):
        gmm = GaussianMixture(3)
        predicted_labels = gmm.fit_predict(matrix)

        complete_graph_labels = predicted_labels[0 : len(complete_graphs)]
        path_graph_labels = predicted_labels[
            len(complete_graphs) : len(complete_graphs) + len(random_graphs)
        ]
        random_graph_labels = predicted_labels[
            len(complete_graphs) + len(random_graphs) :
        ]

        assert len(np.unique(complete_graph_labels)) == 1
        assert len(np.unique(path_graph_labels)) == 1
        assert len(np.unique(random_graph_labels)) == 1

        complete_graph_label = complete_graph_labels[0]
        path_graph_label = path_graph_labels[0]
        random_graph_label = random_graph_labels[0]

        assert complete_graph_label != path_graph_label != random_graph_label

    MultiVerify(dpr).compute(
        "embedding.train.graph2vec",
        graphs,
        subgraph_degree,
        embedding_size,
        epochs,
        learning_rate,
    ).normalize(dpr.types.Matrix.NumpyMatrixType).custom_compare(cmp_func)


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

        class_to_vectors = {c: [] for c in classes}
        for node_id in node_map.pos2id:
            c = node_to_class[node_id]
            class_to_vectors[c].append(matrix[node_id])

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

    scales = np.array([5, 10])
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


def test_hope_katz(default_plugin_resolver):
    try:
        from sklearn.mixture import GaussianMixture
    except ModuleNotFoundError:
        pytest.skip("scikit-learn not installed.")

    dpr = default_plugin_resolver

    # Random Graph Generation Parameters

    graph_size = 100
    erdos_renyi_probability = 0.9

    # Generate Graph (two Erdos-Renyi graphs connected by bridge)

    a_graph = nx.erdos_renyi_graph(graph_size, p=erdos_renyi_probability, directed=True)
    a_end_node = max(a_graph.nodes())

    b_graph = nx.erdos_renyi_graph(
        len(a_graph.nodes()), p=erdos_renyi_probability, directed=True
    )
    b_graph = nx.relabel_nodes(
        b_graph, {i: i + len(a_graph.nodes()) * 2 for i in a_graph.nodes()}
    )
    b_end_node = max(b_graph.nodes())

    nx_graph = nx.compose(a_graph, b_graph)

    number_of_bridging_nodes = 5
    assert number_of_bridging_nodes < len(a_graph.nodes)
    for start_node_delta in range(number_of_bridging_nodes):
        nx_graph.add_edge(
            a_end_node + start_node_delta, a_end_node + start_node_delta + 1
        )
        nx_graph.add_edge(
            b_end_node + start_node_delta, b_end_node + start_node_delta + 1
        )

    center_node_id = max(nx_graph.nodes()) * 2
    nx_graph.add_edge(a_end_node + number_of_bridging_nodes, center_node_id)
    nx_graph.add_edge(b_end_node + number_of_bridging_nodes, center_node_id)

    # Verify Results

    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)
    embedding_size = 24
    beta = 0.1

    def cmp_func(matrix_node_map_pair):
        matrix, node_map = matrix_node_map_pair

        a_indices = node_map[a_graph.nodes]
        b_indices = node_map[b_graph.nodes]

        gmm = GaussianMixture(2)
        predicted_labels = gmm.fit_predict(matrix)
        a_labels = predicted_labels[a_indices]
        b_labels = predicted_labels[b_indices]

        a_histogram = np.histogram(a_labels, bins=[0, 1])[0]
        b_histogram = np.histogram(b_labels, bins=[0, 1])[0]

        assert any(a_histogram > 95)
        assert any(b_histogram > 95)

        a_label = np.argmax(a_histogram)
        b_label = np.argmax(b_histogram)
        a_variances = np.sum(gmm.covariances_[a_label] * np.eye(embedding_size), axis=0)
        b_variances = np.sum(gmm.covariances_[b_label] * np.eye(embedding_size), axis=0)

        assert a_variances.max() < 0.20
        assert b_variances.max() < 0.20

    MultiVerify(dpr).compute(
        "embedding.train.hope.katz", graph, embedding_size, beta,
    ).normalize(
        (dpr.types.Matrix.NumpyMatrixType, dpr.types.NodeMap.NumpyNodeMapType)
    ).custom_compare(
        cmp_func
    )


def test_graph_sage_mean(default_plugin_resolver):
    """
== Training Subgraph ==

[Training Subgraph A (fully connected), nodes 0..9] --------------|
                           |                                      |
                    node 9999_09_10                               |
                           |                                      |
[Training Subgraph B (fully connected), nodes 10..19]     node 9999_29_00
                           |                                      |
                    node 9999_19_20                               |
                           |                                      |
[Training Subgraph C (fully connected), nodes 10..19] -------------

Training Subgraph A nodes all have feature vector [1, 0, 0, 0, 0, ..., 0]
Training Subgraph B nodes all have feature vector [0, 1, 0, 0, 0, ..., 0]
Training Subgraph C nodes all have feature vector [0, 0, 1, 0, 0, ..., 0]
Nodes 9999_09_10, 9999_19_20, and node 9999_29_00 have the zero vector as their node features.



== Testing Subgraph ==

[Testing Subgraph A (fully connected), nodes 8888_00..8888_19]
                        |
                 node 8888_00_20
                        |
[Testing Subgraph B (fully connected), nodes 8888_20..8888_49]

Testing Subgraph A nodes all have feature vector [1, 0, 0, 0, 0, ..., 0]
Testing Subgraph B nodes all have feature vector [0, 1, 0, 0, 0, ..., 0]
Node 8888_00_20 hsa the zero vector as a its node features.



== Differences Between Training & Testing Graphs ==

All the complete subgraphs in the training graph have 10 nodes, but the complete subgraphs in the testing graph do NOT.

The test verifies for the testing graph that the 20 nearest neighbors in the embedding space of each node are all part of the same complete subgraph.
    """

    try:
        from sklearn.neighbors import NearestNeighbors
    except ModuleNotFoundError:
        pytest.skip("scikit-learn not installed.")

    if "metagraph_stellargraph" not in dir(default_plugin_resolver.plugins):
        pytest.skip("metagraph_stellargraph not installed.")

    dpr = default_plugin_resolver

    # Generate Training Graph
    a_nodes = np.arange(10)
    b_nodes = np.arange(10, 20)
    c_nodes = np.arange(20, 30)

    complete_graph_a = nx.complete_graph(a_nodes)
    complete_graph_b = nx.complete_graph(b_nodes)
    complete_graph_c = nx.complete_graph(c_nodes)

    nx_graph = nx.compose(
        nx.compose(complete_graph_a, complete_graph_b), complete_graph_c
    )
    nx_graph.add_edge(9999_09_10, 9)
    nx_graph.add_edge(9999_09_10, 10)
    nx_graph.add_edge(9999_19_20, 19)
    nx_graph.add_edge(9999_19_20, 20)
    nx_graph.add_edge(9999_29_00, 29)
    nx_graph.add_edge(9999_29_00, 0)

    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)

    mv = MultiVerify(dpr)

    embedding_size = 50
    node_feature_nodes = dpr.wrappers.NodeMap.NumpyNodeMap(
        np.arange(33),
        node_ids=np.concatenate(
            [np.arange(30), np.array([9999_09_10, 9999_19_20, 9999_29_00])]
        ),
    )
    node_feature_np_matrix = np.zeros([33, embedding_size])
    node_feature_np_matrix[a_nodes, 0] = 1
    node_feature_np_matrix[b_nodes, 1] = 1
    node_feature_np_matrix[c_nodes, 2] = 1
    node_feature_np_matrix[30:] = np.ones(embedding_size)
    node_feature_matrix = dpr.wrappers.Matrix.NumpyMatrix(node_feature_np_matrix)

    # Run GraphSAGE
    walk_length = 5
    walks_per_node = 1
    layer_sizes = dpr.wrappers.Vector.NumpyVector(np.array([40, 30]))
    samples_per_layer = dpr.wrappers.Vector.NumpyVector(np.array([10, 5]))
    epochs = 35
    learning_rate = 5e-3
    batch_size = 2

    assert len(layer_sizes) == len(samples_per_layer)

    embedding = mv.compute(
        "embedding.train.graph_sage.mean",
        graph,
        node_feature_matrix,
        node_feature_nodes,
        walk_length,
        walks_per_node,
        layer_sizes,
        samples_per_layer,
        epochs,
        learning_rate,
        batch_size,
    ).normalize(dpr.types.GraphSageNodeEmbedding.StellarGraphGraphSageNodeEmbeddingType)

    # Create Testing Graph
    unseen_a_nodes = np.arange(8888_00, 8888_20)
    unseen_b_nodes = np.arange(8888_20, 8888_50)
    unseen_complete_graph_a = nx.complete_graph(unseen_a_nodes)
    unseen_complete_graph_b = nx.complete_graph(unseen_b_nodes)
    unseen_nx_graph = nx.compose(unseen_complete_graph_a, unseen_complete_graph_b)
    unseen_nx_graph.add_edge(8888_00_20, 8888_00)
    unseen_nx_graph.add_edge(8888_00_20, 8888_20)
    unseen_node_feature_np_matrix = np.zeros([51, embedding_size])
    unseen_node_feature_np_matrix[0:20, 0] = 1
    unseen_node_feature_np_matrix[20:50, 1] = 1
    unseen_node_feature_matrix = dpr.wrappers.Matrix.NumpyMatrix(
        unseen_node_feature_np_matrix
    )
    unseen_node_feature_nodes = dpr.wrappers.NodeMap.NumpyNodeMap(
        np.arange(51),
        node_ids=np.concatenate(
            [unseen_a_nodes, unseen_b_nodes, np.array([8888_00_20])]
        ),
    )
    unseen_graph = dpr.wrappers.Graph.NetworkXGraph(unseen_nx_graph)
    matrix = mv.transform(
        dpr.plugins.metagraph_stellargraph.algos.util.graph_sage_node_embedding.apply,
        embedding,
        unseen_graph,
        unseen_node_feature_matrix,
        unseen_node_feature_nodes,
        batch_size=batch_size,
        worker_count=1,
    )

    # Verify GraphSAGE results
    def cmp_func(matrix):
        assert tuple(matrix.shape) == (51, layer_sizes[-1])
        unseen_a_vectors = matrix[0:20]
        unseen_b_vectors = matrix[20:50]

        _, neighbor_indices = (
            NearestNeighbors(n_neighbors=20).fit(matrix).kneighbors(matrix)
        )

        for unseen_a_node_position in range(20):
            unseen_a_node_neighbor_indices = neighbor_indices[unseen_a_node_position]
            for unseen_a_node_neighbor_index in unseen_a_node_neighbor_indices:
                assert 0 <= unseen_a_node_neighbor_index < 20

        for unseen_b_node_position in range(20, 50):
            unseen_b_node_neighbor_indices = neighbor_indices[unseen_b_node_position]
            for unseen_b_node_neighbor_index in unseen_b_node_neighbor_indices:
                assert 20 <= unseen_b_node_neighbor_index < 50

    matrix.normalize(dpr.types.Matrix.NumpyMatrixType).custom_compare(cmp_func)


def test_line(default_plugin_resolver):
    try:
        from sklearn.mixture import GaussianMixture
    except ModuleNotFoundError:
        pytest.skip("scikit-learn not installed.")

    dpr = default_plugin_resolver

    # Graph Generation Parameters

    layer_sizes = [100, 20, 10, 20, 100, 20, 10, 20, 100]

    # Generate Graph

    nx_graph = nx.Graph()

    all_nodes = list(range(sum(layer_sizes)))

    for node in all_nodes:
        nx_graph.add_node(node)

    layer_index_to_end_indices = np.cumsum(layer_sizes).tolist()
    layer_index_to_start_indices = [0] + layer_index_to_end_indices[:-1]
    layer_index_to_nodes = [
        all_nodes[start:end]
        for start, end in zip(layer_index_to_start_indices, layer_index_to_end_indices)
    ]
    assert layer_sizes == list(map(len, layer_index_to_nodes))

    for layer_index, layer_nodes in enumerate(layer_index_to_nodes[:-1]):
        next_layer_nodes = layer_index_to_nodes[layer_index + 1]
        for layer_node in layer_nodes:
            for next_layer_node in next_layer_nodes:
                nx_graph.add_edge(layer_node, next_layer_node)

    assert nx.is_connected(nx_graph)

    def cmp_func(matrix_node_map_pair):
        matrix, node_map = matrix_node_map_pair

        a_nodes = layer_index_to_nodes[0]
        b_nodes = layer_index_to_nodes[4]
        c_nodes = layer_index_to_nodes[8]

        a_indices = node_map[a_nodes]
        b_indices = node_map[b_nodes]
        c_indices = node_map[c_nodes]

        gmm = GaussianMixture(3)
        predicted_labels = gmm.fit_predict(matrix)
        a_labels = predicted_labels[a_indices]
        b_labels = predicted_labels[b_indices]
        c_labels = predicted_labels[c_indices]

        assert len(np.unique(a_labels)) == 1
        assert len(np.unique(b_labels)) == 1
        assert len(np.unique(c_labels)) == 1

        a_label = a_labels[0]
        b_label = b_labels[0]
        c_label = c_labels[0]
        a_variances = np.sum(gmm.covariances_[a_label] * np.eye(embedding_size), axis=0)
        b_variances = np.sum(gmm.covariances_[b_label] * np.eye(embedding_size), axis=0)
        c_variances = np.sum(gmm.covariances_[c_label] * np.eye(embedding_size), axis=0)

        assert a_variances.max() < 0.15
        assert b_variances.max() < 0.15
        assert c_variances.max() < 0.15

    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)
    walks_per_node = 8
    negative_sample_count = 5
    embedding_size = 10
    epochs = 10
    learning_rate = 0.25
    batch_size = 1

    MultiVerify(dpr).compute(
        "embedding.train.line",
        graph,
        walks_per_node,
        negative_sample_count,
        embedding_size,
        epochs,
        learning_rate,
        batch_size,
    ).normalize(
        (dpr.types.Matrix.NumpyMatrixType, dpr.types.NodeMap.NumpyNodeMapType)
    ).custom_compare(
        cmp_func
    )
