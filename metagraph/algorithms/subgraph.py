import metagraph as mg
from metagraph import abstract_algorithm
from metagraph.types import NodeSet, Graph, NodeID


@abstract_algorithm("subgraph.extract_subgraph")
def extract_subgraph(graph: Graph, nodes: NodeSet) -> Graph:
    pass  # pragma: no cover


@abstract_algorithm("subgraph.k_core")
def k_core(graph: Graph(is_directed=False), k: int) -> Graph:
    pass  # pragma: no cover


@abstract_algorithm("subgraph.k_truss")
def k_truss(graph: Graph(is_directed=False), k: int) -> Graph:
    pass  # pragma: no cover


@abstract_algorithm("subgraph.maximal_independent_set")
def maximal_independent_set(graph: Graph) -> NodeSet:
    pass  # pragma: no cover


@abstract_algorithm("subgraph.subisomorphic")
def subisomorphic(graph: Graph, subgraph: Graph) -> bool:
    pass  # pragma: no cover


@abstract_algorithm("subgraph.sample.node_sampling")
def node_sampling(graph: Graph, p: float = 0.20) -> Graph:
    pass  # pragma: no cover


@abstract_algorithm("subgraph.sample.edge_sampling")
def edge_sampling(graph: Graph, p: float = 0.20) -> Graph:
    pass  # pragma: no cover


@abstract_algorithm("subgraph.sample.ties")
def totally_induced_edge_sampling(graph: Graph, p: float = 0.20) -> Graph:
    """
    Totally Induced Edge Sampling method
    https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2743&context=cstech
    """
    pass  # pragma: no cover


@abstract_algorithm("subgraph.sample.random_walk")
def random_walk_sampling(
    graph: Graph,
    num_steps: mg.Optional[int] = None,
    num_nodes: mg.Optional[int] = None,
    num_edges: mg.Optional[int] = None,
    jump_probability: float = 0.15,
    start_node: mg.Optional[NodeID] = None,
) -> Graph:
    """
    Sample using random walks

    Sampling ends when number of steps, nodes, or edges are reached (first to occur if multiple are specified).
    For each step, there is a jump_probability to reset the walk.
    When resetting the walk, if start_node is specified, always reset to this node. If not specified, every reset
        picks a new node in the graph at random.
    """
    # TODO: check that `num_*` variables aren't all `None`
    pass  # pragma: no cover
