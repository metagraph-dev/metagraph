import pytest
import metagraph as mg
from metagraph.explorer import api, service
from metagraph.tests.util import default_plugin_resolver


def test_normalize_abstract_type(default_plugin_resolver):
    dpr = default_plugin_resolver
    result = api.normalize_abstract_type(dpr, "Graph")
    assert result == ("Graph", mg.plugins.core.types.Graph)
    result2 = api.normalize_abstract_type(dpr, mg.plugins.core.types.NodeMap)
    assert result2 == ("NodeMap", mg.plugins.core.types.NodeMap)
    with pytest.raises(ValueError, match="Unknown abstract type: foobar"):
        api.normalize_abstract_type(dpr, "foobar")


def test_normalize_concrete_type(default_plugin_resolver):
    dpr = default_plugin_resolver
    result = api.normalize_concrete_type(dpr, "Graph", "NetworkXGraphType")
    assert result == ("NetworkXGraphType", mg.plugins.networkx.types.NetworkXGraph.Type)
    result2 = api.normalize_concrete_type(
        dpr, mg.plugins.core.types.NodeMap, mg.plugins.python.types.PythonNodeMapType
    )
    assert result2 == ("PythonNodeMapType", mg.types.NodeMap.PythonNodeMapType)
    with pytest.raises(
        ValueError,
        match="Mismatch in abstract type provided and abstract type of concrete provided",
    ):
        api.normalize_concrete_type(dpr, "Graph", mg.types.NodeMap.NumpyNodeMapType)
    with pytest.raises(ValueError, match="Unknown concrete type: Graph/NumpyNodeMap"):
        api.normalize_concrete_type(dpr, "Graph", "NumpyNodeMap")


def test_get_plugins(default_plugin_resolver):
    dpr = default_plugin_resolver
    result = api.get_plugins(dpr)
    assert "core_numpy" in result
    assert result["core_numpy"].keys() == {"children"}
    assert result["core_numpy"]["children"].keys() == {
        "Abstract Algorithms",
        "Abstract Types",
        "Concrete Algorithms",
        "Concrete Types",
        "Translators",
        "Wrappers",
    }
    assert "NumpyNodeMap" in result["core_numpy"]["children"]["Wrappers"]["children"]


def test_get_abstract_types(default_plugin_resolver):
    dpr = default_plugin_resolver
    result = api.get_abstract_types(dpr)
    assert type(result) is list
    assert "NodeMap" in result
    assert "Graph" in result


def test_list_types(default_plugin_resolver):
    dpr = default_plugin_resolver
    result = api.list_types(dpr)
    assert "NodeMap" in result
    assert result["NodeMap"].keys() == {"type", "children"}
    assert result["NodeMap"]["type"] == "abstract_type"
    assert "NumpyNodeMapType" in result["NodeMap"]["children"]
    assert result["NodeMap"]["children"]["NumpyNodeMapType"].keys() == {
        "type",
        "children",
    }
    assert result["NodeMap"]["children"]["NumpyNodeMapType"]["type"] == "concrete_type"


def test_list_translators(default_plugin_resolver):
    dpr = default_plugin_resolver
    result = api.list_translators(dpr, "NodeMap")
    assert result.keys() == {
        "primary_types",
        "secondary_types",
        "primary_translators",
        "secondary_translators",
    }
    assert "NumpyNodeMapType -> PythonNodeMapType" in result["primary_translators"]


def test_list_algorithms(default_plugin_resolver):
    dpr = default_plugin_resolver
    result = api.list_algorithms(dpr)
    assert "util" in result
    assert result["util"].keys() == {"type", "children"}
    assert result["util"]["type"] == "path"
    assert (
        result["util"]["children"]["nodemap"]["children"]["select"]["type"]
        == "abstract_algorithm"
    )


def test_list_algorithm_params(default_plugin_resolver):
    dpr = default_plugin_resolver
    r1 = api.list_algorithm_params(dpr, "util.nodemap.select")
    assert r1.keys() == {"parameters", "returns"}
    assert set(r1["parameters"]) == {"x", "nodes"}
    assert r1["parameters"]["x"]["type"] == "NodeMap"

    r2 = api.list_algorithm_params(dpr, "util.graph.build")
    assert set(r2["parameters"]) == {"edges", "nodes"}
    assert set(r2["parameters"]["nodes"]["type"].split(" or ")) == {
        "NodeMap",
        "NodeSet",
        "None",
    }

    r3 = api.list_algorithm_params(dpr, "util.graph.assign_uniform_weight")
    assert r3["parameters"]["weight"]["type"] == "Any"

    r4 = api.list_algorithm_params(dpr, "centrality.hits")
    assert r4["parameters"]["maxiter"]["type"] == "int"
    assert len(r4["returns"]) == 2
    assert r4["returns"][0]["type"] == "NodeMap"
    assert r4["returns"][1]["type"] == "NodeMap"

    r5 = api.list_algorithm_params(dpr, "util.nodemap.apply")
    assert r5["parameters"]["func"]["type"] == "Callable"


def test_solve_translator(default_plugin_resolver):
    dpr = default_plugin_resolver
    r1 = api.solve_translator(
        dpr, "NodeMap", "NumpyNodeMapType", "NodeSet", "PythonNodeSetType"
    )
    assert r1.keys() == {"src_type", "dst_type", "result_type", "solution"}
    assert r1["result_type"] == "direct"

    r2 = api.solve_translator(
        dpr, "NodeMap", "NumpyNodeMapType", "Graph", "GrblasGraphType"
    )
    assert r2["result_type"] == "unsatisfiable"

    r3 = api.solve_translator(
        dpr, "NodeMap", "NumpyNodeMapType", "NodeMap", "NumpyNodeMapType"
    )
    assert r3["result_type"] == "null"

    r4 = api.solve_translator(
        dpr, "NodeMap", "GrblasNodeMapType", "NodeSet", "PythonNodeSetType"
    )
    assert r4["result_type"] == "multi-step"


def test_solve_algorithm(default_plugin_resolver):
    dpr = default_plugin_resolver
    param_info = {
        "x": {"abstract_type": "NodeMap", "concrete_type": "PythonNodeMapType"},
        "nodes": {"abstract_type": "NodeSet", "concrete_type": "PythonNodeSetType"},
    }
    r1 = api.solve_algorithm(dpr, "util.nodemap.select", param_info)
    assert "plan_0" in r1
    assert r1["plan_0"].keys() == {"type", "plan_index", "children"}

    param_info = {
        "graph": {"abstract_type": "Graph", "concrete_type": "NetworkXGraphType"},
        "weight": {"abstract_type": "Any", "concrete_type": "Any"},
    }
    r2 = api.solve_algorithm(dpr, "util.graph.assign_uniform_weight", param_info)
    assert "plan_0" in r2
    assert r2["plan_0"].keys() == {"type", "plan_index", "children"}

    param_info = {
        "graph": {"abstract_type": "Graph", "concrete_type": "NetworkXGraphType"},
        "maxiter": {"abstract_type": "int", "concrete_type": "int"},
        "tolerance": {"abstract_type": "float", "concrete_type": "float"},
        "normalize": {"abstract_type": "bool", "concrete_type": "bool"},
    }
    r3 = api.solve_algorithm(dpr, "centrality.hits", param_info)
    assert "plan_0" in r3
    assert r3["plan_0"].keys() == {"type", "plan_index", "children"}

    param_info = {
        "x": {"abstract_type": "NodeMap", "concrete_type": "PythonNodeMapType"},
        "func": {"abstract_type": "Callable", "concrete_type": "Callable"},
    }
    r4 = api.solve_algorithm(dpr, "util.nodemap.apply", param_info)
    assert "plan_0" in r4
    assert r4["plan_0"].keys() == {"type", "plan_index", "children"}

    with pytest.raises(ValueError, match='No abstract algorithm "fake.algo" exists'):
        api.solve_algorithm(dpr, "fake.algo", {})

    with pytest.raises(ValueError, match="Unhandled type"):
        param_info = {
            "x": {"abstract_type": "foobar", "concrete_type": "foobar"},
            "func": {"abstract_type": "Callable", "concrete_type": "Callable"},
        }
        api.solve_algorithm(dpr, "util.nodemap.apply", param_info)


def test_service(default_plugin_resolver):
    dpr = default_plugin_resolver
    try:
        service._TEST_FLAG = True
        text = dpr.explore()
        assert len(text) > 60000, f"text length is {len(text)}"
        assert "/* Shadow DOM Initializations */" in text
    finally:
        service._TEST_FLAG = False
