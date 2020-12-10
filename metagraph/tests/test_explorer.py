import pytest
import metagraph as mg
from metagraph.explorer import api
from metagraph.tests.util import default_plugin_resolver


def test_normalize_abstract_type(default_plugin_resolver):
    dpr = default_plugin_resolver
    result = api.normalize_abstract_type(dpr, "Graph")
    assert result == ("Graph", mg.types.Graph)
    result2 = api.normalize_abstract_type(dpr, mg.types.NodeMap)
    assert result2 == ("NodeMap", mg.types.NodeMap)


def test_normalize_concrete_type(default_plugin_resolver):
    dpr = default_plugin_resolver
    result = api.normalize_concrete_type(dpr, "Graph", "NetworkXGraphType")
    assert result == ("NetworkXGraphType", mg.plugins.networkx.types.NetworkXGraph.Type)
    result2 = api.normalize_concrete_type(
        dpr, mg.types.NodeMap, mg.plugins.python.types.PythonNodeMapType
    )
    assert result2 == ("PythonNodeMapType", mg.plugins.python.types.PythonNodeMapType)


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
    result = api.list_algorithm_params(dpr, "util.nodemap.select")
    assert result.keys() == {"parameters", "returns"}


def test_solve_translator(default_plugin_resolver):
    dpr = default_plugin_resolver
    result = api.solve_translator(
        dpr, "NodeMap", "NumpyNodeMapType", "NodeSet", "PythonNodeSetType"
    )
    assert result.keys() == {"src_type", "dst_type", "result_type", "solution"}


def test_solve_algorithm(default_plugin_resolver):
    dpr = default_plugin_resolver
    param_info = {
        "x": {"abstract_type": "NodeMap", "concrete_type": "PythonNodeMapType"},
        "nodes": {"abstract_type": "NodeSet", "concrete_type": "PythonNodeSetType"},
    }
    result = api.solve_algorithm(dpr, "util.nodemap.select", param_info)
    assert "plan_0" in result
    assert result["plan_0"].keys() == {"type", "plan_index", "children"}
