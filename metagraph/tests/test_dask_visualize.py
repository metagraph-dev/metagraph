import numpy as np
from dask.utils import tmpfile
from metagraph.tests.compiler.test_subgraphs import res
from metagraph.core.dask.visualize import visualize, merge_dict_of_dict


def test_merge_dict_of_dict():
    base = {"key1": {"attr1": 1, "attr2": 2}, "key2": {"attr1": 11, "attr3": 13,}}
    overlay = {
        "key1": {"attr1": 100, "attr3": 300,},
        "key3": {"attr1": 1000, "attr4": 4000,},
    }

    result = merge_dict_of_dict(base, overlay)
    expected = {
        "key1": {"attr1": 100, "attr2": 2, "attr3": 300,},
        "key2": {"attr1": 11, "attr3": 13,},
        "key3": {"attr1": 1000, "attr4": 4000,},
    }
    assert result == expected


def test_visualize(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    z1 = scale_func(scale_func(scale_func(a, 2.0), 3.0), 4.0)
    z2 = scale_func(scale_func(scale_func(a, 2.5), 3.5), 4.5)
    merge = res.algos.testing.add(z1, z2)
    ans = scale_func(merge, 2.8)

    with tmpfile(extension="dot") as fn:
        visualize(ans, filename=fn)
        with open(fn) as f:
            contents = f.read()
        # this is an inadequate test, but at least confirms some basic are working
        assert "testing.scale" in contents
        assert "testing.add" in contents
        assert "NumpyVectorType" in contents

    with tmpfile(extension="dot") as fn:
        visualize(ans, collapse_outputs=True, filename=fn)
        with open(fn) as f:
            contents = f.read()
        assert "testing.scale" in contents
        assert "testing.add" in contents
        # data nodes should be hidden
        assert "NumpyVectorType" not in contents


def test_placeholder_visualize(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    z1 = scale_func(scale_func(scale_func(a, 2.0), 3.0), 4.0)

    with tmpfile(extension="dot") as fn:
        z1.visualize(filename=fn)
        with open(fn) as f:
            contents = f.read()
        assert "testing.scale" in contents


def test_optimize_and_visualize(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    z1 = scale_func(scale_func(scale_func(a, 2.0), 3.0), 4.0)
    z2 = scale_func(scale_func(scale_func(a, 2.5), 3.5), 4.5)
    merge = res.algos.testing.add(z1, z2)
    ans = scale_func(merge, 2.8)

    with tmpfile(extension="dot") as fn:
        visualize(ans, filename=fn, optimize_graph=True)
        with open(fn) as f:
            contents = f.read()
        assert "identity_comp fused" in contents
