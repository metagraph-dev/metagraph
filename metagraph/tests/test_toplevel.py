import pytest

import metagraph as mg
from metagraph.core.resolver import Resolver, Namespace


def test_version():
    assert isinstance(mg.__version__, str)


def test_lazy_attributes():
    for attr in ["resolver", "translate", "typeclass_of", "algos"]:
        assert hasattr(mg, attr)

    with pytest.raises(TypeError, match="does not have a registered type"):
        mg.typeclass_of([])

    with pytest.raises(TypeError, match="does not have a registered type"):
        mg.type_of([])

    with pytest.raises(TypeError, match="does not have a registered type"):
        mg.translate([], "unknown type")

    assert isinstance(mg.resolver, Resolver)
    assert isinstance(mg.algos, Namespace)


def test_dir():
    del mg.resolver  # make it seem like it was not loaded
    assert {
        "resolver",
        "translate",
        "typeclass_of",
        "type_of",
        "algos",
        "AbstractType",
    }.issubset(dir(mg))
    mg.resolver  # trigger resolver init
    assert {
        "resolver",
        "translate",
        "typeclass_of",
        "type_of",
        "algos",
        "AbstractType",
    }.issubset(dir(mg))
