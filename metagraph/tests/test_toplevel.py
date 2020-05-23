import pytest

import metagraph as mg
from metagraph.core.resolver import Resolver, Namespace


def test_version():
    assert isinstance(mg.__version__, str)


def test_lazy_attributes():
    for attr in ["resolver", "translate", "typeclass_of", "algos"]:
        assert hasattr(mg, attr)

    with pytest.raises(TypeError, match="does not have a registered type"):
        mg.typeclass_of(set())

    with pytest.raises(TypeError, match="does not have a registered type"):
        mg.type_of(set())

    with pytest.raises(TypeError, match="does not have a registered type"):
        mg.translate(set(), "unknown type")

    assert isinstance(mg.resolver, Resolver)
    assert isinstance(mg.algos, Namespace)


def test_dir():
    del mg.resolver  # make it seem like it was not loaded
    assert set(
        ["resolver", "translate", "typeclass_of", "type_of", "algos", "AbstractType"]
    ).issubset(dir(mg))
    mg.resolver  # trigger resolver init
    assert set(
        ["resolver", "translate", "typeclass_of", "type_of", "algos", "AbstractType"]
    ).issubset(dir(mg))
