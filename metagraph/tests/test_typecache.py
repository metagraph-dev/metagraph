import pytest

from metagraph import (
    AbstractType,
    ConcreteType,
    Wrapper,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)
from metagraph.core.resolver import Resolver
from metagraph.core.typecache import TypeCache, TypeInfo

from .util import site_dir, example_resolver
import numpy as np
from collections import defaultdict


def test_typeinfo(example_resolver):
    res = example_resolver
    ct = res.types.MyAbstractType.StrType
    typeinfo = TypeInfo(ct.abstract, {"level": "low"}, ct, {"encoding": "utf-8"})
    assert typeinfo.known_props == {"encoding": "utf-8", "level": "low"}

    typeinfo2 = TypeInfo(ct.abstract, {"level": "medium"}, ct, {"size": 42})
    typeinfo.update_props(typeinfo2)
    assert typeinfo.known_props == {"encoding": "utf-8", "level": "medium", "size": 42}

    with pytest.raises(TypeError, match="other must be TypeInfo"):
        typeinfo.update_props({"foo": True})


def test_typecache_basic(example_resolver):
    typecache = TypeCache()

    obj = np.zeros(2)
    props = dict(a=1)
    typecache[obj] = props

    # automatic removal
    assert typecache[obj] == props
    assert len(typecache) == 1
    assert obj in typecache
    del obj
    assert len(typecache) == 0

    # del
    obj = np.zeros(2)
    typecache[obj] = props
    assert len(typecache) == 1
    del typecache[obj]
    assert len(typecache) == 0
    with pytest.raises(KeyError):
        del typecache[obj]

    # expire
    typecache[obj] = props
    assert len(typecache) == 1
    typecache.expire(obj)
    assert len(typecache) == 0
    # this should not raise an exception
    typecache.expire(obj)

    # Unhandled special object
    dd = defaultdict(dict)
    with pytest.raises(
        TypeError, match="requires special handling which has not been defined yet"
    ):
        typecache[dd] = props
