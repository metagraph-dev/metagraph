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
