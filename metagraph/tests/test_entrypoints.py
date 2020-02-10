import sys
import os
import pytest

import metagraph.core.entrypoints
from metagraph.core.plugin import (
    AbstractType,
    ConcreteType,
    Translator,
    AbstractAlgorithm,
    ConcreteAlgorithm,
)


@pytest.fixture
def site_dir():
    test_site_dir = os.path.join(os.path.dirname(__file__), "site_dir")
    sys.path.insert(0, test_site_dir)
    yield test_site_dir
    sys.path.remove(test_site_dir)


KINDS = {
    "abstract_types": (issubclass, AbstractType),
    "concrete_types": (issubclass, ConcreteType),
    "translators": (isinstance, Translator),
    "abstract_algorithms": (isinstance, AbstractAlgorithm),
    "concrete_algorithms": (isinstance, ConcreteAlgorithm),
}


def test_find_plugin_loaders(site_dir):
    for kind, (test_func, kind_class) in KINDS.items():
        loaders = metagraph.core.entrypoints.find_plugin_loaders(kind)
        for loader in loaders:
            plugins = loader()
            assert len(plugins) > 0
            for obj in plugins:
                assert test_func(obj, kind_class)


def test_load_plugins(site_dir):
    for kind, (test_func, kind_class) in KINDS.items():
        plugins = metagraph.core.entrypoints.load_plugins(kind)
        assert len(plugins) > 0
        for obj in plugins:
            assert test_func(obj, kind_class)
