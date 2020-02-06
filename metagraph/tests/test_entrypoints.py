import sys
import os
import pytest

import metagraph.entrypoints
from metagraph.plugin import AbstractType, ConcreteType, Translator


@pytest.fixture
def site_dir():
    test_site_dir = os.path.join(os.path.dirname(__file__), "site_dir")
    sys.path.insert(0, test_site_dir)
    yield test_site_dir
    sys.path.remove(test_site_dir)


def test_find_plugin_loaders(site_dir):
    kinds = {
        "abstract_type": AbstractType,
        "concrete_type": ConcreteType,
        "translator": Translator,
    }
    for kind, kind_class in kinds.items():
        loaders = metagraph.entrypoints.find_plugin_loaders(kind)
        for loader in loaders:
            plugins = loader()
            assert len(plugins) > 0
            for obj in plugins:
                assert isinstance(obj, kind_class)


def test_load_plugins(site_dir):
    kinds = {
        "abstract_type": AbstractType,
        "concrete_type": ConcreteType,
        "translator": Translator,
    }
    for kind, kind_class in kinds.items():
        plugins = metagraph.entrypoints.load_plugins(kind)
        assert len(plugins) > 0
        for obj in plugins:
            assert isinstance(obj, kind_class)
