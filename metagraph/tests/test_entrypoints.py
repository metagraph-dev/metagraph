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
    Wrapper,
)


from .util import site_dir, bad_site_dir


ABSTRACT_KINDS = {
    "abstract_types": (issubclass, AbstractType),
    "abstract_algorithms": (isinstance, AbstractAlgorithm),
}

CONCRETE_KINDS = {
    "plugin_name_to_concrete_types": (isinstance, ConcreteAlgorithm),
    "plugin_name_to_concrete_types": (issubclass, ConcreteType),
    "plugin_name_to_wrappers": (issubclass, Wrapper),
    "plugin_name_to_translators": (isinstance, Translator),
}


def test_load_registry(site_dir):
    registry = metagraph.core.entrypoints.load_plugins()
    for kind, (test_func, kind_class) in ABSTRACT_KINDS.items():
        registry_kind = getattr(registry, kind)
        assert len(registry_kind) > 0
        for obj in registry_kind:
            assert test_func(obj, kind_class)
    for kind, (test_func, kind_class) in CONCRETE_KINDS.items():
        registry_kind = getattr(registry, kind)
        assert len(registry_kind) > 0
        for plugin_name, plugin_values in registry_kind.items():
            assert isinstance(plugin_name, str)
            for obj in plugin_values:
                assert test_func(obj, kind_class)


def test_load_failure(bad_site_dir):
    with pytest.raises(metagraph.core.entrypoints.EntryPointsError):
        plugins = metagraph.core.entrypoints.load_plugins()
