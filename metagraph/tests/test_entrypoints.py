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
    "concrete_types": (isinstance, ConcreteAlgorithm),
    "concrete_types": (issubclass, ConcreteType),
    "wrappers": (issubclass, Wrapper),
    "translators": (isinstance, Translator),
}


def test_load_registry(site_dir):
    plugins = metagraph.core.entrypoints.load_plugins()
    for kind, (test_func, kind_class) in ABSTRACT_KINDS.items():
        plugin_kind = plugins["plugin1"][kind]
        assert len(plugin_kind) > 0
        for obj in plugin_kind:
            assert test_func(obj, kind_class)


def test_load_failure(bad_site_dir):
    with pytest.raises(metagraph.core.entrypoints.EntryPointsError):
        plugins = metagraph.core.entrypoints.load_plugins()
