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

from .util import site_dir


KINDS = {
    "abstract_types": (issubclass, AbstractType),
    "concrete_types": (issubclass, ConcreteType),
    "translators": (isinstance, Translator),
    "wrappers": (issubclass, Wrapper),
    "abstract_algorithms": (isinstance, AbstractAlgorithm),
    "concrete_algorithms": (isinstance, ConcreteAlgorithm),
}


def test_load_plugins(site_dir):
    plugins = metagraph.core.entrypoints.load_plugins()
    for kind, (test_func, kind_class) in KINDS.items():
        kind_plugins = plugins[kind]
        assert len(kind_plugins) > 0
        for obj in kind_plugins:
            assert test_func(obj, kind_class)
