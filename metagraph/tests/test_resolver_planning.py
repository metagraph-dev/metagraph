import pytest

import metagraph as mg
from metagraph import (
    AbstractType,
    ConcreteType,
    Wrapper,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)
from metagraph.core.plugin_registry import PluginRegistry
from metagraph.core.resolver import (
    Resolver,
    _ResolverRegistrar,
    Namespace,
    Dispatcher,
    NamespaceError,
    AlgorithmWarning,
)
from metagraph.core.planning import MultiStepTranslator, AlgorithmPlan
from metagraph import config
import typing
from typing import Tuple, List, Dict, Any
from collections import OrderedDict

from .util import site_dir, example_resolver


def test_translate_plan(example_resolver):
    from .util import StrNum, OtherType

    translator = example_resolver.plan.translate(4, StrNum.Type)
    assert not translator.unsatisfiable
    assert translator.final_type == StrNum.Type
    assert len(translator) == 1
    translator = example_resolver.plan.translate(4, OtherType)
    assert translator.unsatisfiable
    assert translator.final_type == OtherType


def test_run_algorithm_plan(example_resolver, capsys):
    capsys.readouterr()
    plan = example_resolver.plan.run("power", 2, 3)
    text = repr(plan)
    assert "int_power" in text
    assert "Argument Translations" in text
    example_resolver.plan.run("power", 2, "4")
    captured = capsys.readouterr()
    assert (
        'No concrete algorithm for "power" can be satisfied for the given inputs'
        in captured.out
    )
