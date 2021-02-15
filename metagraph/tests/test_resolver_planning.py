import pytest
import codecs

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
    from .util import OtherType, IntType, StrNum, StrNumRot13

    # Direct
    translator = example_resolver.plan.translate(4, StrNum.Type)
    assert not translator.unsatisfiable
    assert translator.final_type == StrNum.Type
    assert len(translator) == 1
    assert len(list(translator)) == 1
    assert "[Direct Translation]" in repr(translator)

    # Unsatisfiable
    translator = example_resolver.plan.translate(4, OtherType)
    assert translator.unsatisfiable
    assert translator.final_type == OtherType
    assert "[Unsatisfiable Translation]" in repr(translator)
    with pytest.raises(ValueError, match="No translation path found for"):
        len(translator)
    with pytest.raises(ValueError, match="No translation path found for"):
        iter(translator)
    with pytest.raises(ValueError, match="No translation path found for"):
        result = translator(4)

    # Null
    translator = example_resolver.plan.translate(4, IntType)
    assert not translator.unsatisfiable
    assert translator.final_type == IntType
    assert len(translator) == 0
    assert "[Null Translation]" in repr(translator)

    # Multi-step
    translator = example_resolver.plan.translate(4, StrNumRot13.Type)
    assert not translator.unsatisfiable
    assert translator.final_type == StrNumRot13.Type
    assert len(translator) == 2
    assert "[Multi-step Translation]" in repr(translator)


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


def test_build_algorithm_plan(example_resolver):
    res = example_resolver
    StrNum = res.wrappers.MyNumericAbstractType.StrNum
    StrNumRot13 = res.wrappers.MyNumericAbstractType.StrNumRot13

    # No translation needed
    power = list(res.plugins.example2_plugin.concrete_algorithms["power"])[0]
    plan = AlgorithmPlan.build(res, power, StrNum("2"), StrNum("3"))
    assert " Translation]" not in repr(plan)

    # Direct translation
    power = list(res.plugins.example2_plugin.concrete_algorithms["power"])[0]
    plan = AlgorithmPlan.build(res, power, 2, 3)
    assert "[Direct Translation]" in repr(plan)

    # Multi-step translation
    power = list(res.plugins.example3_plugin.concrete_algorithms["power"])[0]
    plan = AlgorithmPlan.build(res, power, 2, 3)
    assert "[Multi-step Translation]" in repr(plan)

    # Unsatisfiable
    power = list(res.plugins.example_plugin.concrete_algorithms["power"])[0]
    rot = StrNumRot13(codecs.encode("2", "rot-13"))
    plan = AlgorithmPlan.build(res, power, rot, 3)
    assert "Failed to find translator" in repr(plan)
    with pytest.raises(ValueError, match="Algorithm not callable because"):
        plan(rot, 3)


def test_build_algorithm_plan_errors(example_resolver):
    res = example_resolver
    StrNum = res.wrappers.MyNumericAbstractType.StrNum
    # crazy = list(res.plugins.example_plugin.concrete_algorithms["crazy_inputs"])[0]

    kwargs = {
        "a1": 1,  # NodeID
        "a2": 2,  # int
        "b1": 3.3,  # Union[int, float]
        "b2": 4.4,  # mg.Union[int, float]
        "c1": None,  # c1: Optional[int]
        "c2": None,  # c2: mg.Optional[int]
        "d1": [StrNum("1"), StrNum("2")],  # d1: List[int]
        "d2": [2, 3, 4],  # d2: mg.List[int]
        "e": lambda x: x + 1,  # e: Callable[[Any], Any]
    }

    # plan = AlgorithmPlan.build(res, crazy, **kwargs)
    # print(repr(plan))
    # assert not plan.unsatisfiable
    #
    # plan = AlgorithmPlan.build(res, crazy, **{**kwargs, "a1": "node_name"})
    # assert plan.unsatisfiable
    # assert "`a1` Not a valid NodeID: node_name" in repr(plan)
    #
    # plan = AlgorithmPlan.build(res, crazy, **{**kwargs, "a2": 2.2})
    # assert plan.unsatisfiable
    # assert "Failed to find translator to IntType for a2" in repr(plan)
    #
    # plan = AlgorithmPlan.build(res, crazy, **{**kwargs, "b1": None})
    # assert plan.unsatisfiable
    # assert "b1 is not Optional, but None was given" in repr(plan)
    #
    # plan = AlgorithmPlan.build(res, crazy, **{**kwargs, "b2": "a"})
    # assert plan.unsatisfiable
    # assert "`b2` with type <class 'str'> does not match any of Union" in repr(plan)
    #
    # plan = AlgorithmPlan.build(res, crazy, **{**kwargs, "c1": 3.3})
    # assert plan.unsatisfiable
    #
    # plan = AlgorithmPlan.build(res, crazy, **{**kwargs, "c2": 3+2j})
    # assert plan.unsatisfiable

    # plan = AlgorithmPlan.build(res, crazy, **{**kwargs, "d1": StrNum})
    # assert plan.unsatisfiable
    # assert "" in repr(plan)

    # plan = AlgorithmPlan.build(res, crazy, **{})
    # assert plan.unsatisfiable
    # assert "" in repr(plan)
    #
    # plan = AlgorithmPlan.build(res, crazy, **{})
    # assert plan.unsatisfiable
    # assert "" in repr(plan)
    #
    # plan = AlgorithmPlan.build(res, crazy, **{})
    # assert plan.unsatisfiable
    # assert "" in repr(plan)

    # print(repr(plan))
    # raise Exception('foobar')
