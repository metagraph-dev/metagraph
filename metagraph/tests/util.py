import os
import sys
import pytest

from metagraph.core import plugin
from metagraph.core.resolver import Resolver


def make_site_dir_fixture(site_dir):
    test_site_dir = os.path.join(os.path.dirname(__file__), site_dir)
    sys.path.insert(0, test_site_dir)
    yield test_site_dir
    sys.path.remove(test_site_dir)


@pytest.fixture
def site_dir():
    yield from make_site_dir_fixture("site_dir")


@pytest.fixture
def bad_site_dir():
    yield from make_site_dir_fixture("bad_site_dir")


# Handy for manual testing
def make_example_resolver():
    res = Resolver()
    import metagraph

    registry = metagraph.PluginRegistry()
    from . import example_plugin_util

    registry.register_from_modules(
        "example_plugin", [example_plugin_util, metagraph.types, metagraph.algorithms]
    )
    res.register(registry)
    return res


@pytest.fixture
def example_resolver():
    return make_example_resolver()


@pytest.fixture(scope="session")
def default_plugin_resolver(request):  # pragma: no cover
    res = Resolver()
    if request.config.getoption("--no-plugins", default=False):
        from metagraph.plugins import find_plugins

        res.register(**find_plugins())
    else:
        res.load_plugins_from_environment()
    return res
