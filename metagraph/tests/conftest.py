import pytest
from metagraph.core.resolver import Resolver


@pytest.fixture(scope="session")
def default_plugin_resolver():
    from metagraph.default_plugins import find_plugins

    res = Resolver()
    res.register(**find_plugins())
    return res
