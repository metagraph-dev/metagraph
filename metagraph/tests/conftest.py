import pytest
from metagraph.core.resolver import Resolver


@pytest.fixture(scope="session")
def default_plugin_resolver():
    from metagraph.plugins import find_plugins

    res = Resolver()
    # res.register(**find_plugins())
    res.load_plugins_from_environment()
    return res
