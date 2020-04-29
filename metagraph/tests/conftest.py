import pytest
from metagraph.core.resolver import Resolver


@pytest.fixture(scope="session")
def default_plugin_resolver():
    res = Resolver()
    res.load_plugins_from_environment()
    return res
