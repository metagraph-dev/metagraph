import pytest
from metagraph.core.resolver import Resolver


def pytest_addoption(parser):
    parser.addoption(
        "--no-plugins",
        action="store_true",
        help="Exclude plugins when running algorithm tests.",
    )


@pytest.fixture(scope="session")
def default_plugin_resolver(request):
    res = Resolver()
    if request.config.getoption("--no-plugins"):
        from metagraph.plugins import find_plugins

        res.register(**find_plugins())
    else:
        res.load_plugins_from_environment()
    return res
