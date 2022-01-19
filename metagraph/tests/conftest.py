import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-plugins",
        action="store_true",
        help="Exclude plugins when running algorithm tests.",
    )
    parser.addoption(
        "--dask",
        action="store_true",
        help="Use a DaskResolver instead of the normal Resolver.",
    )
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_runtest_setup(item):
    if "runslow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run this test")
