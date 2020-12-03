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
