import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-plugins",
        action="store_true",
        help="Exclude plugins when running algorithm tests.",
    )
