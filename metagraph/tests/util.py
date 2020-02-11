import os
import sys

import pytest


@pytest.fixture
def site_dir():
    test_site_dir = os.path.join(os.path.dirname(__file__), "site_dir")
    sys.path.insert(0, test_site_dir)
    yield test_site_dir
    sys.path.remove(test_site_dir)
