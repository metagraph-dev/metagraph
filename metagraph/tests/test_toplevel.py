import metagraph as mg


def test_version():
    assert isinstance(mg.__version__, str)
