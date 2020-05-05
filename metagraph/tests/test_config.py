import pytest
import metagraph as mg


def test_defaults():
    with pytest.raises(KeyError):
        v = mg.config.get("not.an.attribute")

    # Not an exhaustive list, but enough to check that things are working
    assert mg.config.get("core.logging.plans") == False
    assert mg.config.get("core.dispatch.allow_translation") == True
