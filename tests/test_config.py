import pytest
from hermes.core.config import settings

def test_settings_persona_weights():
    # Ensure settings.persona_weights is a dictionary and contains required keys.
    assert isinstance(settings.persona_weights, dict), "persona_weights should be a dict"
    for key in ["apre", "magi"]:
        assert key in settings.persona_weights, f"{key} key missing in persona_weights"

def test_enable_gpu_setting():
    # Check that enable_gpu is a boolean.
    assert isinstance(settings.enable_gpu, bool), "enable_gpu must be a boolean"
