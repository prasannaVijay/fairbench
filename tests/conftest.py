"""Shared test fixtures."""

import pytest

from fairbench_genai.scenarios.registry import reset_registry


@pytest.fixture(autouse=True)
def _isolate_scenario_registry():
    """Give each test a fresh global scenario registry.

    The registry is a module-level singleton; without this, two tests that load
    the same scenario file collide with 'already registered'.
    """
    reset_registry()
    yield
    reset_registry()
