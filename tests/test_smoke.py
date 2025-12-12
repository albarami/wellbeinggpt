"""
Pytest smoke test to ensure the test framework is working.

This test ensures that pytest -v runs successfully from day one.
"""


def test_pytest_smoke():
    """
    Minimal smoke test that always passes.

    This test exists to ensure pytest doesn't fail with
    "no tests collected" on a fresh repository.
    """
    assert True


def test_import_main_app():
    """Test that the main app can be imported."""
    from apps.api.main import app

    assert app is not None
    assert app.title == "Wellbeing Data Foundation API"

