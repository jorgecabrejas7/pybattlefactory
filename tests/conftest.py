import os
import sys

# Import the in-tree package (dev builds put the native modules in pybattle/), not an installed copy.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_training_mark():
    """Training entry points run in tests (tiny runs) call rl.search.mark_training(), which sets PYB_TRAINING in
    os.environ for good: restore it after each test so later tests (e.g. rl.autopsy's) run as a diagnostic process."""
    before = os.environ.get("PYB_TRAINING")
    yield
    if before is None:
        os.environ.pop("PYB_TRAINING", None)
    else:
        os.environ["PYB_TRAINING"] = before
