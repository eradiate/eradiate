import os
import subprocess
import sys


def test_eager_import():
    """
    This test checks if Eradiate can be imported in eager mode. It is primarily
    intended to detect circular imports, which are much harder to spot since
    we introduced lazy loading.
    """
    env = os.environ.copy()
    env["EAGER_IMPORT"] = "1"

    result = subprocess.call([sys.executable, "-c", "import eradiate"], env=env)
    assert result == 0
