import sys
import sysconfig

import pytest


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="requires a free-threaded build",
)
def test_gil_stays_disabled():
    # importing an extension module that doesn't declare free-threading support
    # silently re-enables the GIL, so make sure ours declares it
    import pcodec  # noqa: F401

    assert not sys._is_gil_enabled()
