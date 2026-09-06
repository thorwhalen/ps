"""Regression tests for bounding how long ``ps`` waits on a subprocess.

Reading ``Command.__doc__`` shells out (``man 1 <cmd>``, then ``<cmd> --help``),
so an unbounded, stdin-inheriting probe turns plain attribute access into a
potential hang. These tests pin the timeout plumbing that keeps it bounded.
"""

import subprocess
import sys
import time

import pytest

from ps.base import DFLT_DOC_TIMEOUT, Command, dash_dash_help_str, man_1_page_str
from ps.util import ProcessError, ProcessTimeout, run

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="uses POSIX shell script fixtures"
)


def _executable_script(path, body):
    """Write ``body`` as an executable /bin/sh script and return its path string."""
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def test_run_accepts_a_timeout():
    started = time.monotonic()
    with pytest.raises(ProcessTimeout):
        run("sleep 5", timeout=0.3)
    assert time.monotonic() - started < 3, "run waited well past its timeout"


def test_process_timeout_is_catchable_as_before():
    """Existing ``except ProcessError`` / ``except OSError`` handlers still work."""
    assert issubclass(ProcessTimeout, ProcessError)
    assert issubclass(ProcessTimeout, OSError)


def test_run_without_a_timeout_is_unchanged():
    assert run("echo hello world") == b"hello world"


def test_doc_probes_are_bounded_and_close_stdin(monkeypatch):
    calls = []

    def recorder(*args, **kwargs):
        calls.append(kwargs)
        return b""

    monkeypatch.setattr("ps.base.run", recorder)

    man_1_page_str("pwd")
    dash_dash_help_str("pwd")

    assert len(calls) == 2
    for kwargs in calls:
        assert isinstance(kwargs.get("timeout"), (int, float))
        assert kwargs["timeout"] > 0
        assert kwargs.get("stdin") is subprocess.DEVNULL


def test_a_hanging_doc_probe_does_not_block_forever(tmp_path):
    """``--help`` on a command that ignores it must not hang ``__doc__``."""
    script = _executable_script(tmp_path / "hangs", "sleep 10\n")
    command = Command(script, get_doc=dash_dash_help_str)

    started = time.monotonic()
    with pytest.raises(ProcessTimeout):
        command.__doc__
    assert time.monotonic() - started < 2 * DFLT_DOC_TIMEOUT


def test_timeout_does_not_wait_on_a_grandchild_holding_the_pipes(tmp_path):
    """Killing the child is not enough: a grandchild keeps the pipes open.

    Draining them unconditionally would block for as long as the grandchild
    lives, which is the very hang the timeout exists to prevent.
    """
    script = _executable_script(tmp_path / "spawns", "sleep 10 &\nsleep 10\n")

    started = time.monotonic()
    with pytest.raises(ProcessTimeout):
        run(script, timeout=0.5)
    assert time.monotonic() - started < 5, "timeout path blocked on the grandchild"
