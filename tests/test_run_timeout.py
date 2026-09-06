"""Regression tests for bounding how long ``ps`` waits on a subprocess.

Reading ``Command.__doc__`` shells out (``man 1 <cmd>``, then ``<cmd> --help``),
so an unbounded, stdin-inheriting probe turns plain attribute access into a
potential hang. These tests pin the timeout plumbing that keeps it bounded.

Most of them are portable: the timeout and reap paths differ most between POSIX
and Windows (``Popen.communicate`` drains in reader threads there), which is
exactly where they should not go untested. Only the tests that need a ``/bin/sh``
fixture script are POSIX-only.
"""

import os
import signal
import sys
import time
from functools import partial
from subprocess import PIPE, Popen

import pytest

from ps.base import DFLT_DOC_TIMEOUT, Command, dash_dash_help_str, find_doc
from ps.util import ProcessError, ProcessTimeout, _kill_and_reap, run

posix_only = pytest.mark.skipif(
    os.name == "nt", reason="uses POSIX shell script fixtures"
)

# ``run`` splits its command with POSIX ``shlex``, which eats the backslashes of a
# Windows path, so portable command strings must be spelled without one.
_ECHO_COMMAND = "cmd /c echo hi" if os.name == "nt" else "echo hi"

# A python child that hands our pipes to a grandchild and then hangs around,
# recording the grandchild's pid in the file named by argv[1] so the test can
# reap it. Its argv is a list, so no shell (and no ``shlex``) is involved, which
# is what makes it usable on Windows too -- the platform whose reap path differs.
_SPAWNS_A_GRANDCHILD = (
    "import subprocess, sys, time; "
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']); "
    "open(sys.argv[1], 'w').write(str(child.pid)); "
    "time.sleep(10)"
)


def _long_running_command(seconds):
    """A command string that simply takes at least ``seconds`` to finish."""
    if os.name == "nt":
        # ``ping`` waits ~1s between echoes and, unlike ``timeout``, needs no console.
        return f"ping -n {int(seconds) + 1} 127.0.0.1"
    return f"sleep {seconds}"


def _executable_script(path, body):
    """Write ``body`` as an executable /bin/sh script and return its path string."""
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def _sleeping_python_process():
    """A live child that does nothing for a while, with both its pipes ours."""
    return Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], stdout=PIPE, stderr=PIPE
    )


@pytest.fixture
def hanging_command(tmp_path, monkeypatch):
    """A command on ``PATH``, reachable by bare name, that never answers.

    The bare name matters: handed an absolute path, macOS ``man`` cheerfully
    formats the script file itself and returns in milliseconds, so ``find_doc``
    would never reach the probe that hangs.
    """
    _executable_script(tmp_path / "hangs", "sleep 3\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return "hangs"


@pytest.fixture
def orphan_pidfile(tmp_path):
    """A file for the pids of deliberately-orphaned grandchildren, reaped after.

    The tests below *have* to leave a grandchild holding the pipes -- that is the
    scenario under test. Collecting them here is what keeps the suite from
    leaking processes that outlive the pytest session.
    """
    pidfile = tmp_path / "grandchildren.pid"
    yield pidfile
    for pid in pidfile.read_text().split() if pidfile.exists() else ():
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (OSError, ValueError):
            pass  # already gone, or never really a pid


@pytest.fixture
def stdin_that_never_ends():
    """Install a pipe nobody writes to as fd 0, so an inheriting child blocks.

    Needed because under pytest's default capture fd 0 is already ``/dev/null``,
    which silently hides the very bug this is here to catch.
    """
    read_fd, write_fd = os.pipe()
    saved_stdin = os.dup(0)
    os.dup2(read_fd, 0)
    try:
        yield
    finally:
        os.dup2(saved_stdin, 0)
        os.close(saved_stdin)
        os.close(read_fd)
        os.close(write_fd)


# --------------------------------------------------------------------- run ----


def test_run_accepts_a_timeout():
    started = time.monotonic()
    with pytest.raises(ProcessTimeout):
        run(_long_running_command(30), timeout=0.3)
    assert time.monotonic() - started < 5, "run waited well past its timeout"


def test_a_timeout_is_catchable_the_way_process_errors_always_were():
    """Existing ``except ProcessError`` / ``except OSError`` handlers still catch."""
    with pytest.raises(ProcessError):
        run(_long_running_command(30), timeout=0.2)
    with pytest.raises(OSError):
        run(_long_running_command(30), timeout=0.2)


def test_passing_timeout_none_is_the_same_as_not_passing_it():
    assert run(_ECHO_COMMAND, timeout=None) == run(_ECHO_COMMAND) == b"hi"


def test_an_infinite_timeout_means_no_limit():
    """``inf`` is a natural way to spell it; ``subprocess`` can't swallow it."""
    assert run(_ECHO_COMMAND, timeout=float("inf")) == b"hi"


@pytest.mark.parametrize(
    "bad_timeout, expected",
    [
        ("5", TypeError),
        (-1, ValueError),
        (float("nan"), ValueError),
        (float("-inf"), ValueError),
    ],
)
def test_a_timeout_that_could_not_bound_anything_starts_no_process(
    bad_timeout, expected, monkeypatch
):
    """Rejected up front, so the failure cannot leak a running command."""

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("a process was started despite an unusable timeout")

    monkeypatch.setattr("ps.util.Popen", unexpected_popen)

    with pytest.raises(expected):
        run(_long_running_command(30), timeout=bad_timeout)


# ------------------------------------------------------------ kill and reap ----


def test_a_failure_other_than_a_timeout_still_kills_the_command(monkeypatch):
    """Any exception out of ``communicate`` would otherwise leak a live command."""
    started = []

    class ExplodingPopen(Popen):
        """Fails its first ``communicate`` -- as a KeyboardInterrupt would."""

        exploded = False

        def communicate(self, *args, **kwargs):
            if not type(self).exploded:
                type(self).exploded = True
                raise RuntimeError("boom")
            return super().communicate(*args, **kwargs)

    def remembered_popen(*args, **kwargs):
        started.append(ExplodingPopen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr("ps.util.Popen", remembered_popen)

    with pytest.raises(RuntimeError, match="boom"):
        run(_long_running_command(30))

    (process,) = started
    assert process.poll() is not None, "the command was left running"


def test_kill_and_reap_kills_the_command_and_collects_it():
    """The ``run`` docstring promises the command dies, not that we stop waiting."""
    process = _sleeping_python_process()

    _kill_and_reap(process)

    assert process.poll() is not None, "the timed-out command was left running"


def test_kill_and_reap_is_bounded_when_a_grandchild_holds_the_pipes(orphan_pidfile):
    """Killing the child is not enough: a grandchild keeps the pipes open.

    Draining them unconditionally -- or closing a stream a reader thread is still
    inside, which is how Windows drains -- blocks for as long as the grandchild
    lives, which is the very hang the timeout exists to prevent.
    """
    process = Popen(
        [sys.executable, "-c", _SPAWNS_A_GRANDCHILD, str(orphan_pidfile)],
        stdout=PIPE,
        stderr=PIPE,
    )
    deadline = time.monotonic() + 10  # let the grandchild take hold of the pipes
    while not orphan_pidfile.exists():
        assert time.monotonic() < deadline, "the fixture child never spawned"
        time.sleep(0.05)

    started = time.monotonic()
    _kill_and_reap(process)
    elapsed = time.monotonic() - started

    assert process.poll() is not None, "the direct child was not collected"
    assert elapsed < 4, "the reap blocked on the grandchild"


@posix_only
def test_run_kills_the_command_it_timed_out_on(tmp_path):
    """End to end: ``run`` does not merely stop waiting, it stops the command."""
    marker = tmp_path / "FINISHED"
    script = _executable_script(tmp_path / "slow", f"sleep 0.5\ntouch {marker}\n")

    with pytest.raises(ProcessTimeout):
        run(script, timeout=0.1)

    time.sleep(1.5)  # comfortably past the script's own sleep
    assert not marker.exists(), "the timed-out command ran on to completion"


@posix_only
def test_run_returns_promptly_when_a_grandchild_holds_the_pipes(
    tmp_path, orphan_pidfile
):
    # Both sleeps hold the pipes and both outlive the killed shell, so both pids
    # are recorded for the fixture to collect.
    script = _executable_script(
        tmp_path / "spawns",
        f"sleep 10 &\necho $! > {orphan_pidfile}\n"
        f"sleep 10 &\necho $! >> {orphan_pidfile}\nwait\n",
    )

    started = time.monotonic()
    with pytest.raises(ProcessTimeout):
        run(script, timeout=0.5)
    assert time.monotonic() - started < 5, "timeout path blocked on the grandchild"


# --------------------------------------------------------------- doc probes ----


def test_the_default_doc_timeout_is_a_sane_bound():
    assert 0 < DFLT_DOC_TIMEOUT <= 10


@posix_only
def test_the_default_doc_timeout_is_read_at_call_time(hanging_command, monkeypatch):
    """It reads as configuration, so it has to actually work as configuration."""
    monkeypatch.setattr("ps.base.DFLT_DOC_TIMEOUT", 0.3)

    started = time.monotonic()
    with pytest.raises(ProcessTimeout):
        dash_dash_help_str(hanging_command)

    assert time.monotonic() - started < DFLT_DOC_TIMEOUT, "the bound was frozen"


@posix_only
def test_doc_probes_do_not_inherit_a_live_stdin(tmp_path, stdin_that_never_ends):
    """A probe must not sit forever on a stdin that nobody is going to write to."""
    reads_stdin = _executable_script(tmp_path / "reads_stdin", "cat\n")

    assert dash_dash_help_str(reads_stdin, timeout=0.5) == ""

    # ...but a caller who wants the old inherited-stdin behaviour can still ask:
    with pytest.raises(ProcessTimeout):
        dash_dash_help_str(reads_stdin, timeout=0.5, stdin=None)


@posix_only
def test_find_doc_falls_through_a_probe_that_timed_out(hanging_command):
    """The documented degradation, and the warning that makes it attributable."""

    def fallback(command, **kwargs):
        return "FALLBACK DOC"

    with pytest.warns(UserWarning, match="Gave up documenting"):
        doc = find_doc(
            hanging_command,
            doc_finders=(dash_dash_help_str, fallback),
            timeout=0.3,
        )

    assert doc == "FALLBACK DOC"


@posix_only
def test_a_command_no_probe_can_document_gets_an_empty_doc(hanging_command):
    """``__doc__`` on the default path degrades to '' rather than hanging or raising."""
    assert Command(hanging_command).get_doc is find_doc, "this is the default path"
    command = Command(hanging_command, get_doc=partial(find_doc, timeout=0.3))

    started = time.monotonic()
    with pytest.warns(UserWarning):
        assert command.__doc__ == ""

    assert time.monotonic() - started < DFLT_DOC_TIMEOUT
