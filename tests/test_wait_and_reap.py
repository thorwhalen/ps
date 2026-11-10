"""Tests for ps.util.wait_and_reap: what a process started is killed however the process
ends (on its own, by a crash, on timeout), and a recycled pid is never mistaken for it."""

import os
import subprocess
import sys
import time

import pytest

import ps.util
from ps.util import NEW_PROCESS_GROUP, start_time, wait_and_reap

pytestmark = pytest.mark.skipif(
    not (sys.platform == "darwin" or sys.platform.startswith("linux")),
    reason="the tree is read from macOS libproc or Linux /proc",
)

# Starts a sleeper (in a session of its own when argv[1] == "own", else in this
# process's group), writes its pid to argv[2], lingers argv[3] s, then exits with 3
# without cleaning up -- as a crash does.
PARENT = """
import os, subprocess, sys, time
w = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                     start_new_session=sys.argv[1] == "own")
with open(sys.argv[2] + ".tmp", "w") as f:
    f.write(str(w.pid))
os.replace(sys.argv[2] + ".tmp", sys.argv[2])
time.sleep(float(sys.argv[3]))
os._exit(3)
"""


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    if sys.platform.startswith("linux"):  # a zombie counts as gone
        try:
            with open(f"/proc/{pid}/stat") as f:
                return f.read().rsplit(")", 1)[1].split()[0] != "Z"
        except OSError:
            return False
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")


def _gone_soon(pid, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _start(tmp_path, where, linger):
    pidfile = tmp_path / "worker.pid"
    child = subprocess.Popen(
        [sys.executable, "-c", PARENT, where, str(pidfile), str(linger)],
        **NEW_PROCESS_GROUP,
    )
    return child, pidfile


def _worker(pidfile):
    return int(pidfile.read_text())


@pytest.fixture
def cleanup():
    pids = []
    yield pids
    for pid in pids:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def test_a_worker_in_its_own_session_dies_when_the_child_exits_on_its_own(
    tmp_path, cleanup
):
    child, pidfile = _start(tmp_path, "own", linger=1.0)
    assert wait_and_reap(child, timeout=30, interval=0.05) == 3
    worker = _worker(pidfile)
    cleanup.append(worker)
    assert _gone_soon(worker)


def test_an_orphan_in_the_childs_own_group_dies_too(tmp_path, cleanup):
    # linger 0: the child may exit before any reading; its group still reaches the orphan
    child, pidfile = _start(tmp_path, "group", linger=0)
    assert wait_and_reap(child, timeout=30, interval=5) == 3
    worker = _worker(pidfile)
    cleanup.append(worker)
    assert _gone_soon(worker)


def test_without_it_the_worker_outlives_the_child(tmp_path, cleanup):
    """The failure wait_and_reap is for (so the tests above test something)."""
    child, pidfile = _start(tmp_path, "own", linger=0.5)
    assert child.wait(timeout=30) == 3
    worker = _worker(pidfile)
    cleanup.append(worker)
    assert not _gone_soon(worker, timeout=1.0)


def _wait_for(path, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.05)


def test_on_timeout_the_tree_is_killed_and_the_timeout_raised(tmp_path, cleanup):
    child, pidfile = _start(tmp_path, "own", linger=120)
    _wait_for(pidfile)  # a slow runner may take seconds to get this far
    cleanup.append(_worker(pidfile))
    with pytest.raises(subprocess.TimeoutExpired):
        wait_and_reap(child, timeout=0.5, interval=0.05)
    assert child.returncode is not None
    assert _gone_soon(_worker(pidfile))


def test_an_interrupted_wait_still_kills_everything(tmp_path, cleanup, monkeypatch):
    """Ctrl-C does not reach a child in its own session: the wait must clean up."""
    child, pidfile = _start(tmp_path, "own", linger=120)
    _wait_for(pidfile)
    cleanup.append(_worker(pidfile))
    real_wait = child.wait
    calls = []

    def interrupted(timeout=None):
        if not calls:
            calls.append(timeout)
            time.sleep(0.5)  # the watch sees the worker
            raise KeyboardInterrupt
        return real_wait(timeout=timeout)

    monkeypatch.setattr(child, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        wait_and_reap(child, timeout=30, interval=0.05)
    assert _gone_soon(child.pid)
    assert _gone_soon(_worker(pidfile))


def test_a_child_that_exits_between_the_timeout_and_the_kill(
    tmp_path, cleanup, monkeypatch
):
    """Its worker is re-parented by the time the tree is read: only the watch finds it."""
    child, pidfile = _start(tmp_path, "own", linger=1.5)
    real = ps.util.kill_process_tree

    def late(process):  # the child finishes exiting before the kill reads its tree
        process.wait()
        real(process)

    monkeypatch.setattr(ps.util, "kill_process_tree", late)
    with pytest.raises(subprocess.TimeoutExpired):
        wait_and_reap(child, timeout=0.8, interval=0.05)
    worker = _worker(pidfile)
    cleanup.append(worker)
    assert _gone_soon(worker)


def test_a_pid_now_owned_by_another_process_is_spared(tmp_path, cleanup, monkeypatch):
    child, pidfile = _start(tmp_path, "own", linger=1.0)
    readings = {}
    real = start_time

    def recycled(pid):  # real while the child runs; after, every pid is someone else's
        if child.returncode is None:
            readings[pid] = real(pid)
            return readings[pid]
        return ("someone", "else")

    monkeypatch.setattr(ps.util, "start_time", recycled)
    assert wait_and_reap(child, timeout=30, interval=0.05) == 3
    worker = _worker(pidfile)
    cleanup.append(worker)
    assert worker in readings  # it was seen ...
    assert not _gone_soon(worker, timeout=1.0)  # ... and spared


def test_a_worker_left_in_the_callers_own_group_dies_but_the_caller_does_not(
    tmp_path, cleanup
):
    """No NEW_PROCESS_GROUP: the worker shares the caller's group, which is never killed whole."""
    pidfile = tmp_path / "worker.pid"
    child = subprocess.Popen([sys.executable, "-c", PARENT, "group", str(pidfile), "1.0"])
    assert wait_and_reap(child, timeout=30, interval=0.05) == 3
    cleanup.append(_worker(pidfile))
    assert _gone_soon(_worker(pidfile))
    assert _alive(os.getpid())


def test_a_bad_timeout_is_refused_before_waiting():
    child = subprocess.Popen([sys.executable, "-c", "pass"], **NEW_PROCESS_GROUP)
    with pytest.raises(TypeError):
        wait_and_reap(child, timeout="5")
    with pytest.raises(ValueError):
        wait_and_reap(child, interval=0)
    child.wait()


def test_start_time_tells_two_processes_apart():
    a = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    time.sleep(0.05)
    b = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert start_time(a.pid) is not None
        assert start_time(a.pid) == start_time(a.pid)
        assert start_time(a.pid) != start_time(b.pid)
    finally:
        a.kill(), b.kill(), a.wait(), b.wait()
    assert start_time(a.pid) is None
