"""Windows: the process tree from the process table, and wait_and_reap's survivors (#25)."""

import subprocess
import sys
import time

import pytest

from ps.util import NEW_PROCESS_GROUP, descendants, start_time, wait_and_reap

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows process table")

# Starts a sleeper, writes its pid to argv[1], lingers argv[2] s, then exits with 3
# without cleaning up, as a crash does: the sleeper is left behind.
PARENT = """
import os, subprocess, sys, time
w = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
time.sleep(0.5)  # let the watcher see the sleeper before it is written down
with open(sys.argv[1] + ".tmp", "w") as f:
    f.write(str(w.pid))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(float(sys.argv[2]))
os._exit(3)
"""


def _gone_soon(pid, started, timeout=10.0):
    """Whether the process ``pid`` that started at ``started`` is gone (its pid may
    already belong to another process, which is why the start time is compared)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if start_time(pid) != started:
            return True
        time.sleep(0.1)
    return False


def test_descendants_and_start_time():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert child.pid in descendants()
        first = start_time(child.pid)
        assert first is not None and first == start_time(child.pid)
    finally:
        child.kill()
        child.wait()
    assert _gone_soon(child.pid, first)


def test_a_worker_left_behind_is_killed_when_the_child_exits_on_its_own(tmp_path):
    pidfile = tmp_path / "worker.pid"
    child = subprocess.Popen(
        [sys.executable, "-c", PARENT, str(pidfile), "2.0"], **NEW_PROCESS_GROUP
    )
    while not pidfile.exists():  # read the worker's start time while it runs
        time.sleep(0.05)
    worker = int(pidfile.read_text())
    started = start_time(worker)
    assert started is not None
    assert wait_and_reap(child, timeout=60, interval=0.1) == 3
    try:
        assert _gone_soon(worker, started), "the sleeper the child started outlived it"
    finally:
        subprocess.call(
            ["taskkill", "/F", "/PID", str(worker)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
