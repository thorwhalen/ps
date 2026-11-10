"""Tests for ps.util.kill_process_tree and descendants: process trees spanning sessions,
the caller's own group, and a reaped process whose pid now belongs to someone else."""

import os
import subprocess
import sys
import time
import types

import pytest

from ps.util import NEW_PROCESS_GROUP, descendants, kill_process_tree

posix_only = pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX")
tree_readable = pytest.mark.skipif(
    not (sys.platform == "darwin" or sys.platform.startswith("linux")),
    reason="descendants() reads macOS libproc or Linux /proc",
)

# A parent that starts a child (in a new session when argv[1] == "own") and prints its pid.
PARENT = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                         start_new_session=sys.argv[1] == "own")
print(child.pid, flush=True)
time.sleep(120)
"""


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie (killed, not yet reaped by its parent) counts as gone
    if sys.platform.startswith("linux"):
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")


def _wait_gone(pid, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _start(mode, **popen):
    parent = subprocess.Popen([sys.executable, "-c", PARENT, mode], stdout=subprocess.PIPE,
                              text=True, **popen)
    return parent, int(parent.stdout.readline())


@posix_only
@tree_readable
def test_a_child_in_a_session_of_its_own_is_killed_too():
    parent, child = _start("own", **NEW_PROCESS_GROUP)
    try:
        assert child in descendants(parent.pid)
        kill_process_tree(parent)
        parent.wait(5)
        assert _wait_gone(child), "the child in its own session survived"
    finally:
        for pid in (child,):
            try:
                os.kill(pid, 9)
            except OSError:
                pass


@posix_only
@tree_readable
def test_the_callers_own_group_is_spared():
    # the parent stays in our process group: its group must not be killed (it is ours)
    parent, child = _start("same")
    try:
        kill_process_tree(parent)
        parent.wait(5)
        assert _wait_gone(child)
        assert _alive(os.getpid())  # we are still here
    finally:
        try:
            os.kill(child, 9)
        except OSError:
            pass


@posix_only
def test_a_reaped_process_whose_pid_is_reused_is_left_alone():
    """ps#13: the pid of a process reaped earlier may now lead an unrelated group."""
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 **NEW_PROCESS_GROUP)
    try:
        reaped = types.SimpleNamespace(pid=unrelated.pid, returncode=0,
                                       poll=lambda: 0, kill=lambda: None)
        kill_process_tree(reaped)
        time.sleep(0.2)
        assert unrelated.poll() is None, "an unrelated process with the reused pid was killed"
    finally:
        unrelated.kill()
        unrelated.wait()


@posix_only
def test_orphans_of_a_reaped_leader_are_still_killed():
    # the leader starts a child in its own group, prints its pid, and exits
    leader = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys; c = subprocess.Popen([sys.executable, '-c', "
         "'import time; time.sleep(60)']); print(c.pid, flush=True)"],
        stdout=subprocess.PIPE, text=True, **NEW_PROCESS_GROUP)
    orphan = int(leader.stdout.readline())
    leader.wait(5)  # reaped: returncode is set, its pid is free but the group lives on
    try:
        kill_process_tree(leader)
        assert _wait_gone(orphan), "the orphan in the leader's group survived"
    finally:
        try:
            os.kill(orphan, 9)
        except OSError:
            pass


def test_descendants_of_a_childless_process_is_empty():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert descendants(proc.pid) == []
    finally:
        proc.kill()
        proc.wait()
