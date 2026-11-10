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
    except OSError:  # alive, not ours
        return True
    # a zombie (killed, not yet reaped by its parent) counts as gone
    try:
        if sys.platform.startswith("linux"):
            with open(f"/proc/{pid}/stat") as f:
                return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")


def _wait_gone(pid, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _cleanup(parent, *pids):
    for pid in pids:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    if parent.poll() is None:
        parent.kill()
    parent.wait()
    parent.stdout.close()


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


OWN_GROUP_SCENARIO = """
import os, subprocess, sys, time
from ps.util import kill_process_tree
parent = subprocess.Popen([sys.executable, "-c", sys.argv[1], "same"], stdout=subprocess.PIPE, text=True)
child = int(parent.stdout.readline())
kill_process_tree(parent)
parent.wait(5)
time.sleep(0.3)
try:
    os.kill(child, 0); alive = True
except ProcessLookupError:
    alive = False
print("survived", alive)
"""


@posix_only
@tree_readable
def test_the_callers_own_group_is_spared():
    # run in a session of its own: a regression kills that subprocess's group, not pytest's
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    done = subprocess.run([sys.executable, "-c", OWN_GROUP_SCENARIO, PARENT], capture_output=True,
                          text=True, timeout=30, env=env, **NEW_PROCESS_GROUP)
    assert done.returncode == 0, done.stderr  # the caller was not killed with its group
    assert "survived False" in done.stdout


@posix_only
@tree_readable
def test_the_second_read_finds_what_the_first_missed(monkeypatch):
    import ps.util

    parent, child = _start("own", **NEW_PROCESS_GROUP)
    real, calls = ps.util.descendants, []
    def blind_first(pid=None):
        calls.append(pid)
        return [] if len(calls) == 1 else real(pid)  # the first read misses the child

    monkeypatch.setattr(ps.util, "descendants", blind_first)
    try:
        kill_process_tree(parent)
        parent.wait(5)
        assert len(calls) == 2 and _wait_gone(child), "only the second read saw the child"
    finally:
        _cleanup(parent, child)


@posix_only
@tree_readable
def test_nothing_is_left_stopped_when_the_second_read_fails(monkeypatch):
    import ps.util

    parent, child = _start("own", **NEW_PROCESS_GROUP)
    real, calls = ps.util.descendants, []

    def failing(pid=None):
        calls.append(pid)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(pid)

    monkeypatch.setattr(ps.util, "descendants", failing)
    try:
        with pytest.raises(KeyboardInterrupt):
            kill_process_tree(parent)
        parent.wait(5)
        assert _wait_gone(child), "the child was left stopped"
    finally:
        _cleanup(parent, child)


@posix_only
@tree_readable
def test_several_children_are_all_killed():
    script = ("import subprocess, sys, time\n"
              "kids = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],"
              " start_new_session=i % 2 == 0) for i in range(3)]\n"
              "print(' '.join(str(k.pid) for k in kids), flush=True)\ntime.sleep(120)\n")
    parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
                              **NEW_PROCESS_GROUP)
    kids = [int(p) for p in parent.stdout.readline().split()]
    try:
        assert set(kids) <= set(descendants(parent.pid))
        kill_process_tree(parent)
        parent.wait(5)
        assert all(_wait_gone(k) for k in kids)
    finally:
        _cleanup(parent, *kids)


def test_a_pid_we_may_not_signal_counts_as_alive(monkeypatch):
    import ps.util

    def denied(pid, sig):
        raise PermissionError
    monkeypatch.setattr(ps.util.os, "kill", denied)
    assert ps.util._pid_alive(12345) is True


def test_descendants_refuses_a_non_pid():
    with pytest.raises(ValueError):
        descendants(0)


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
