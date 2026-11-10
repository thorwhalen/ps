"""Tests for ps.isolated: locating and building environments, the adapter worker, and
the one-shot runner. No models, no network: workers run this interpreter, and the one
real ``uv`` build is skipped when uv is missing.
"""

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ps import isolated
from ps.isolated import (
    LOG_TAIL_CHARS,
    EnvSpec,
    Runtime,
    RuntimeNotInstalled,
    Worker,
    WorkerError,
    clean_environ,
    ensure_env,
    is_env_ready,
    run_process,
    venv_python,
)

posix_only = pytest.mark.skipif(
    os.name == "nt", reason="checks a pid with os.kill(pid, 0)"
)

ECHO_ADAPTER = """\
import os, sys, threading, time
print('import-time noise')

def load(**kw):
    return kw

def echo(state, **kw):
    print('noise on stdout')
    sys.__stdout__.write('noise on the original stdout\\n')
    os.write(1, b'C-level noise on fd 1\\n')
    return {'state': state, **kw}

def nothing(state):
    return None

def zero(state):
    return 0

def environ(state):
    return {k: os.environ.get(k) for k in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}

def where(state):
    return {'prefix': sys.prefix, 'cwd': os.getcwd(), 'path': sys.path, 'pid': os.getpid()}

def boom(state):
    raise KeyError('nope')

def die(state, pad=0):
    sys.stderr.write('x' * pad + 'dying clue in the log\\n'); sys.stderr.flush(); os._exit(3)

def sleep(state, seconds):
    time.sleep(seconds); return {'slept': seconds}

def linger(state):
    # a non-daemon thread: the process will not end when its stdin closes
    threading.Thread(target=time.sleep, args=(60,)).start(); return {}
"""


@pytest.fixture
def echo(tmp_path):
    path = tmp_path / "adapters" / "echo_adapter.py"
    path.parent.mkdir()
    path.write_text(ECHO_ADAPTER)
    return path


def _worker(adapter, tmp_path, **kw):
    return Worker(sys.executable, adapter, stderr=tmp_path / "w.log", **kw)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pid, seconds=10):
    deadline = time.monotonic() + seconds
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _pid_alive(pid)


# Runtime -------------------------------------------------------------------------------


def _runtime(**kw):
    kw = {"default_dir": "/unused", **kw}
    return Runtime(
        name="toy",
        env_var="PS_TOY_DIR",
        base_dir_var="PS_TOY_BASE",
        setup="install into {root}",
        **kw,
    )


def test_runtime_root_order_own_variable_then_base_variable_then_default(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("PS_TOY_DIR", raising=False)
    monkeypatch.delenv("PS_TOY_BASE", raising=False)
    assert _runtime(default_dir=str(tmp_path / "d")).root() == tmp_path / "d" / "toy"
    monkeypatch.setenv("PS_TOY_BASE", str(tmp_path / "all"))
    assert _runtime().root() == tmp_path / "all" / "toy"
    monkeypatch.setenv("PS_TOY_DIR", str(tmp_path / "mine"))
    assert _runtime().root() == tmp_path / "mine"


def test_missing_runtime_raises_with_the_setup_commands(monkeypatch, tmp_path):
    monkeypatch.setenv("PS_TOY_DIR", str(tmp_path))
    rt = _runtime()
    assert not rt.installed()
    with pytest.raises(
        RuntimeNotInstalled, match=re.escape(f"install into {tmp_path}")
    ) as err:
        rt.interpreter()
    assert "$PS_TOY_DIR" in str(err.value)


def test_installed_runtime_gives_its_interpreter_and_workdir(monkeypatch, tmp_path):
    monkeypatch.setenv("PS_TOY_DIR", str(tmp_path))
    (tmp_path / "sub" / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / "sub" / ".venv" / "bin" / "python").touch()
    rt = _runtime(python="sub/.venv/bin/python", cwd="sub")
    assert rt.interpreter() == tmp_path / "sub/.venv/bin/python"
    assert rt.workdir() == tmp_path / "sub"


def test_runtime_finds_a_windows_layout_venv(monkeypatch, tmp_path):
    monkeypatch.setenv("PS_TOY_DIR", str(tmp_path))
    (tmp_path / "sub" / ".venv" / "Scripts").mkdir(parents=True)
    (tmp_path / "sub" / ".venv" / "Scripts" / "python.exe").touch()
    assert _runtime(venv="sub/.venv").interpreter() == (
        tmp_path / "sub/.venv/Scripts/python.exe"
    )


def test_venv_python_prefers_the_layout_that_exists(tmp_path):
    native = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    other = "bin/python" if os.name == "nt" else "Scripts/python.exe"
    assert venv_python(tmp_path) == tmp_path / native  # neither exists: this platform's
    (tmp_path / other).parent.mkdir(parents=True)
    (tmp_path / other).touch()
    assert venv_python(tmp_path) == tmp_path / other


# EnvSpec / ensure_env -------------------------------------------------------------------


def test_digest_is_the_one_voxy_built_its_environments_with():
    """Same fields, same JSON: environments built before the move are not rebuilt."""
    spec = EnvSpec("demo", ("numpy",), python="3.11", torch_backend="cpu")
    content = {
        "name": "demo",
        "requirements": ["numpy"],
        "python": "3.11",
        "torch_backend": "cpu",
    }
    blob = json.dumps(content, sort_keys=True).encode()
    assert spec.digest() == hashlib.sha256(blob).hexdigest()[:12]


def test_digest_ignores_fields_a_host_adds():
    @dataclass(frozen=True)
    class HostSpec(EnvSpec):
        env_vars: dict = field(default_factory=dict)

    plain = EnvSpec("demo", ("numpy",))
    assert HostSpec("demo", ("numpy",), env_vars={"X": "1"}).digest() == plain.digest()


@pytest.fixture
def fake_uv(monkeypatch):
    """A ``run`` that records commands and fakes what uv would leave on disk."""
    monkeypatch.setattr(isolated, "_find_uv", lambda: "uv")
    commands = []

    def run(cmd, **kwargs):
        commands.append(list(cmd))
        assert kwargs["check"] is True
        for var in isolated.HOST_UV_VARS:  # host pins don't leak in
            assert kwargs["env"][var] == ""
        if cmd[1] == "venv":
            path = Path(cmd[-1])
            assert not path.exists() or not any(path.iterdir()), "a fresh venv"
            path.mkdir(parents=True, exist_ok=True)
            (path / "pyvenv.cfg").write_text(f"version = {cmd[3]}\n")
            venv_python(path).parent.mkdir(parents=True)
            venv_python(path).touch()
        else:
            site = Path(cmd[cmd.index("--python") + 1]).parent.parent / "site"
            site.mkdir(exist_ok=True)
            for req in (a for a in cmd[5:] if not a.startswith("-") and a != "cpu"):
                (site / req).touch()

    run.commands = commands
    return run


def _build(spec, path, run, **kw):
    return ensure_env(spec, path, run=run, log=lambda m: None, **kw)


def test_ensure_env_builds_once_then_reuses(tmp_path, fake_uv, monkeypatch):
    for var in isolated.HOST_UV_VARS:
        monkeypatch.setenv(var, "/host/pins.txt")
    spec, path = EnvSpec("demo", ("numpy",), torch_backend="cpu"), tmp_path / "demo"
    assert _build(spec, path, fake_uv) == venv_python(path)
    assert is_env_ready(spec, path)
    assert [c[:2] for c in fake_uv.commands] == [["uv", "venv"], ["uv", "pip"]]
    assert fake_uv.commands[1][-3:] == ["--torch-backend", "cpu", "numpy"]
    _build(spec, path, fake_uv)
    assert len(fake_uv.commands) == 2  # ready: nothing runs


def test_a_changed_python_rebuilds_the_venv_from_scratch(tmp_path, fake_uv):
    path = tmp_path / "demo"
    _build(EnvSpec("demo", ("numpy",), python="3.11"), path, fake_uv)
    (path / "left-over").touch()
    changed = EnvSpec("demo", ("numpy",), python="3.12")
    _build(changed, path, fake_uv)
    assert [c[:2] for c in fake_uv.commands[2:]] == [["uv", "venv"], ["uv", "pip"]]
    assert "3.12" in (path / "pyvenv.cfg").read_text()
    assert not (path / "left-over").exists() and is_env_ready(changed, path)


def test_a_dropped_requirement_is_gone_after_the_rebuild(tmp_path, fake_uv):
    path = tmp_path / "demo"
    _build(EnvSpec("demo", ("numpy", "torch")), path, fake_uv)
    site = venv_python(path).parent.parent / "site"
    assert sorted(os.listdir(site)) == ["numpy", "torch"]
    _build(EnvSpec("demo", ("numpy",)), path, fake_uv)
    assert sorted(os.listdir(site)) == ["numpy"]


def test_ensure_env_refuses_to_replace_a_folder_that_is_not_a_venv(tmp_path, fake_uv):
    path = tmp_path / "precious"
    path.mkdir()
    (path / "data.txt").write_text("keep me")
    with pytest.raises(RuntimeError, match="not a virtual environment"):
        _build(EnvSpec("demo", ()), path, fake_uv)
    assert (path / "data.txt").read_text() == "keep me" and not fake_uv.commands


def test_the_build_notice_shows_the_commands_with_the_pins_cleared(tmp_path, fake_uv):
    said = []
    ensure_env(
        EnvSpec("demo", ("numpy<2",)), tmp_path / "e", run=fake_uv, log=said.append
    )
    assert "numpy<2" in said[0] and "uv venv" in said[0]
    for var in isolated.HOST_UV_VARS:
        assert var in said[0]


def test_a_host_keeps_its_own_marker(tmp_path, fake_uv):
    spec, path = EnvSpec("demo", ("numpy",)), tmp_path / "demo"
    _build(spec, path, fake_uv, marker=".host-env.json")
    assert (path / ".host-env.json").exists()
    assert is_env_ready(spec, path, marker=".host-env.json")
    assert not is_env_ready(spec, path)


def test_a_missing_uv_says_how_to_install_it(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeNotInstalled, match="astral.sh/uv"):
        ensure_env(EnvSpec("demo", ()), tmp_path / "demo")


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_a_real_uv_env_runs_an_adapter_in_isolation(tmp_path, monkeypatch, echo):
    """No requirements and this interpreter as the base: no download, about a second."""
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    monkeypatch.setenv("UV_OFFLINE", "1")
    spec, path = EnvSpec("bare", (), python=sys.executable), tmp_path / "bare"
    python = ensure_env(spec, path, log=lambda m: None)
    assert is_env_ready(spec, path)
    with Worker(python, echo, stderr=tmp_path / "w.log") as w:
        where = w.call("where")
    assert Path(where["prefix"]).resolve() == path.resolve()


# Worker ---------------------------------------------------------------------------------


def test_round_trip_survives_stdout_noise_at_import_and_call_time(echo, tmp_path):
    with _worker(echo, tmp_path, init={"a": 1}) as w:
        assert w.call("echo", b=2) == {"state": {"a": 1}, "b": 2}
    text = (tmp_path / "w.log").read_text()
    for noise in ("import-time", "noise on stdout", "original stdout", "C-level"):
        assert noise in text


def test_a_string_path_is_an_adapter_too(echo, tmp_path):
    with _worker(str(echo), tmp_path) as w:
        assert w.call("echo", x=1) == {"state": {}, "x": 1}


def test_only_none_becomes_an_empty_result(echo, tmp_path):
    with _worker(echo, tmp_path) as w:
        assert w.call("nothing") == {} and w.call("zero") == 0


def test_errors_are_reported_and_the_worker_keeps_serving(echo, tmp_path):
    with _worker(echo, tmp_path) as w:
        with pytest.raises(WorkerError, match="KeyError: 'nope'"):
            w.call("boom")
        with pytest.raises(WorkerError, match="Unknown operation '_private'"):
            w.call("_private")
        assert w.running and w.call("echo") == {"state": {}}


def test_a_worker_that_fails_to_load_raises_and_stops(tmp_path):
    adapter = tmp_path / "b_adapter.py"
    adapter.write_text("def load(**kw): raise ImportError('no torch here')\n")
    worker = _worker(adapter, tmp_path)
    with pytest.raises(WorkerError, match="no torch here"):
        worker.call("anything")
    assert not worker.running


def test_a_dying_worker_puts_its_log_tail_in_the_error(echo, tmp_path):
    worker = _worker(echo, tmp_path)
    with pytest.raises(WorkerError, match=r"(?s)exited \(code 3\).*dying clue") as err:
        worker.call("die", pad=3 * LOG_TAIL_CHARS)
    assert len(worker.log_tail()) == LOG_TAIL_CHARS
    assert worker.log_tail().endswith("dying clue in the log\n")
    assert len(str(err.value)) < 2 * LOG_TAIL_CHARS
    assert not worker.running
    assert worker.call("echo", x=1) == {"state": {}, "x": 1}, "a fresh one starts"
    worker.close()


def test_stderr_none_leaves_the_output_on_the_hosts_terminal(echo, capfd):
    worker = Worker(sys.executable, echo, stderr=None)
    assert worker.call("echo") == {"state": {}}
    with pytest.raises(WorkerError, match="stderr went to this process") as err:
        worker.call("die")
    worker.close()
    assert "dying clue" in capfd.readouterr().err
    assert "dying clue" not in str(err.value) and worker.log_path is None


def test_the_default_log_is_made_on_start_and_deleted_on_close(echo):
    worker = Worker(sys.executable, echo)
    assert worker.log_path is None
    worker.call("echo")
    log = worker.log_path
    assert "import-time noise" in log.read_text()
    worker.close()
    assert not log.exists() and worker.log_path is None


def test_an_explicit_log_is_kept(echo, tmp_path):
    with _worker(echo, tmp_path) as w:
        w.call("echo")
    assert (tmp_path / "w.log").exists()


def test_host_python_variables_do_not_reach_the_worker(echo, tmp_path, monkeypatch):
    """A leaked PYTHONHOME would not even let the worker start."""
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages-that-must-not-leak")
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "no-such-home"))
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    with _worker(echo, tmp_path) as w:
        assert w.call("environ") == dict.fromkeys(
            ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
        )


def test_the_path_has_the_adapters_folder_and_not_the_working_directory(echo, tmp_path):
    with _worker(echo, tmp_path) as w:
        where = w.call("where")
    assert os.listdir(where["cwd"]) == []
    assert Path(where["path"][0]) == echo.parent
    assert "" not in where["path"]
    assert str(Path(isolated.__file__).parent) not in where["path"]


def test_adapters_can_share_a_helper_module(tmp_path):
    (tmp_path / "shared_helper.py").write_text("def double(x): return 2 * x\n")
    adapter = tmp_path / "h_adapter.py"
    adapter.write_text(
        "from shared_helper import double\n"
        "def load(**kw): return None\n"
        "def twice(state, x): return double(x)\n"
    )
    with _worker(adapter, tmp_path) as w:
        assert w.call("twice", x=21) == 42


def test_close_from_another_thread_waits_for_the_call_in_flight(echo, tmp_path):
    worker = _worker(echo, tmp_path)
    worker.call("echo")
    results = []
    caller = threading.Thread(
        target=lambda: results.append(worker.call("sleep", seconds=1))
    )
    caller.start()
    time.sleep(0.3)  # the call is in flight
    # with a short timeout, a close that did not wait would kill the call in flight
    closer = threading.Thread(target=worker.close, kwargs={"timeout": 0.2})
    closer.start()
    closer.join(0.3)
    assert closer.is_alive(), "close waits for the call"
    caller.join(10)
    closer.join(10)
    assert results == [{"slept": 1}] and not worker.running


def test_kill_does_not_wait_for_the_call_in_flight(echo, tmp_path):
    worker = _worker(echo, tmp_path)
    worker.call("echo")
    errors = []

    def call():
        try:
            worker.call("sleep", seconds=30)
        except WorkerError as error:
            errors.append(error)

    caller = threading.Thread(target=call)
    caller.start()
    time.sleep(0.3)
    started = time.monotonic()
    worker.kill()
    caller.join(10)
    assert time.monotonic() - started < 5 and len(errors) == 1
    assert not worker.running


def test_closing_a_stuck_worker_kills_it_after_the_timeout(echo, tmp_path):
    worker = _worker(echo, tmp_path)
    worker.call("linger")
    proc = worker._proc
    started = time.monotonic()
    worker.close(timeout=0.5)
    assert time.monotonic() - started < 5
    assert proc.poll() is not None and not worker.running


def test_a_closed_pipe_is_reported_as_the_worker_exiting(echo, tmp_path):
    worker = _worker(echo, tmp_path)
    worker.call("echo")
    worker._proc.stdin.close()  # as a close() racing the write would leave it
    with pytest.raises(WorkerError, match="exited"):
        worker.call("echo")
    worker.close()


def test_a_request_that_cannot_be_serialised_leaves_the_worker_alive(echo, tmp_path):
    worker = _worker(echo, tmp_path)
    worker.call("echo")
    circular = {}
    circular["self"] = circular
    with pytest.raises(ValueError, match="[Cc]ircular"):
        worker.call("echo", x=circular)
    assert worker.running
    worker.close()


_HOST_SCRIPT = """\
import sys, threading, time
from ps.isolated import Worker
worker = Worker(sys.executable, sys.argv[1])
print(worker.call("where")["pid"], flush=True)
if sys.argv[2] == "in-flight":
    threading.Thread(target=worker.call, args=("sleep",), kwargs={"seconds": 60}, daemon=True).start()
    time.sleep(0.5)
"""


@pytest.mark.parametrize("state", ["idle", "in-flight"])
def test_the_host_exits_and_takes_its_workers_with_it(echo, tmp_path, state):
    """At exit every worker is killed, even with a daemon thread inside call()."""
    script = tmp_path / "host.py"
    script.write_text(_HOST_SCRIPT)
    started = time.monotonic()
    out = subprocess.run(
        [sys.executable, str(script), str(echo), state],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert time.monotonic() - started < 20
    pid = int(out.stdout.split()[0])
    if os.name != "nt":
        assert _wait_dead(pid), "the worker outlived its host"


class _FakeProc:
    """A worker process whose replies are scripted (a line, or an exception)."""

    def __init__(self, reply):
        self.stdin, self.killed = io.StringIO(), False
        self.stdout = self
        self._reply = reply

    def readline(self):
        if isinstance(self._reply, BaseException):
            raise self._reply
        return self._reply

    def poll(self):
        return 0 if self.killed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


@pytest.mark.parametrize(
    "reply, error",
    [
        (KeyboardInterrupt(), KeyboardInterrupt),
        ("garbage\n", WorkerError),
        ("[1]\n", WorkerError),
    ],
)
def test_an_interrupted_or_garbled_exchange_kills_the_worker(reply, error, tmp_path):
    procs = []
    w = Worker(
        "python",
        "m_adapter.py",
        stderr=tmp_path / "w.log",
        spawn=lambda *a, **k: procs.append(_FakeProc(reply)) or procs[-1],
    )
    with pytest.raises(error):
        w.call("x")
    assert procs[0].killed and not w.running, "a stale reply must never be read next"


def test_spawn_gets_the_command_and_a_clean_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/host")
    seen = []

    def spawn(argv, **kwargs):
        seen.append((argv, kwargs))
        return _FakeProc('{"ok": true, "result": 7}\n')

    w = Worker(
        "py",
        "a_adapter.py",
        init={"size": "s"},
        env={"X": "1"},
        cwd=tmp_path,
        stderr=tmp_path / "w.log",
        spawn=spawn,
    )
    assert w.call("op") == 7
    argv, kwargs = seen[0]
    assert argv[:3] == ["py", "-u", "-c"]
    assert argv[-2:] == [isolated.__file__, "a_adapter.py"]
    assert kwargs["cwd"] == str(tmp_path) and kwargs["env"]["X"] == "1"
    assert "PYTHONPATH" not in kwargs["env"]
    assert json.loads(w._proc.stdin.getvalue().splitlines()[0]) == {
        "op": "load",
        "size": "s",
    }


# run_process ----------------------------------------------------------------------------


def test_run_process_interleaves_output_and_cleans_the_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PYTHONPATH", "/host")
    code = (
        "import os, sys\n"
        "print('out', flush=True); print('err', file=sys.stderr, flush=True)\n"
        "print(os.environ.get('PYTHONPATH'), os.environ.get('MINE'), flush=True)\n"
        "print(os.getcwd(), flush=True)\n"
        "sys.exit(2)"
    )
    returncode, output = run_process(
        [sys.executable, "-c", code], cwd=tmp_path, env={"MINE": "1"}
    )
    lines = output.splitlines()
    assert returncode == 2 and lines[:3] == ["out", "err", "None 1"]
    assert Path(lines[3]).resolve() == tmp_path.resolve()


def test_run_process_raises_timeout_expired():
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)


@posix_only
def test_run_process_timeout_kills_the_grandchildren_too(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    code = textwrap.dedent(
        """
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open(sys.argv[1], "w").write(str(child.pid))
        time.sleep(60)
        """
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", code, pid_file], timeout=1)
    assert time.monotonic() - started < 10
    assert _wait_dead(int(pid_file.read_text())), "the grandchild survived"


def test_clean_environ_defaults_to_this_process(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/v")
    monkeypatch.setenv("PS_TEST_KEEP", "1")
    env = clean_environ()
    assert "VIRTUAL_ENV" not in env and env["PS_TEST_KEEP"] == "1"
