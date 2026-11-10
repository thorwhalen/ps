"""Tests for ps.isolated: locating and building environments, the worker, its two driver
styles, the pool and the one-shot runner. No models, no network: workers run this
interpreter, and the one real ``uv`` build is skipped when uv is missing.
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import ps
from ps import isolated
from ps.isolated import (
    EnvSpec,
    Runtime,
    RuntimeNotInstalled,
    Worker,
    WorkerError,
    WorkerPool,
    clean_environ,
    ensure_env,
    env_commands,
    is_env_ready,
    run_process,
    venv_python,
)

TESTS_DIR = str(Path(__file__).resolve().parent)


def _module_worker(**kw):
    """A worker running tests/_echo_driver.py with only ``ps`` exposed from the host."""
    return Worker(
        sys.executable,
        "_echo_driver",
        expose=(ps,),
        env={"PYTHONPATH": TESTS_DIR},
        **kw,
    )


def _adapter(tmp_path, source, name="a_adapter.py"):
    path = tmp_path / name
    path.write_text(source)
    return path


ECHO_ADAPTER = (
    "import os, sys\n"
    "print('import-time noise')\n"
    "def load(**kw): return kw\n"
    "def echo(state, **kw):\n"
    "    print('noise on stdout')\n"
    "    sys.__stdout__.write('noise on the original stdout\\n')\n"
    "    os.write(1, b'C-level noise on fd 1\\n')\n"
    "    return {'state': state, **kw}\n"
    "def environ(state):\n"
    "    return {k: os.environ.get(k) for k in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}\n"
    "def where(state): return {'prefix': sys.prefix, 'cwd': os.getcwd(), 'path': sys.path}\n"
    "def boom(state): raise KeyError('nope')\n"
    "def die(state):\n"
    "    sys.stderr.write('dying clue in the log\\n'); sys.stderr.flush(); os._exit(3)\n"
)


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
    assert (
        _runtime(default_dir=str(tmp_path / "dflt")).root() == tmp_path / "dflt" / "toy"
    )
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
    rt = _runtime(venv="sub/.venv")
    assert rt.interpreter() == tmp_path / "sub/.venv/Scripts/python.exe"


def test_venv_python_prefers_the_layout_that_exists(tmp_path):
    native = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    other = "bin/python" if os.name == "nt" else "Scripts/python.exe"
    assert venv_python(tmp_path) == tmp_path / native  # neither exists: this platform's
    (tmp_path / other).parent.mkdir(parents=True)
    (tmp_path / other).touch()
    assert venv_python(tmp_path) == tmp_path / other


# EnvSpec / ensure_env ---------------------------------------------------------------------


def test_digest_is_the_one_voxy_built_its_environments_with():
    """Same fields, same JSON: environments built before the move are not rebuilt."""
    import hashlib

    spec = EnvSpec("demo", ("numpy",), python="3.11", torch_backend="cpu")
    content = {
        "name": "demo",
        "requirements": ["numpy"],
        "python": "3.11",
        "torch_backend": "cpu",
    }
    expected = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    assert spec.digest() == expected
    assert (
        EnvSpec("demo", ("numpy<2",), python="3.11", torch_backend="cpu").digest()
        != expected
    )


def test_digest_ignores_fields_a_host_adds():
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class HostSpec(EnvSpec):
        env_vars: dict = field(default_factory=dict)

    plain = EnvSpec("demo", ("numpy",))
    assert HostSpec("demo", ("numpy",), env_vars={"X": "1"}).digest() == plain.digest()


def _fake_uv(commands):
    """A ``run`` that records commands and makes the venv interpreter ``uv venv`` would."""

    def run(cmd, **kwargs):
        commands.append(list(cmd))
        assert kwargs["check"] is True
        assert kwargs["env"]["UV_CONSTRAINT"] == ""  # host pins don't leak in
        if cmd[1] == "venv":
            python = venv_python(cmd[-1])
            python.parent.mkdir(parents=True, exist_ok=True)
            python.touch()

    return run


def test_ensure_env_builds_once_then_reuses_and_rebuilds_on_a_changed_spec(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("UV_CONSTRAINT", "/host/constraints.txt")
    spec, path, commands = (
        EnvSpec("demo", ("numpy",), torch_backend="cpu"),
        tmp_path / "demo",
        [],
    )
    run = _fake_uv(commands)
    python = ensure_env(spec, path, uv="uv", run=run, log=lambda m: None)
    assert python == venv_python(path) and is_env_ready(spec, path)
    assert [c[:2] for c in commands] == [["uv", "venv"], ["uv", "pip"]]
    assert commands[1][-3:] == ["--torch-backend", "cpu", "numpy"]

    ensure_env(spec, path, uv="uv", run=run, log=lambda m: None)
    assert len(commands) == 2  # ready: nothing runs

    changed = EnvSpec("demo", ("numpy<2",), torch_backend="cpu")
    assert not is_env_ready(changed, path)
    ensure_env(changed, path, uv="uv", run=run, log=lambda m: None)
    assert [c[:2] for c in commands[2:]] == [["uv", "pip"]]  # venv kept, reinstalled
    assert is_env_ready(changed, path) and not is_env_ready(spec, path)


def test_a_host_keeps_its_own_marker(tmp_path):
    spec, path = EnvSpec("demo", ("numpy",)), tmp_path / "demo"
    ensure_env(
        spec,
        path,
        uv="uv",
        run=_fake_uv([]),
        log=lambda m: None,
        marker=".host-env.json",
    )
    assert (path / ".host-env.json").exists()
    assert is_env_ready(spec, path, marker=".host-env.json") and not is_env_ready(
        spec, path
    )


def test_an_unbuilt_env_without_build_raises_with_the_commands(tmp_path):
    spec, path = EnvSpec("demo", ("numpy<2",), python="3.11"), tmp_path / "demo env"
    with pytest.raises(RuntimeNotInstalled) as err:
        ensure_env(spec, path, build=False)
    message = str(err.value)
    for command in env_commands(spec, path):
        assert isolated._shell_join(command) in message
    assert "numpy<2" in message


def test_a_missing_uv_says_how_to_install_it(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeNotInstalled, match="astral.sh/uv"):
        ensure_env(EnvSpec("demo", ()), tmp_path / "demo")


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_a_real_uv_env_runs_an_adapter_in_isolation(tmp_path, monkeypatch):
    """No requirements and this interpreter as the base: no download, about a second."""
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    monkeypatch.setenv("UV_OFFLINE", "1")
    spec, path = EnvSpec("bare", (), python=sys.executable), tmp_path / "bare"
    python = ensure_env(spec, path, log=lambda m: None)
    assert is_env_ready(spec, path)
    with Worker(
        python, _adapter(tmp_path, ECHO_ADAPTER), log_path=tmp_path / "w.log"
    ) as w:
        where = w.call("where")
    assert Path(where["prefix"]).resolve() == path.resolve()
    assert not any(Path(p).name == "ps" for p in where["path"])


# Worker: adapter files (voxy's style) -----------------------------------------------------


def test_adapter_worker_round_trip_survives_stdout_noise(tmp_path):
    log = tmp_path / "w.log"
    with Worker(
        sys.executable, _adapter(tmp_path, ECHO_ADAPTER), init={"a": 1}, log_path=log
    ) as w:
        assert w.call("echo", b=2) == {"state": {"a": 1}, "b": 2}
        assert w.call({"op": "echo", "c": 3}) == {"state": {"a": 1}, "c": 3}
    text = log.read_text()
    for noise in (
        "import-time noise",
        "noise on stdout",
        "original stdout",
        "C-level noise",
    ):
        assert noise in text


def test_adapter_worker_reports_errors_and_keeps_serving(tmp_path):
    with Worker(
        sys.executable, _adapter(tmp_path, ECHO_ADAPTER), log_path=tmp_path / "w.log"
    ) as w:
        with pytest.raises(WorkerError, match="KeyError: 'nope'"):
            w.call("boom")
        with pytest.raises(WorkerError, match="Unknown operation '_private'"):
            w.call("_private")
        assert w.running
        assert w.call("echo") == {"state": {}}


def test_a_worker_that_fails_to_load_raises_and_stops(tmp_path):
    adapter = _adapter(tmp_path, "def load(**kw): raise ImportError('no torch here')\n")
    worker = Worker(sys.executable, adapter, log_path=tmp_path / "w.log")
    with pytest.raises(WorkerError, match="no torch here"):
        worker.call("anything")
    assert not worker.running


def test_a_dying_worker_puts_its_log_tail_in_the_error(tmp_path):
    worker = Worker(
        sys.executable, _adapter(tmp_path, ECHO_ADAPTER), log_path=tmp_path / "w.log"
    )
    with pytest.raises(
        WorkerError, match=r"(?s)exited \(code 3\).*dying clue in the log"
    ):
        worker.call("die")
    assert not worker.alive
    assert worker.call("echo", x=1) == {"state": {}, "x": 1}, (
        "the next call starts a fresh one"
    )
    worker.close()


def test_host_python_variables_do_not_reach_the_worker(tmp_path, monkeypatch):
    """A leaked PYTHONHOME would not even let the worker start."""
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages-that-must-not-leak")
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "no-such-home"))
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    with Worker(
        sys.executable, _adapter(tmp_path, ECHO_ADAPTER), log_path=tmp_path / "w.log"
    ) as w:
        assert w.call("environ") == {
            "PYTHONPATH": None,
            "PYTHONHOME": None,
            "VIRTUAL_ENV": None,
        }


def test_the_working_directory_shadows_no_import(tmp_path):
    with Worker(
        sys.executable, _adapter(tmp_path, ECHO_ADAPTER), log_path=tmp_path / "w.log"
    ) as w:
        where = w.call("where")
    assert os.listdir(where["cwd"]) == []
    assert (
        "" not in where["path"]
        and str(Path(isolated.__file__).parent) not in where["path"]
    )


def test_close_from_another_thread_waits_for_the_call_in_flight(tmp_path):
    worker = Worker(
        sys.executable, _adapter(tmp_path, ECHO_ADAPTER), log_path=tmp_path / "w.log"
    )
    in_call, release = threading.Event(), threading.Event()

    def call_in_flight():  # holds the lock as call() does
        with worker._lock:
            in_call.set()
            release.wait(5)

    caller = threading.Thread(target=call_in_flight)
    caller.start()
    in_call.wait(5)
    closer = threading.Thread(target=worker.close)
    closer.start()
    closer.join(0.2)
    assert closer.is_alive()  # waiting for the call
    release.set()
    closer.join(5)
    caller.join(5)
    assert not closer.is_alive()


def test_a_closed_pipe_is_reported_as_the_worker_exiting(tmp_path):
    worker = Worker(
        sys.executable, _adapter(tmp_path, ECHO_ADAPTER), log_path=tmp_path / "w.log"
    )
    worker.call("echo")
    worker._proc.stdin.close()  # as a close() racing the write would leave it
    with pytest.raises(WorkerError, match="exited"):
        worker.call("echo")
    worker.close()


def test_a_request_that_cannot_be_serialised_leaves_the_worker_alive(tmp_path):
    worker = Worker(
        sys.executable, _adapter(tmp_path, ECHO_ADAPTER), log_path=tmp_path / "w.log"
    )
    worker.call("echo")
    circular = {}
    circular["self"] = circular
    with pytest.raises(ValueError, match="[Cc]ircular"):
        worker.call("echo", x=circular)
    assert worker.running
    worker.close()


# Worker: module drivers (arioso's style) --------------------------------------------------


def test_module_worker_round_trip_survives_stdout_noise_and_errors(tmp_path):
    with _module_worker(log_path=tmp_path / "w.log") as w:
        assert w.call({"a": 1})["echo"] == {"a": 1}
        with pytest.raises(WorkerError, match="asked to fail"):
            w.call({"fail": True})
        assert w.alive, "a failed request must not kill the worker"
        assert w.call({"b": 2})["echo"] == {"b": 2}
    assert not w.alive
    assert "C-level noise on fd 1" in (tmp_path / "w.log").read_text()


def test_module_worker_path_holds_only_the_exposed_packages_and_what_was_passed(
    monkeypatch,
):
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages-that-must-not-leak")
    with _module_worker() as w:
        reply = w.call({})
    path = reply["pythonpath"].split(os.pathsep)
    assert "/host/site-packages-that-must-not-leak" not in path
    assert path[-1] == TESTS_DIR
    assert os.listdir(path[0]) == ["ps"]
    assert Path(reply["ps_file"]).parent.parent == Path(path[0]), (
        "ps came from the exposed dir"
    )


def test_module_worker_reports_a_driver_that_dies_with_its_log(tmp_path):
    with _module_worker(log_path=tmp_path / "w.log") as w:
        with pytest.raises(WorkerError, match="(?s)exited.*dying clue"):
            w.call({"die": True})


def test_a_missing_driver_module_is_reported():
    with pytest.raises(WorkerError, match="(?s)exited.*No module named"):
        Worker(sys.executable, "ps_no_such_driver_module").call({})


def test_exposing_copies_when_symlinks_are_not_allowed(monkeypatch):
    def refuse(*a, **k):
        raise OSError("no symlink rights")

    monkeypatch.setattr(Path, "symlink_to", refuse)
    monkeypatch.setattr(isolated, "_EXPOSED_DIRS", {})
    base = Path(isolated._exposed_dir([ps]))
    assert os.listdir(base) == ["ps"] and not (base / "ps").is_symlink()
    assert (base / "ps" / "isolated.py").read_text() == Path(
        isolated.__file__
    ).read_text()
    assert not (base / "ps" / "__pycache__").exists()


class _FakeProc:
    """A worker process whose reply is scripted (a line, or an exception from readline)."""

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
        "m",
        log_path=tmp_path / "w.log",
        spawn=lambda *a, **k: procs.append(_FakeProc(reply)) or procs[-1],
    )
    with pytest.raises(error):
        w.call({"x": 1})
    assert procs[0].killed and not w.alive, (
        "a stale reply must never be read by the next call"
    )


def test_spawn_gets_the_command_and_a_clean_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/host")
    seen = []

    def spawn(argv, **kwargs):
        seen.append((argv, kwargs))
        return _FakeProc('{"ok": true, "result": 7}\n')

    w = Worker(
        "py",
        "drv.mod",
        env={"X": "1"},
        cwd=tmp_path,
        log_path=tmp_path / "w.log",
        spawn=spawn,
    )
    assert w.call({}) == 7
    argv, kwargs = seen[0]
    assert argv == ["py", "-u", "-m", "drv.mod"]
    assert kwargs["cwd"] == str(tmp_path) and kwargs["env"]["X"] == "1"
    assert "PYTHONPATH" not in kwargs["env"]


# WorkerPool -------------------------------------------------------------------------------


def test_worker_pool_makes_each_worker_once_across_threads():
    made, closed = [], []

    class FakeWorker:
        def __init__(self, key):
            self.key = key

        def close(self):
            closed.append(self.key)

    def make(model, threads=None):
        made.append((model, threads))
        time.sleep(0.05)  # an env build: two unguarded threads would both run it
        return FakeWorker((model, threads))

    pool = WorkerPool(make)
    got = []
    threads = [
        threading.Thread(target=lambda: got.append(pool.get("base"))) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made) == 1 and len({id(w) for w in got}) == 1
    assert pool.get("small", threads=2) is not got[0]
    pool.close()
    assert sorted(closed) == [("base", None), ("small", 2)]
    pool.get("base")
    assert len(made) == 3  # a fresh worker after close


# run_process ------------------------------------------------------------------------------


def test_run_process_interleaves_output_and_cleans_the_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PYTHONPATH", "/host")
    code = (
        "import os, sys\n"
        "print('out', flush=True); print('err', file=sys.stderr, flush=True)\n"
        "print(os.environ.get('PYTHONPATH'), os.environ.get('MINE'), os.getcwd(), flush=True)\n"
        "sys.exit(2)"
    )
    returncode, output = run_process(
        [sys.executable, "-c", code], cwd=tmp_path, env={"MINE": "1"}
    )
    assert returncode == 2
    lines = output.splitlines()
    assert lines[:2] == ["out", "err"]
    assert lines[2].split(" ", 2)[:2] == ["None", "1"]
    assert Path(lines[2].split(" ", 2)[2]).resolve() == tmp_path.resolve()


def test_run_process_kills_on_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)


def test_clean_environ_defaults_to_this_process(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/v")
    monkeypatch.setenv("PS_TEST_KEEP", "1")
    env = clean_environ()
    assert "VIRTUAL_ENV" not in env and env["PS_TEST_KEEP"] == "1"
