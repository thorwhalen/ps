"""Tests for ps.isolated's lock helpers and the ``build``/``status``/``lock`` command line."""

import json
import shutil
import subprocess
import sys

import pytest

from ps import isolated
from argparse import ArgumentParser
from types import SimpleNamespace

from ps.isolated import (
    DFLT_ENV_MARKER,
    EnvCommandError,
    add_env_commands,
    run_env_command,
    HOST_UV_VARS,
    EnvSpec,
    cli,
    compile_lock,
    lock_requirements,
)


def _built(path, spec, marker=DFLT_ENV_MARKER):
    """Make ``path`` look like ps built ``spec`` there (marker + interpreter)."""
    python = isolated.venv_python(path)
    python.parent.mkdir(parents=True)
    python.write_text("")
    (path / marker).write_text(json.dumps({"digest": spec.digest()}))


def _run(argv, specs, tmp_path, **kw):
    lines = []
    code = cli(specs, env_dir=lambda name: tmp_path / name, argv=argv, out=lines.append, **kw)
    return code, lines


def test_build_builds_the_named_environments_or_all(tmp_path):
    built = []
    specs = {"a": EnvSpec("a", ("x",)), "b": EnvSpec("b", ("y",))}
    fake = lambda spec, path: built.append((spec.name, path)) or path / "py"
    assert _run(["build", "b"], specs, tmp_path, build=fake) == (0, [f"b\t{tmp_path / 'b' / 'py'}"])
    assert built == [("b", tmp_path / "b")]
    built.clear()
    code, lines = _run(["build", "--all"], specs, tmp_path, build=fake)
    assert lines == [f"a\t{tmp_path / 'a' / 'py'}", f"b\t{tmp_path / 'b' / 'py'}"]


def test_a_bare_build_builds_nothing(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        _run(["build"], {"a": EnvSpec("a", ())}, tmp_path, build=lambda s, p: 1 / 0)
    assert error.value.code == 2 and "--all" in capsys.readouterr().err


def test_env_dir_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    lines = []
    cli({"a": EnvSpec("a", ())}, env_dir=lambda n: "~/envs/" + n, argv=["status"], out=lines.append)
    assert lines == [f"a: not built ({tmp_path / 'envs' / 'a'})"]


def test_commands_mount_in_a_host_parser_and_run_without_reparsing(tmp_path):
    parser = ArgumentParser(prog="host")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("speak")
    add_env_commands(commands, {"a": EnvSpec("a", ())})
    args = parser.parse_args(["build", "a"])
    lines = []
    run_env_command(args, {"a": EnvSpec("a", ())}, env_dir=lambda n: tmp_path / n,
                    build=lambda s, p: "py", out=lines.append)
    assert lines == ["a\tpy"]
    # a host with its own command shape (voxy's `setup [name ...]`: all by default)
    lines.clear()
    run_env_command(SimpleNamespace(env_command="build", names=[], all=True),
                    {"a": EnvSpec("a", ()), "b": EnvSpec("b", ())},
                    env_dir=lambda n: tmp_path / n, build=lambda s, p: "py", out=lines.append)
    assert lines == ["a\tpy", "b\tpy"]
    with pytest.raises(EnvCommandError, match="unknown"):
        run_env_command(SimpleNamespace(env_command="build", names=["z"]), {"a": EnvSpec("a", ())},
                        env_dir=lambda n: tmp_path / n)


def test_a_spec_that_fails_reads_as_no_spec_in_status(tmp_path):
    def missing_lock():
        raise FileNotFoundError("a.lock")

    code, lines = _run(["status"], {"a": missing_lock, "b": EnvSpec("b", ())}, tmp_path)
    assert lines[0] == "a: no spec (FileNotFoundError: a.lock)"
    assert lines[1] == f"b: not built ({tmp_path / 'b'})"


def test_an_error_inside_a_build_is_not_a_usage_error(tmp_path):
    def broken(spec, path):
        raise ValueError("bad requirement")

    with pytest.raises(ValueError, match="bad requirement"):
        _run(["build", "a"], {"a": EnvSpec("a", ())}, tmp_path, build=broken)


def test_the_default_build_is_ensure_env_with_the_host_marker(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(isolated, "ensure_env", lambda spec, path, marker: seen.append(marker) or "py")
    _run(["build", "a"], {"a": EnvSpec("a", ())}, tmp_path, marker=".host.json")
    assert seen == [".host.json"]


def test_status_reads_each_marker_and_evaluates_lazy_specs(tmp_path):
    a, b = EnvSpec("a", ("x",)), EnvSpec("b", ("y",))
    _built(tmp_path / "a", a, marker=".m.json")
    _built(tmp_path / "b", EnvSpec("b", ("old",)), marker=".m.json")  # stale
    code, lines = _run(["status"], {"a": a, "b": lambda: b}, tmp_path, marker=".m.json")
    assert lines == [f"a: built ({tmp_path / 'a'})", f"b: not built ({tmp_path / 'b'})"]


def test_a_lazy_spec_is_not_made_for_another_name(tmp_path):
    def broken():
        raise FileNotFoundError("no lock yet")

    built = []
    specs = {"a": EnvSpec("a", ()), "b": broken}
    _run(["build", "a"], specs, tmp_path, build=lambda s, p: built.append(s.name) or p)
    assert built == ["a"]


def test_unknown_names_are_refused_with_the_choices(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        _run(["build", "nope"], {"a": EnvSpec("a", ())}, tmp_path, build=lambda s, p: p)
    assert error.value.code == 2 and "choose from ['a']" in capsys.readouterr().err


def test_lock_exists_only_with_locks_and_calls_them(tmp_path, capsys):
    specs = {"a": EnvSpec("a", ())}
    with pytest.raises(SystemExit):
        _run(["lock", "a"], specs, tmp_path)
    assert "invalid choice: 'lock'" in capsys.readouterr().err
    assert _run(["lock", "a"], specs, tmp_path, locks={"a": lambda: tmp_path / "a.lock"}) == (
        0, [str(tmp_path / "a.lock")])
    with pytest.raises(SystemExit):
        _run(["lock", "b"], specs, tmp_path, locks={"a": lambda: 1})
    with pytest.raises(ValueError, match="not in specs"):
        _run(["status"], specs, tmp_path, locks={"z": lambda: 1})


def test_compile_lock_runs_uv_with_the_host_pins_cleared(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_CONSTRAINT", "/host/constraints.txt")
    seen = {}

    def run(command, **kw):
        seen.update(command=command, **kw)

    path = compile_lock(["numpy<2", "torch"], tmp_path / "x.lock", python="3.11",
                        custom_command="python -m host lock x", run=run)
    assert path == tmp_path / "x.lock"
    command = seen["command"]
    assert command[:5] == ["uv", "pip", "compile", "-", "--universal"]
    assert command[command.index("--python-version") + 1] == "3.11"
    assert command[command.index("--output-file") + 1] == str(tmp_path / "x.lock")
    assert command[-2:] == ["--custom-compile-command", "python -m host lock x"]
    assert seen["input"] == "numpy<2\ntorch\n" and seen["check"]
    assert all(seen["env"][v] == "" for v in HOST_UV_VARS)


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_a_real_lock_round_trips_into_a_spec(tmp_path):
    """uv resolves offline from its cache if it can; skip when it cannot reach an index."""
    lock = tmp_path / "x.lock"
    try:
        compile_lock(["six==1.16.0"], lock, python="3.11")
    except subprocess.CalledProcessError:
        pytest.skip("uv could not resolve (no network or cache)")
    assert lock_requirements(lock.read_text()) == ("six==1.16.0",)


def test_compile_lock_without_uv_says_how_to_get_it(tmp_path):
    def run(command, **kw):
        raise FileNotFoundError(command[0])

    with pytest.raises(isolated.RuntimeNotInstalled, match="Resolving a lock needs uv"):
        compile_lock(["x"], tmp_path / "l", python="3.11", run=run)


def test_lock_requirements_keeps_editables_and_refuses_index_options():
    assert lock_requirements("-e ./pkg\nnumpy==2.0\n") == ("-e ./pkg", "numpy==2.0")
    with pytest.raises(ValueError, match="--index-url"):
        lock_requirements("--index-url https://example.org/simple\nnumpy==2.0\n")


def test_lock_requirements_drops_comments_keeps_markers():
    text = (
        "# This file was autogenerated by uv\n"
        "numpy==2.1.0\n"
        "    # via x\n"
        "torch==2.5.0 ; sys_platform == 'darwin'  # via y\n"
        "\n"
    )
    assert lock_requirements(text) == ("numpy==2.1.0", "torch==2.5.0 ; sys_platform == 'darwin'")
