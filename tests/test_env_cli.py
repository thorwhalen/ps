"""Tests for ps.isolated's lock helpers, build_envs/env_status, and the build/status/lock cli."""

import json
import shutil
import subprocess

import pytest

from ps import isolated
from ps.isolated import (
    DFLT_ENV_MARKER,
    HOST_UV_VARS,
    EnvSpec,
    build_envs,
    cli,
    compile_lock,
    env_status,
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


def test_build_prints_each_interpreter_in_order(tmp_path):
    built = []
    specs = {"a": EnvSpec("a", ("x",)), "b": EnvSpec("b", ("y",))}
    fake = lambda spec, path: built.append((spec.name, path)) or path / "py"
    assert _run(["build", "b", "a"], specs, tmp_path, build=fake) == (
        0, [str(tmp_path / "b" / "py"), str(tmp_path / "a" / "py")])  # denote's output: the path
    assert built == [("b", tmp_path / "b"), ("a", tmp_path / "a")]


@pytest.mark.parametrize("argv", [["build"], ["build", "nope"], []])
def test_usage_errors_exit_2_and_build_nothing(argv, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        _run(argv, {"a": EnvSpec("a", ())}, tmp_path, build=lambda s, p: 1 / 0)
    assert error.value.code == 2


def test_the_default_build_is_ensure_env_with_the_host_marker(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(isolated, "ensure_env", lambda spec, path, marker: seen.append(marker) or "py")
    assert list(build_envs(["a"], {"a": EnvSpec("a", ())}, env_dir=lambda n: tmp_path / n,
                           marker=".host.json")) == [("a", "py")]
    assert seen == [".host.json"]


def test_build_envs_checks_every_name_first_and_makes_lazy_specs_on_demand(tmp_path):
    def broken():
        raise FileNotFoundError("no lock yet")

    built = []
    specs = {"a": EnvSpec("a", ()), "b": broken}
    fake = lambda s, p: built.append(s.name) or p
    with pytest.raises(ValueError, match="unknown environments \\['z'\\]"):
        list(build_envs(["a", "z"], specs, env_dir=lambda n: tmp_path / n, build=fake))
    assert built == []  # nothing built before the bad name was found
    list(build_envs(["a"], specs, env_dir=lambda n: tmp_path / n, build=fake))
    assert built == ["a"]  # b's spec never made


def test_build_envs_serves_a_host_command_like_voxys_setup(tmp_path):
    """voxy: ``voxy setup [backend ...]``, all by default, ``name<TAB>python`` lines."""
    specs = {"a": EnvSpec("a", ()), "b": EnvSpec("b", ())}
    lines = [f"{n}\t{py}" for n, py in build_envs(list(specs), specs,
             env_dir=lambda n: tmp_path / n, build=lambda s, p: "py")]
    assert lines == ["a\tpy", "b\tpy"]


def test_status_reads_each_marker_and_reports_a_failing_spec(tmp_path):
    a, b = EnvSpec("a", ("x",)), EnvSpec("b", ("y",))
    _built(tmp_path / "a", a, marker=".m.json")
    _built(tmp_path / "b", EnvSpec("b", ("old",)), marker=".m.json")  # stale

    def missing_lock():
        raise FileNotFoundError("c.lock")

    specs = {"a": a, "b": lambda: b, "c": missing_lock}
    code, lines = _run(["status"], specs, tmp_path, marker=".m.json")
    assert lines == [
        f"a: built ({tmp_path / 'a'})",
        f"b: not built ({tmp_path / 'b'})",
        f"c: no spec (FileNotFoundError: c.lock) ({tmp_path / 'c'})",
    ]
    assert [s for _, s, _ in env_status(specs, env_dir=lambda n: tmp_path / n, marker=".m.json")][:2] == [
        "built", "not built"]


def test_env_dir_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    lines = []
    cli({"a": EnvSpec("a", ())}, env_dir=lambda n: "~/envs/" + n, argv=["status"], out=lines.append)
    assert lines == [f"a: not built ({tmp_path / 'envs' / 'a'})"]


def test_an_error_inside_a_build_propagates(tmp_path):
    def broken(spec, path):
        raise ValueError("bad requirement")

    with pytest.raises(ValueError, match="bad requirement"):
        _run(["build", "a"], {"a": EnvSpec("a", ())}, tmp_path, build=broken)


def test_lock_exists_only_with_locks_and_calls_them(tmp_path, capsys):
    specs = {"a": EnvSpec("a", ()), "b": EnvSpec("b", ())}
    with pytest.raises(SystemExit):
        _run(["lock", "a"], specs, tmp_path)
    assert "invalid choice: 'lock'" in capsys.readouterr().err
    locks = {"a": lambda: tmp_path / "a.lock"}
    assert _run(["lock", "a"], specs, tmp_path, locks=locks) == (0, [str(tmp_path / "a.lock")])
    with pytest.raises(SystemExit):
        _run(["lock", "b"], specs, tmp_path, locks=locks)  # b has no lock
    with pytest.raises(ValueError, match="not in specs"):
        _run(["status"], specs, tmp_path, locks={"z": lambda: 1})


def test_compile_lock_runs_uv_with_the_spec_settings_and_host_pins_cleared(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_CONSTRAINT", "/host/constraints.txt")
    seen = {}

    def run(command, **kw):
        seen.update(command=command, **kw)

    path = compile_lock(["numpy<2", "torch"], tmp_path / "x.lock", python="3.11",
                        torch_backend="cpu", custom_command="python -m host lock x", run=run)
    assert path == tmp_path / "x.lock"
    command = seen["command"]
    assert command[:5] == ["uv", "pip", "compile", "-", "--universal"]
    assert command[command.index("--python-version") + 1] == "3.11"
    assert command[command.index("--output-file") + 1] == str(tmp_path / "x.lock")
    assert command[command.index("--torch-backend") + 1] == "cpu"
    assert command[-2:] == ["--custom-compile-command", "python -m host lock x"]
    assert seen["input"] == "numpy<2\ntorch\n" and seen["check"]
    assert all(seen["env"][v] == "" for v in HOST_UV_VARS)
    compile_lock(["x"], tmp_path / "y.lock", python="3.11", run=run)
    assert "--torch-backend" not in seen["command"]


def test_compile_lock_without_uv_says_how_to_get_it(tmp_path):
    def run(command, **kw):
        raise FileNotFoundError(command[0])

    with pytest.raises(isolated.RuntimeNotInstalled, match="Resolving a lock needs uv"):
        compile_lock(["x"], tmp_path / "l", python="3.11", run=run)


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_a_real_lock_round_trips_into_requirements(tmp_path):
    """Network: uv resolves against PyPI (from its cache when it can); skipped when it cannot."""
    lock = tmp_path / "x.lock"
    try:
        compile_lock(["six==1.16.0"], lock, python="3.11")
    except subprocess.CalledProcessError:
        pytest.skip("uv could not resolve (no network or cache)")
    assert lock_requirements(lock.read_text()) == ("six==1.16.0",)


def test_lock_requirements_keeps_editables_and_refuses_what_a_spec_cannot_carry():
    assert lock_requirements("-e ./pkg\nnumpy==2.0\n") == ("-e ./pkg", "numpy==2.0")
    with pytest.raises(ValueError, match="--index-url"):
        lock_requirements("--index-url https://example.org/simple\nnumpy==2.0\n")
    with pytest.raises(ValueError, match="cannot pass on"):
        lock_requirements("numpy==2.0 \\\n    --hash=sha256:abc\n")


def test_lock_requirements_drops_comments_keeps_markers():
    text = (
        "# This file was autogenerated by uv\n"
        "numpy==2.1.0\n"
        "    # via x\n"
        "torch==2.5.0 ; sys_platform == 'darwin'  # via y\n"
        "\n"
    )
    assert lock_requirements(text) == ("numpy==2.1.0", "torch==2.5.0 ; sys_platform == 'darwin'")
