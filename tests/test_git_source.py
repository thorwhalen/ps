"""Tests for ps.isolated's pinned git sources: GitSource, ensure_git_source,
check_git_source and Runtime(source=). Real git on local repositories (no
network); a fake ``run`` only for races and failures git cannot be made to produce.
"""

import shutil
import subprocess
import types
from pathlib import Path

import pytest

from ps.isolated import (
    GitSource,
    Runtime,
    RuntimeNotInstalled,
    SourceMismatch,
    check_git_source,
    ensure_git_source,
)
from ps import isolated

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
QUIET = dict(log=lambda message: None)


def _git(*args, cwd):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True,
    )


@pytest.fixture
def upstream(tmp_path):
    """A local repository with two commits; ``(url, first_sha, second_sha)``."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git("init", "--quiet", cwd=repo)
    (repo / "code.py").write_text("v = 1\n")
    _git("add", "code.py", cwd=repo)
    _git("commit", "--quiet", "-m", "one", cwd=repo)
    first = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    (repo / "code.py").write_text("v = 2\n")
    _git("commit", "--quiet", "-am", "two", cwd=repo)
    second = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return repo.as_uri(), first, second, repo


def test_a_source_needs_a_full_sha():
    with pytest.raises(ValueError, match="full commit SHA"):
        GitSource("https://example.org/r.git", "81aeb3a")
    with pytest.raises(ValueError, match="full commit SHA"):
        GitSource("https://example.org/r.git", "A" * 40)  # git prints lowercase


@needs_git
def test_fetches_the_pinned_commit_not_the_tip(upstream, tmp_path, monkeypatch):
    url, first, second, _ = upstream
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))  # must not reach git
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "elsewhere-index"))
    path = ensure_git_source(GitSource(url, first), tmp_path / "src" / "x", **QUIET)
    assert path == tmp_path / "src" / "x"
    assert (path / "code.py").read_text() == "v = 1\n"  # the tip says v = 2
    assert not path.with_name("x.partial").exists() and not path.with_name("x.lock").exists()
    monkeypatch.delenv("GIT_DIR")
    monkeypatch.delenv("GIT_INDEX_FILE")
    tip = ensure_git_source(GitSource(url, second), tmp_path / "y", **QUIET)
    log = subprocess.check_output(["git", "-C", str(tip), "log", "--format=%H"], text=True)
    assert log.split() == [second]  # only the commit: its parent was not fetched


@needs_git
def test_a_checkout_is_reused_and_checked(upstream, tmp_path):
    url, first, second, _ = upstream
    source = GitSource(url, first)
    path = ensure_git_source(source, tmp_path / "x", **QUIET)
    calls = []
    def counting(command, **kw):
        calls.append(next(a for a in command if a in ("init", "fetch", "rev-parse", "status")))
        return subprocess.run(command, **kw)
    (path / "venv-or-weights").mkdir()  # untracked files are fine
    assert ensure_git_source(source, path, run=counting, **QUIET) == path
    assert calls == ["rev-parse", "status"]  # no fetch


@needs_git
def test_edits_to_tracked_files_are_refused(upstream, tmp_path):
    url, first, _, _ = upstream
    source = GitSource(url, first)
    path = ensure_git_source(source, tmp_path / "x", **QUIET)
    (path / "code.py").write_text("v = 'patched'\n")
    with pytest.raises(SourceMismatch, match="edits to tracked files"):
        ensure_git_source(source, path, **QUIET)


@needs_git
def test_a_checkout_at_another_commit_is_refused_with_the_way_back(upstream, tmp_path):
    url, first, second, _ = upstream
    path = ensure_git_source(GitSource(url, second), tmp_path / "x", **QUIET)
    with pytest.raises(SourceMismatch) as error:
        check_git_source(GitSource(url, first), path)
    message = str(error.value)
    assert f"at {second[:12]}, not the pinned {first[:12]}" in message
    # the commands in the message do move it
    for line in message.split("\n\n", 1)[1].splitlines():
        subprocess.run(line, shell=True, check=True, capture_output=True)
    assert check_git_source(GitSource(url, first), path) == path
    assert (path / "code.py").read_text() == "v = 1\n"


@needs_git
def test_a_folder_that_is_not_a_checkout_is_refused(upstream, tmp_path):
    url, first, _, repo = upstream
    inside = repo / "sub"  # inside another repository: not a checkout of its own
    inside.mkdir()
    with pytest.raises(SourceMismatch, match="(?s)not a git checkout.*fetch"):
        check_git_source(GitSource(url, first), inside)
    (inside / "something").write_text("x")
    with pytest.raises(SourceMismatch, match="delete or move it"):
        check_git_source(GitSource(url, first), inside)
    assert isinstance(SourceMismatch("x"), RuntimeNotInstalled)  # hosts catch either


@needs_git
def test_runtime_with_a_source_checks_its_root(upstream, tmp_path):
    url, first, second, _ = upstream
    rt = Runtime(name="r", setup="cd {root} && make", default_dir=str(tmp_path),
                 python="bin/py", source=GitSource(url, first))
    hint = rt.hint()
    assert f"fetch --quiet --depth 1 {url} {first}" in hint
    assert hint.index("git init") < hint.index("make")
    with pytest.raises(RuntimeNotInstalled):
        rt.interpreter()
    root = ensure_git_source(GitSource(url, second), rt.root(), **QUIET)
    (root / "bin").mkdir()
    (root / "bin" / "py").write_text("")
    with pytest.raises(SourceMismatch, match="not the pinned"):
        rt.interpreter()
    with pytest.raises(SourceMismatch, match="re-run the install step:\n\ncd .* && make"):
        rt.interpreter()
    unpinned = Runtime(name="r", setup="x", default_dir=str(tmp_path), python="bin/py")
    assert unpinned.interpreter() == root / "bin" / "py"  # no source, no check


@needs_git
def test_a_runtime_checks_once_until_its_checkout_moves(upstream, tmp_path, monkeypatch):
    url, first, second, _ = upstream
    rt = Runtime(name="r", setup="", default_dir=str(tmp_path), python="bin/py",
                 source=GitSource(url, first))
    root = ensure_git_source(GitSource(url, first), rt.root(), **QUIET)
    (root / "bin").mkdir()
    (root / "bin" / "py").write_text("")
    calls = []
    real = isolated.check_git_source
    monkeypatch.setattr(isolated, "check_git_source", lambda *a, **k: calls.append(1) or real(*a, **k))
    rt.interpreter(), rt.interpreter()
    assert len(calls) == 1  # cached while HEAD and the index are unchanged
    for command in GitSource(url, second).fetch_commands(root)[1:]:
        subprocess.run(command, check=True, capture_output=True)
    with pytest.raises(SourceMismatch):
        rt.interpreter()  # moved: checked again


@needs_git
def test_a_runtime_named_by_its_variable_warns_instead(upstream, tmp_path, monkeypatch):
    url, first, second, _ = upstream
    root = ensure_git_source(GitSource(url, second), tmp_path / "mine", **QUIET)
    (root / "bin").mkdir()
    (root / "bin" / "py").write_text("")
    monkeypatch.setenv("PS_TEST_RT", str(root))
    rt = Runtime(name="r", setup="", default_dir=str(tmp_path / "rt"), env_var="PS_TEST_RT",
                 python="bin/py", source=GitSource(url, first))
    with pytest.warns(UserWarning, match="PS_TEST_RT names a runtime that is not the pinned one"):
        assert rt.interpreter() == root / "bin" / "py"


@needs_git
def test_an_interrupted_real_fetch_is_cleared_and_redone(upstream, tmp_path):
    url, first, second, _ = upstream
    partial = tmp_path / "x.partial"  # a real one: git's read-only objects included
    for command in GitSource(url, second).fetch_commands(partial)[:2]:
        subprocess.run(command, check=True, capture_output=True)
    path = ensure_git_source(GitSource(url, first), tmp_path / "x", **QUIET)
    assert (path / "code.py").read_text() == "v = 1\n" and not partial.exists()


class FakeGit:
    """``run`` for git: makes ``.git`` on init, answers rev-parse and status."""

    def __init__(self, head, *, edits=""):
        self.head, self.edits, self.commands = head, edits, []

    def __call__(self, command, check, capture_output, text, env):
        self.commands.append(command[1:])
        if command[1] == "init":
            Path(command[-1], ".git").mkdir(parents=True)
            return types.SimpleNamespace(stdout="")
        out = self.head + "\n" if "rev-parse" in command else self.edits if "status" in command else ""
        return types.SimpleNamespace(stdout=out)


def test_losing_the_rename_race_keeps_the_winners_copy(tmp_path):
    source = GitSource("https://example.org/r.git", "e" * 40)
    final = tmp_path / "x"

    class Racing(FakeGit):
        def __call__(self, command, **kw):
            if command[1:4] == ["-C", str(final.with_name("x.partial")), "checkout"]:
                (final / ".git").mkdir(parents=True)  # another process finishes first
                (final / "theirs").write_text("x")
            return super().__call__(command, **kw)

    assert ensure_git_source(source, final, run=Racing(source.commit), **QUIET) == final
    assert (final / "theirs").exists() and not final.with_name("x.partial").exists()


def test_an_interrupted_fetch_is_started_afresh(tmp_path):
    source = GitSource("https://example.org/r.git", "c" * 40)
    partial = tmp_path / "x.partial"
    partial.mkdir()
    (partial / "half-written").write_text("x")
    path = ensure_git_source(source, tmp_path / "x", run=FakeGit(source.commit), **QUIET)
    assert not partial.exists() and not (path / "half-written").exists()


def test_a_server_refusing_a_sha_gets_a_blobless_fetch(tmp_path):
    source = GitSource("https://example.org/r.git", "b" * 40)

    class NoShaWant(FakeGit):
        def __call__(self, command, **kw):
            if "--depth" in command:
                raise subprocess.CalledProcessError(128, command, stderr="not our ref")
            return super().__call__(command, **kw)

    run = NoShaWant(source.commit)
    assert ensure_git_source(source, tmp_path / "x", run=run, **QUIET) == tmp_path / "x"
    fetches = [c for c in run.commands if "fetch" in c]
    assert len(fetches) == 1 and "--filter=blob:none" in fetches[0]
    assert ["checkout", "--quiet", "--detach", source.commit] == [c for c in run.commands if "checkout" in c][0][2:]


def test_a_second_waiter_reuses_the_first_fetch(tmp_path):
    import threading

    source = GitSource("https://example.org/r.git", "a" * 40)
    final, run = tmp_path / "x", FakeGit("a" * 40)
    lock = final.with_name("x.lock")
    lock.mkdir()  # the first process is fetching

    def first_finishes():
        (final / ".git").mkdir(parents=True)
        shutil.rmtree(lock)

    threading.Timer(0.2, first_finishes).start()
    assert ensure_git_source(source, final, run=run, **QUIET) == final
    assert not any("fetch" in c or "init" in c for c in run.commands)  # only checked


def test_git_runs_without_location_variables_and_never_prompts(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/elsewhere")
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i key")
    seen = {}

    def run(command, **kw):
        seen.update(kw["env"])
        return FakeGit("a" * 40)(command, **kw)

    ensure_git_source(GitSource("https://example.org/r.git", "a" * 40), tmp_path / "x", run=run, **QUIET)
    assert "GIT_DIR" not in seen and seen["GIT_SSH_COMMAND"] == "ssh -i key"
    assert seen["GIT_TERMINAL_PROMPT"] == "0"


def test_a_git_failure_carries_its_message(tmp_path):
    def run(command, **kw):
        if "fetch" in command:
            raise subprocess.CalledProcessError(128, command, stderr="fatal: no such commit")
        return FakeGit("f" * 40)(command, **kw)

    with pytest.raises(RuntimeError, match="git -C .* fetch .* failed: fatal: no such commit"):
        ensure_git_source(GitSource("https://example.org/r.git", "f" * 40), tmp_path / "x", run=run, **QUIET)
    assert not (tmp_path / "x").exists()  # nothing half-made in place


def test_missing_git_says_how_to_get_it(tmp_path):
    def run(command, **kw):
        raise FileNotFoundError(command[0])

    with pytest.raises(RuntimeNotInstalled, match="git-scm"):
        ensure_git_source(GitSource("https://example.org/r.git", "f" * 40), tmp_path / "x", run=run, **QUIET)


def test_a_held_lock_times_out_with_a_hint(tmp_path):
    (tmp_path / "x.lock").mkdir()
    with pytest.raises(TimeoutError, match="delete it"):
        ensure_git_source(GitSource("https://example.org/r.git", "d" * 40), tmp_path / "x",
                          run=FakeGit("d" * 40), lock_timeout_s=0, **QUIET)
    assert (tmp_path / "x.lock").exists()  # someone else's lock is left alone


def test_the_lock_is_released_on_error(tmp_path):
    lock = tmp_path / "l.lock"
    with pytest.raises(KeyError):
        with isolated._dir_lock(lock, timeout_s=0):
            raise KeyError
    assert not lock.exists()


def test_a_sha256_commit_is_accepted():
    assert GitSource("https://example.org/r.git", "c" * 64).commit == "c" * 64
