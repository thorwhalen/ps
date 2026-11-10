"""Run Python code inside another Python environment, from this one.

Some libraries cannot share an environment with yours (or with each other): a model
pins its own torch, numpy or MLX, an evaluation toolkit wants ``numpy<2``. This module
lets a host process drive such code anyway, with the standard library only.

Two ways to get the environment, and one way to talk to it:

- :class:`Runtime` *locates* an environment someone installed by hand (a checkout with
  its own venv): ``$<env_var>``, else ``$<base_dir_var>/<name>``, else
  ``<default_dir>/<name>``. ps never installs it; a missing one raises
  :class:`RuntimeNotInstalled` carrying the setup commands. Given a ``source=``
  (a :class:`GitSource`: a repository at one full commit SHA), its install commands
  check that commit out and it refuses a root at another commit or with edits.
- :func:`ensure_git_source` fetches a :class:`GitSource` into a folder once (locked,
  atomic, verified on every reuse), for code that is a checkout rather than a package.
- :class:`EnvSpec` describes an environment ps *builds* with ``uv`` from a list of
  requirements: :func:`ensure_env` builds it once, and rebuilds it from scratch when
  the spec changes (a marker records the spec's digest).
- :func:`compile_lock` pins an environment's requirements into a lock file
  (:func:`lock_requirements` reads it back), and :func:`cli` is a ready
  ``build`` / ``status`` / ``lock`` command line over a host's named environments
  (:func:`build_envs` and :func:`env_status` are its parts, for a host's own).
- :class:`Worker` keeps one process of an environment alive, running an **adapter
  file**, and talks to it in JSON lines, so a model loads once instead of once per
  call. :func:`run_process` is the one-shot alternative: run a command, get
  ``(returncode, output)``.

An adapter file is a plain ``.py`` file that imports nothing from the host: it defines
``load(**init) -> state`` and operations ``op(state, **request) -> result``. The worker
loads this module file standalone in the environment (``ps`` need not be installed
there; this side loads on Python 3.8+), with the adapter's own folder on ``sys.path``
(so adapters can share a helper module), sends ``{"op": "load", **init}``, and
``call("op", **kw)`` returns ``op(state, **kw)`` (``None`` becomes ``{}``).

>>> import sys, tempfile, pathlib
>>> adapter = pathlib.Path(tempfile.mkdtemp()) / "echo_adapter.py"
>>> _ = adapter.write_text(
...     "def load(**kw): return kw\\n"
...     "def echo(state, **kw): print('noise'); return {**state, **kw}\\n"
... )
>>> with Worker(sys.executable, adapter, init={"a": 1}) as w:
...     w.call("echo", b=2)
{'a': 1, 'b': 2}

How a worker behaves:

- Protocol: one JSON request per line on its stdin, one reply per line on its stdout,
  ``{"ok": true, "result": ...}`` or ``{"ok": false, "error": "Type: message",
  "traceback": "..."}``. Everything else written to stdout (Python or C level, at
  import time too) goes to stderr, so it cannot corrupt the replies.
- stderr goes to a log file (a temporary one by default, deleted on close) whose tail
  is put into the error when the process dies; ``stderr=None`` leaves it on the
  host's terminal instead.
- The environment is the host's (``PATH`` included) minus ``PYTHONPATH``,
  ``PYTHONHOME`` and ``VIRTUAL_ENV``, plus ``env=``. The default working directory is
  an empty folder, so nothing shadows the environment's imports.
- A failed request leaves the worker serving; a failed ``load`` stops it; an
  interrupted or garbled exchange kills it (a stale reply must never be read by the
  next call), and the next call starts a fresh one. :meth:`Worker.close` waits for a
  call in flight; :meth:`Worker.kill` does not, and is what runs at interpreter exit.

What the copies this module merges map to
-----------------------------------------

arioso (``arioso._isolated``):

- ``Runtime(name=, env_var=, setup=, python=, cwd=)`` -> :class:`Runtime`, arioso
  supplying ``base_dir_var="ARIOSO_RUNTIMES_DIR"`` and its ``default_dir``.
- ``RuntimeNotInstalled``, ``WorkerError``, ``Runner``, ``run_process`` -> same names
  (``run_process`` now kills the process tree on timeout).
- ``Worker(python, module)`` + ``serve(handle)`` drivers -> :class:`Worker` with adapter
  files: each ``_driver`` module becomes an adapter file (``load`` + an operation).
- ``draw_seed``, ``read_wav``, ``is_apple_silicon``, ``song_from_wav``: arioso's own.

voxy (``voxy.envs``, ``voxy.worker``, ``voxy/workers/serve.py``):

- ``EnvSpec(name, requirements, python, torch_backend)`` -> :class:`EnvSpec` (same
  digest, so built environments are not rebuilt); voxy subclasses it to keep
  ``env_vars`` and its ``torch_backend="cpu"`` default.
- ``env_python(spec, rootdir=)`` -> ``venv_python(env_dir(spec, rootdir=rootdir))``;
  ``is_env_ready`` / ``ensure_env`` -> :func:`is_env_ready` / :func:`ensure_env` with
  ``path=env_dir(...)`` and ``marker=".voxy-env.json"``.
- ``worker_environ``'s stripping -> :func:`clean_environ` (every worker does it);
  threads, ``HF_HOME``, ``env_dir``, ``models_dir``, ``env_worker``, ``WorkerPool``
  stay voxy's.
- ``Worker(python, adapter, init=, env=, log_path=, cwd=)`` -> :class:`Worker`
  (``log_path=`` is ``stderr=``); ``.call(op, **kw)``, ``.running``, ``.close()`` same.
- ``GitSource``, ``ensure_git_source(source, name, rootdir=)`` (voxy#123) ->
  :class:`GitSource`, :func:`ensure_git_source` with ``path=git_source_dir(...)``;
  its ``_dir_lock`` -> a private ``_dir_lock``.
- ``python -m voxy setup [backend ...]`` -> :func:`cli` (``build``).
- ``WorkerError``, ``LOG_TAIL_CHARS`` -> same names; ``workers/serve.py`` ->
  :func:`serve_adapter` (its torch thread cap moves into the adapters' ``load``).

foley (``foley/sources/stable_audio_3/adapter.py``): ``_run_process`` ->
:func:`run_process` (stdout and stderr together, in order); ``_venv_python`` ->
:func:`venv_python`. Its ``setup_hint`` and ``runtime_ready`` stay foley's.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import weakref
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

#: How much of a worker's log (or a failed request's traceback) goes into an error.
LOG_TAIL_CHARS = 2000
#: Seconds :meth:`Worker.close` waits for a process to end before killing it.
DFLT_CLOSE_TIMEOUT = 10
#: The file, in a built environment, recording the digest of the spec it was built from.
DFLT_ENV_MARKER = ".ps-env.json"
#: Variables that point a Python at the host's packages, never passed to another env.
HOST_ONLY_VARS = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
#: uv settings holding the host's pins, cleared when building an environment.
HOST_UV_VARS = ("UV_CONSTRAINT", "UV_OVERRIDE", "UV_BUILD_CONSTRAINT")
#: The :class:`EnvSpec` fields that decide what is installed (hence the digest).
DIGEST_FIELDS = ("name", "requirements", "python", "torch_backend")

#: ``runner(cmd, *, cwd, env, timeout) -> (returncode, output_text)``.
Runner = Callable[..., Tuple[int, str]]
PathLike = Union[str, "os.PathLike[str]"]

_UV_INSTALL = (
    'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    if os.name == "nt"
    else "curl -LsSf https://astral.sh/uv/install.sh | sh"
)


class RuntimeNotInstalled(RuntimeError):
    """An environment (or the tool that builds it) is missing; the message says how to install it."""


class WorkerError(RuntimeError):
    """The worker failed a request (its traceback is in the message) or died (its log tail is)."""


# --------------------------------------------------------------------------------------
# Environments


def venv_python(path: PathLike) -> Path:
    """The interpreter of the virtual environment at ``path`` (POSIX or Windows layout).

    The layout that exists wins; when neither does, this platform's.

    >>> venv_python("/nowhere/.venv").relative_to("/nowhere/.venv").as_posix() in (
    ...     "bin/python", "Scripts/python.exe")
    True
    """
    path = Path(path)
    posix, windows = path / "bin" / "python", path / "Scripts" / "python.exe"
    candidates = (windows, posix) if os.name == "nt" else (posix, windows)
    return next((p for p in candidates if p.exists()), candidates[0])


def clean_environ(base: Optional[Mapping[str, str]] = None) -> dict:
    """``base`` (default ``os.environ``) without the variables that leak the host's packages.

    >>> clean_environ({"PYTHONPATH": "/host", "VIRTUAL_ENV": "/v", "HOME": "/h"})
    {'HOME': '/h'}
    """
    env = dict(os.environ if base is None else base)
    for name in HOST_ONLY_VARS:
        env.pop(name, None)
    return env


# --------------------------------------------------------------------------------------
# Pinned sources: code that is a git checkout at one commit


#: A full commit SHA (SHA-1 or SHA-256): anything shorter, a branch or a tag does not pin.
_FULL_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
#: Seconds :func:`ensure_git_source` waits for another process fetching the same source.
DFLT_SOURCE_LOCK_TIMEOUT_S = 900
_LOCK_POLL_S = 0.5
#: The variables that point git at another repository (``git rev-parse
#: --local-env-vars``): a caller's (a git hook's, say) must not reach our commands.
#: Credentials and transport settings (``GIT_SSH_COMMAND``, ``GIT_ASKPASS``) are kept.
GIT_LOCATION_VARS = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
)
#: What a server says when it will not send one commit by its SHA.
_SHA_REFUSED = ("not our ref", "unadvertised object", "does not allow request")
#: Never wait on a credential prompt (a terminal one, or Git Credential Manager's window).
_GIT_NO_PROMPT = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}


class SourceMismatch(RuntimeNotInstalled):
    """A checkout is not the pinned commit, has edits to its tracked files, or cannot be read."""


@dataclass(frozen=True)
class GitSource:
    """A repository at one commit: code a host runs that is a checkout, not a package.

    Args:
        url: The repository (anything ``git fetch`` takes).
        commit: The full commit SHA, lowercase as git prints it (a branch, a tag or a
            short SHA would not pin it).

    >>> GitSource("https://example.org/r.git", "main")
    Traceback (most recent call last):
      ...
    ValueError: GitSource needs a full commit SHA (40 or 64 lowercase hex characters), not 'main'
    """

    url: str
    commit: str

    def __post_init__(self):
        if not _FULL_SHA.fullmatch(self.commit):
            raise ValueError(
                "GitSource needs a full commit SHA (40 or 64 lowercase hex characters), "
                f"not {self.commit!r}"
            )

    def fetch_commands(self, path: PathLike) -> Tuple[Tuple[str, ...], ...]:
        """The git commands that check this commit out (only it, no history) into ``path``.

        >>> src = GitSource("https://example.org/r.git", "0" * 40)
        >>> [c[:4] for c in src.fetch_commands("/x")]
        [('git', 'init', '--quiet', '/x'), ('git', '-C', '/x', 'fetch'), ('git', '-C', '/x', 'checkout')]
        """
        path = str(path)
        return (
            ("git", "init", "--quiet", path),
            (
                "git",
                "-C",
                path,
                "fetch",
                "--quiet",
                "--depth",
                "1",
                self.url,
                self.commit,
            ),
            ("git", "-C", path, "checkout", "--quiet", "--detach", self.commit),
        )


def _git_environ() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in GIT_LOCATION_VARS}
    if "GIT_SSH_COMMAND" not in env and "GIT_SSH" not in env:
        env["GIT_SSH_COMMAND"] = (
            "ssh -o BatchMode=yes"  # no host-key or passphrase prompt
        )
    return {**env, **_GIT_NO_PROMPT}


def _git(run: Callable, *args: str) -> str:
    """``git args...`` (see :func:`_git_environ`); its stdout, stripped."""
    try:
        done = run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            env=_git_environ(),
        )
    except FileNotFoundError:
        raise RuntimeNotInstalled("This needs git: https://git-scm.com") from None
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}") from error
    return (done.stdout or "").strip()


def _commands_text(commands) -> str:
    return "\n".join(_shown(c, clear_uv=False) for c in commands)


def check_git_source(
    source: GitSource, path: PathLike, *, run: Callable = subprocess.run
) -> Path:
    """``path`` (``~`` expanded), checked to be ``source``'s commit with no edits to tracked files.

    Untracked files (a virtualenv, downloaded weights) are allowed. The check takes no
    lock in the checkout (``git --no-optional-locks``).

    Raises:
        SourceMismatch: ``path`` is not a checkout, is at another commit, has edits,
            or git cannot read it; the message says what to do.
    """
    path = Path(path).expanduser()
    if not (path / ".git").exists():
        if path.is_dir() and any(path.iterdir()):
            raise SourceMismatch(
                f"{path} is not a git checkout of {source.url}: delete or move it "
                "(it is not empty), then check the source out there"
            )
        raise SourceMismatch(
            f"{path} is not a git checkout of {source.url}; check it out with:\n\n"
            + _commands_text(source.fetch_commands(path))
        )
    try:
        head = _git(run, "-C", str(path), "rev-parse", "HEAD")
        if head != source.commit:
            raise SourceMismatch(
                f"{path} is at {head[:12] or 'no commit'}, not the pinned "
                f"{source.commit[:12]} of {source.url}. Move it there with:\n\n"
                + _commands_text(source.fetch_commands(path)[1:])
            )
        status = ("--no-optional-locks", "-C", str(path), "status", "--porcelain")
        edits = _git(run, *status, "--untracked-files=no")
    except SourceMismatch:
        raise
    except RuntimeError as error:  # git could not read it (ownership, corruption)
        raise SourceMismatch(f"{path}: {error}") from error
    if edits:
        raise SourceMismatch(
            f"{path} has edits to tracked files ({edits.splitlines()[0].strip()}, ...): "
            f"commit or discard them, or delete the folder to fetch "
            f"{source.commit[:12]} again"
        )
    return path


def _rmtree(path: Path) -> None:
    """Remove ``path`` if it exists, read-only files included (git's objects, on Windows)."""
    import stat

    def clear_and_retry(func, name, exc):
        error = exc[1] if isinstance(exc, tuple) else exc  # onerror passes exc_info
        if not isinstance(error, PermissionError) or func not in (os.unlink, os.rmdir):
            raise error
        os.chmod(name, os.stat(name).st_mode | stat.S_IWRITE)
        func(name)

    if path.exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=clear_and_retry)
        else:
            shutil.rmtree(path, onerror=clear_and_retry)


@contextmanager
def _dir_lock(path: Path, *, timeout_s: float):
    """Hold the lock ``path`` (a folder, made atomically) for the block, waiting up to ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{path} has been held for over {timeout_s:.0f} s; if no process "
                    "is fetching, delete it"
                ) from None
            time.sleep(_LOCK_POLL_S)
    try:
        yield
    finally:
        shutil.rmtree(path, ignore_errors=True)


def ensure_git_source(
    source: GitSource,
    path: PathLike,
    *,
    run: Callable = subprocess.run,
    log: Optional[Callable[[str], Any]] = None,
    lock_timeout_s: float = DFLT_SOURCE_LOCK_TIMEOUT_S,
) -> Path:
    """Check ``source`` out into ``path`` once (that commit only, no history); return ``path``, checked.

    The fetch holds a lock (``<path>.lock``: another process fetching the same source
    waits), goes into ``<path>.partial`` (an interrupted one starts afresh), and is
    renamed into place once verified. A server that refuses to send one commit by
    its SHA gets a blobless fetch of its branches and tags instead. A checkout
    already at ``path`` is verified (:func:`check_git_source`), never replaced.

    Args:
        source: The repository and commit.
        path: The checkout's folder (``~`` expanded; its parent is created).
        run: Runs each git command (seam; ``subprocess.run`` signature).
        log: Receives a notice before a fetch (default: stderr).
        lock_timeout_s: How long to wait for another process's fetch.
    """
    path = Path(path).expanduser()
    log = log or _log_to_stderr
    if not path.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        with _dir_lock(path.with_name(path.name + ".lock"), timeout_s=lock_timeout_s):
            if not path.is_dir():  # not fetched by the process we waited for
                _fetch_git_source(source, path, run=run, log=log)
    return check_git_source(source, path, run=run)


def _fetch_git_source(source: GitSource, path: Path, *, run, log) -> None:
    partial = path.with_name(path.name + ".partial")
    _rmtree(partial)
    log(f"Fetching {source.url} at {source.commit[:12]} into {path} (one-time)")
    init, fetch, checkout = source.fetch_commands(partial)
    _git(run, *init[1:])
    # no background gc or maintenance holding files in .partial (it is renamed next)
    _git(run, "-C", str(partial), "config", "gc.auto", "0")
    _git(run, "-C", str(partial), "config", "maintenance.auto", "false")
    try:
        _git(run, *fetch[1:])
    except RuntimeError as error:
        if not any(sign in str(error) for sign in _SHA_REFUSED):
            raise  # a wrong URL, no access: the first error is the one to see
        refs = ("+refs/heads/*:refs/remotes/origin/*", "+refs/tags/*:refs/tags/*")
        _git(
            run,
            "-C",
            str(partial),
            "fetch",
            "--quiet",
            "--filter=blob:none",
            source.url,
            *refs,
        )
        try:
            _git(run, "-C", str(partial), "cat-file", "-e", source.commit + "^{commit}")
        except RuntimeError:
            raise RuntimeError(
                f"{source.commit[:12]} is not on any branch or tag of {source.url} "
                "(and the server will not send it by its SHA)"
            ) from None
    _git(run, *checkout[1:])
    check_git_source(source, partial, run=run)
    try:
        partial.rename(path)
    except OSError:
        if not path.is_dir():
            raise
        _rmtree(partial)  # another process's copy won


@dataclass(frozen=True)
class Runtime:
    """Where a hand-installed environment (a checkout with its venv) lives, and how to install it.

    Pass the fields as keywords. The root is ``$<env_var>``, else
    ``$<base_dir_var>/<name>``, else ``<default_dir>/<name>``.

    Attributes:
        name: Folder name under the base or default directory.
        setup: Shell commands that install it; ``{root}`` and ``{python}`` are replaced.
        default_dir: The folder holding runtimes when no variable says otherwise.
        env_var: Variable naming this runtime's root (optional).
        base_dir_var: Variable naming the folder holding runtimes (optional).
        venv: The virtual environment, relative to the root (either OS layout).
        python: The interpreter relative to the root, overriding ``venv``.
        cwd: The processes' working directory, relative to the root.
        source: The :class:`GitSource` the root is a checkout of (optional). Then
            :meth:`hint` starts with the commands that check that commit out, and
            :meth:`interpreter` refuses a root at another commit or with edits
            (:class:`SourceMismatch`; a warning for a root named by ``env_var``), so
            two machines run the same code.

    >>> rt = Runtime(name="toy", setup="install {root}", default_dir="/rt")
    >>> rt.root().as_posix(), rt.installed()
    ('/rt/toy', False)
    """

    name: str
    setup: str
    default_dir: str
    env_var: Optional[str] = None
    base_dir_var: Optional[str] = None
    venv: str = ".venv"
    python: Optional[str] = None
    cwd: str = "."
    source: Optional[GitSource] = None

    def root(self) -> Path:
        """The runtime's root directory (it may not exist yet)."""
        if self.env_var and os.environ.get(self.env_var):
            return Path(os.environ[self.env_var]).expanduser()
        base = (
            self.base_dir_var and os.environ.get(self.base_dir_var)
        ) or self.default_dir
        return Path(base).expanduser() / self.name

    def _python(self) -> Path:
        root = self.root()
        return root / self.python if self.python else venv_python(root / self.venv)

    def installed(self) -> bool:
        """Whether the runtime's interpreter exists (a file check: the :attr:`source` is not read)."""
        return self._python().exists()

    def interpreter(self) -> Path:
        """The runtime's Python, or :class:`RuntimeNotInstalled` with the setup commands.

        With a :attr:`source`, the root is checked first (:func:`check_git_source`:
        two local git calls, some 20 ms, on every call; not cached, since a cache
        keyed on file times misses ``git reset`` and later edits). A
        root the runtime chose (under the base or default directory) that fails it
        raises :class:`SourceMismatch`, saying how to move it to the pinned commit;
        a root named by :attr:`env_var` is the user's own installation, so the
        mismatch is a warning and the root is used.
        """
        if not self.installed():
            raise RuntimeNotInstalled(self.hint())
        if self.source is not None:
            self._check_source()
        return self._python()

    def _check_source(self) -> None:
        root = self.root()
        try:
            check_git_source(self.source, root)
        except SourceMismatch as error:
            message = f"{error}\nThen re-run the install step:\n\n" + self.setup.format(
                root=root, python=self._python()
            )
            if self.env_var and os.environ.get(self.env_var):
                import warnings

                warnings.warn(
                    f"${self.env_var} names a runtime that is not the pinned one; "
                    f"using it anyway. {message}",
                    stacklevel=3,
                )
                return
            raise SourceMismatch(message) from None

    def workdir(self) -> Path:
        """The directory the runtime's processes run in."""
        return self.root() / self.cwd

    def hint(self) -> str:
        """How to install this runtime, or point to an existing one."""
        root, python = self.root(), self._python()
        fetch = ""
        if self.source is not None:
            commands = self.source.fetch_commands(root)
            fetch = "\n".join(_shown(c, clear_uv=False) for c in commands) + "\n"
        text = (
            f"The {self.name!r} runtime is not installed at {root} (no {python}). "
            "Install it with:\n\n" + fetch + self.setup.format(root=root, python=python)
        )
        if self.env_var:
            text += f"\n\nor set ${self.env_var} to an existing installation."
        return text


@dataclass(frozen=True)
class EnvSpec:
    """What an environment built by :func:`ensure_env` holds.

    Attributes:
        name: A name for messages (and usually the environment's folder).
        requirements: ``uv pip install`` arguments (packages, pins, extras).
        python: Python version (or interpreter) for ``uv venv``; default: this one's.
        torch_backend: ``uv``'s ``--torch-backend`` (``"cpu"`` keeps CUDA wheels out);
            None leaves the index alone.

    >>> EnvSpec("demo", ("numpy",)).digest() == EnvSpec("demo", ("numpy",)).digest()
    True
    """

    name: str
    requirements: Tuple[str, ...]
    python: str = "{}.{}".format(*sys.version_info[:2])
    torch_backend: Optional[str] = None

    def digest(self) -> str:
        """A short hash of what is installed (only the :data:`DIGEST_FIELDS`)."""
        content = {k: v for k, v in asdict(self).items() if k in DIGEST_FIELDS}
        blob = json.dumps(content, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def is_env_ready(
    spec: EnvSpec, path: PathLike, *, marker: str = DFLT_ENV_MARKER
) -> bool:
    """True if the environment at ``path`` (``~`` expanded) was built from this exact spec."""
    path = Path(path).expanduser()
    try:
        built = json.loads((path / marker).read_text()).get("digest")
    except (OSError, ValueError):
        return False
    return built == spec.digest() and venv_python(path).exists()


def _env_commands(spec: EnvSpec, path: Path, uv: str) -> list:
    commands = [[uv, "venv", "--python", spec.python, str(path)]]
    if spec.requirements:
        install = [uv, "pip", "install", "--python", str(venv_python(path))]
        if spec.torch_backend:
            install += ["--torch-backend", spec.torch_backend]
        commands.append(install + list(spec.requirements))
    return commands


def _shown(command, *, clear_uv: bool = True) -> str:
    """``command`` as a user would type it (with the host's uv pins cleared, by default)."""
    argv = [str(a) for a in command]
    if os.name == "nt":
        clear = "".join(f'set "{var}=" & ' for var in HOST_UV_VARS) if clear_uv else ""
        return clear + subprocess.list2cmdline(argv)
    clear = "".join(f"{var}= " for var in HOST_UV_VARS) if clear_uv else ""
    return clear + shlex.join(argv)


def _find_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeNotInstalled(
            f"Building an isolated environment needs uv: {_UV_INSTALL} "
            "(see https://docs.astral.sh/uv/)"
        )
    return uv


def _clear_env_dir(path: Path, marker: str) -> None:
    """Remove a stale environment at ``path`` that ps built (it holds ``marker``).

    Anything else is refused: an empty folder is reused, and a folder ps did not
    build (a hand-made venv, the running interpreter's own) is never deleted.
    """
    resolved = path.resolve()
    if resolved in {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}:
        raise RuntimeError(
            f"{path} is this interpreter's own environment: not replacing it"
        )
    if not path.exists() or not any(path.iterdir()):
        return
    if not (path / marker).exists():
        raise RuntimeError(
            f"{path} is not empty and was not built by ps (no {marker}): "
            "delete it yourself, or choose another folder"
        )
    shutil.rmtree(path)


def _write_marker(path: Path, marker: str, spec: EnvSpec, digest) -> None:
    content = {"digest": digest, "spec": asdict(spec)}
    (path / marker).write_text(json.dumps(content, indent=2))


def _log_to_stderr(message: str) -> None:
    print(message, file=sys.stderr)


def ensure_env(
    spec: EnvSpec,
    path: PathLike,
    *,
    run: Callable = subprocess.run,
    log: Callable[[str], Any] = _log_to_stderr,
    marker: str = DFLT_ENV_MARKER,
) -> Path:
    """Build ``spec``'s environment at ``path`` unless it was built from this spec; return its Python.

    A stale environment ps built (from another spec, or half-built) is removed and
    built afresh, so a changed Python version or a dropped requirement does take
    effect; a non-empty folder ps did not build raises instead (a provisional marker,
    written right after ``uv venv``, makes a half-built one count as ps's). The
    host's uv pins (:data:`HOST_UV_VARS`) are cleared for the build: they are for the
    host, and an isolated environment exists precisely to differ.

    Args:
        spec: What to install.
        path: The environment's folder (``~`` expanded).
        run: Runs each command (seam; ``subprocess.run`` signature).
        log: Receives a notice, with the commands, before a build.
        marker: The marker file name (a host may keep its own).
    """
    path = Path(path).expanduser()
    if is_env_ready(spec, path, marker=marker):
        return venv_python(path)
    commands = _env_commands(spec, path, _find_uv())
    shown = "\n".join(_shown(c) for c in commands)
    log(f"Building the {spec.name!r} environment in {path} (one-time):\n{shown}")
    _clear_env_dir(path, marker)
    env = {**os.environ, **{var: "" for var in HOST_UV_VARS}}
    venv, *install = commands
    run(venv, check=True, env=env)
    _write_marker(path, marker, spec, None)  # ps's, but not ready until the end
    for command in install:
        run(command, check=True, env=env)
    _write_marker(path, marker, spec, spec.digest())
    return venv_python(path)


# --------------------------------------------------------------------------------------
# Locks, and a command line over named environments

#: A named environment for :func:`cli`: a spec, or a function making one when needed
#: (a spec read from a lock file, say, which may not exist yet).
SpecSource = Union[EnvSpec, Callable[[], EnvSpec]]


def lock_requirements(text: str) -> Tuple[str, ...]:
    """The requirement lines of a lock (``uv pip compile`` output): comments dropped, markers kept.

    Editable and path requirements (``-e ./pkg``) are kept (resolved where the build
    runs; their content is not in the digest); any other option line
    (``--index-url`` ...) or a hashed, continued line raises, as :class:`EnvSpec`
    cannot carry it and dropping it would install something else.

    >>> lock_requirements("# header\\nnumpy==2.1.0\\n    # via x\\ntorch==2.5.0 ; sys_platform == 'darwin'\\n")
    ('numpy==2.1.0', "torch==2.5.0 ; sys_platform == 'darwin'")
    """
    lines = (line.split(" #")[0].strip() for line in text.splitlines())
    kept = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\") or (
            line.startswith("-") and not line.startswith(("-e ", "--editable "))
        ):  # options (an index), or hashes (continued lines): EnvSpec cannot carry them
            raise ValueError(f"A lock line ps cannot pass on: {line!r}")
        kept.append(line)
    return tuple(kept)


def compile_lock(
    requirements: Iterable[str],
    path: PathLike,
    *,
    python: str,
    torch_backend: Optional[str] = None,
    custom_command: Optional[str] = None,
    run: Callable = subprocess.run,
) -> Path:
    """Resolve ``requirements`` into a lock at ``path`` (``uv pip compile --universal``); return ``path``.

    Every package is pinned, with platform markers, so the lock serves every OS. Its
    lines (:func:`lock_requirements`) become an :class:`EnvSpec`'s requirements, so
    a changed lock changes the digest and rebuilds the environment. The host's uv
    pins (:data:`HOST_UV_VARS`) are cleared, as for :func:`ensure_env`.

    Args:
        requirements: The top-level requirements (what the environment is for).
        path: The lock file to write.
        python: The Python version to resolve for: the spec's ``python``.
        torch_backend: The spec's ``torch_backend`` (``"cpu"`` keeps CUDA wheels out
            of the Linux pins); None leaves the index alone.
        custom_command: The command the lock's header says regenerates it.
        run: Runs the command (seam; ``subprocess.run`` signature).
    """
    command = ["uv", "pip", "compile", "-", "--universal"]
    command += ["--python-version", python, "--output-file", str(path)]
    if torch_backend:
        command += ["--torch-backend", torch_backend]
    if custom_command:
        command += ["--custom-compile-command", custom_command]
    try:
        run(
            command,
            input="\n".join(requirements) + "\n",
            text=True,
            check=True,
            env={**os.environ, **{var: "" for var in HOST_UV_VARS}},
        )
    except FileNotFoundError:
        raise RuntimeNotInstalled(
            f"Resolving a lock needs uv: {_UV_INSTALL} (see https://docs.astral.sh/uv/)"
        ) from None
    return Path(path)


def _make_spec(source: SpecSource) -> EnvSpec:
    return source if isinstance(source, EnvSpec) else source()


def build_envs(
    names: Iterable[str],
    specs: Mapping[str, SpecSource],
    *,
    env_dir: Callable[[str], PathLike],
    build: Optional[Callable[[EnvSpec, Path], Any]] = None,
    marker: str = DFLT_ENV_MARKER,
) -> Iterable[Tuple[str, Any]]:
    """Build each named environment in turn; yield ``(name, python)`` as each is ready.

    Args:
        names: Which of ``specs`` to build (all of them: ``list(specs)``).
        specs: name -> :class:`EnvSpec`, or a function making it (called when needed).
        env_dir: name -> the environment's folder (``~`` expanded).
        build: ``(spec, folder) -> python``; default :func:`ensure_env` with ``marker``.
        marker: The marker file of the host's environments.

    Raises:
        KeyError: A name not in ``specs`` (checked before anything is built).
    """
    names = list(names)
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise KeyError(f"unknown environments {unknown}; choose from {list(specs)}")
    build = build or (lambda spec, path: ensure_env(spec, path, marker=marker))
    for name in names:
        yield name, build(_make_spec(specs[name]), Path(env_dir(name)).expanduser())


def env_status(
    specs: Mapping[str, SpecSource],
    *,
    env_dir: Callable[[str], PathLike],
    marker: str = DFLT_ENV_MARKER,
) -> Iterable[Tuple[str, str, Path]]:
    """``(name, state, folder)`` for each environment: ``"built"``, ``"not built"``, or
    ``"no spec (<error>)"`` when making its spec fails (a missing lock, say).

    ``env_dir`` must be where the environments are built: the marker is read there.
    """
    for name, source in specs.items():
        folder = Path(env_dir(name)).expanduser()
        try:
            ready = is_env_ready(_make_spec(source), folder, marker=marker)
        except Exception as error:  # noqa: BLE001 - one broken spec must not hide the rest
            yield name, f"no spec ({type(error).__name__}: {error})", folder
            continue
        yield name, "built" if ready else "not built", folder


def cli(
    specs: Mapping[str, SpecSource],
    *,
    env_dir: Callable[[str], PathLike],
    locks: Optional[Mapping[str, Callable[[], Any]]] = None,
    build: Optional[Callable[[EnvSpec, Path], Any]] = None,
    marker: str = DFLT_ENV_MARKER,
    prog: Optional[str] = None,
    argv: Optional[Iterable[str]] = None,
    out: Callable[[str], Any] = print,
) -> int:
    """A command line over named environments: ``build name ...``, ``status``, ``lock name ...``.

    - ``build`` builds the named environments, in order, and prints each one's
      interpreter, one per line.
    - ``status`` prints ``name: built (folder)``, ``name: not built (folder)``, or
      ``name: no spec (error) (folder)``.
    - ``lock`` (only when ``locks`` is given) calls each named lock function (a
      :func:`compile_lock` with its arguments bound, typically) and prints what it
      returns.

    A host exposes it in a few lines, e.g. in ``mypkg/isolated.py``::

        if __name__ == "__main__":
            sys.exit(isolated.cli(SPECS, env_dir=env_dir, prog="python -m mypkg.isolated"))

    A host with a command line of its own calls :func:`build_envs` and
    :func:`env_status` from it instead.

    Args:
        specs: name -> :class:`EnvSpec`, or a function making it (called when needed).
        env_dir: name -> the environment's folder; it must be where ``build`` builds
            it, as ``status`` reads the marker there.
        locks: name -> a function resolving that environment's lock (names must be
            among ``specs``).
        build: ``(spec, folder) -> python``; default :func:`ensure_env` with ``marker``.
        marker: The marker file of the host's environments.
        prog: The program name shown in help.
        argv: The arguments (default: ``sys.argv[1:]``).
        out: Receives each output line.

    >>> specs = {"a": EnvSpec("a", ()), "b": lambda: EnvSpec("b", ())}
    >>> cli(specs, env_dir=lambda name: "nowhere/" + name, argv=["status"],
    ...     out=lambda line: print(line.replace(os.sep, "/")))
    a: not built (nowhere/a)
    b: not built (nowhere/b)
    0
    """
    import argparse

    stray = sorted(set(locks or {}) - set(specs))
    if stray:
        raise ValueError(f"locks for environments not in specs: {stray}")
    parser = argparse.ArgumentParser(prog=prog, description="Isolated environments.")
    commands = parser.add_subparsers(dest="command", required=True)
    build_cmd = commands.add_parser("build", help="build environments")
    build_cmd.add_argument("names", nargs="+", choices=list(specs), metavar="name",
                           help="{" + ",".join(specs) + "}")
    commands.add_parser("status", help="where each environment is, and if it is built")
    if locks:
        lock_cmd = commands.add_parser("lock", help="re-resolve locks (needs uv)")
        lock_cmd.add_argument("names", nargs="+", choices=list(locks), metavar="name",
                              help="{" + ",".join(locks) + "}")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.command == "build":
        for _, python in build_envs(args.names, specs, env_dir=env_dir, build=build,
                                    marker=marker):
            out(str(python))
    elif args.command == "lock":
        for name in args.names:
            out(str(locks[name]()))
    else:
        for name, state, folder in env_status(specs, env_dir=env_dir, marker=marker):
            out(f"{name}: {state} ({folder})")
    return 0


# --------------------------------------------------------------------------------------
# The host side: one long-lived process


_LIVE_WORKERS: "weakref.WeakSet[Worker]" = weakref.WeakSet()
_NEUTRAL_DIR: list = []
_TEMP_LOG = object()  # the default ``stderr``: a temporary log file


@atexit.register
def _kill_all() -> None:
    """Kill every live worker, without waiting on a call in flight.

    This only helps calls made from daemon threads: Python joins non-daemon threads
    before running exit handlers, so a non-daemon thread stuck in a call keeps the
    host alive (stop it with :meth:`Worker.kill`).
    """
    for worker in list(_LIVE_WORKERS):
        try:
            worker.kill()
        except Exception:  # noqa: BLE001 - one stuck worker must not keep the rest
            pass


def _neutral_dir() -> str:
    """An empty folder, made once per host process: a default ``cwd`` that shadows nothing."""
    if not _NEUTRAL_DIR or not os.path.isdir(_NEUTRAL_DIR[0]):  # temp dirs get swept
        _NEUTRAL_DIR[:] = [tempfile.mkdtemp(prefix="ps-isolated-")]
    return _NEUTRAL_DIR[0]


#: Run in the worker: load this module standalone, serve the adapter. One line (no
#: newlines to survive Windows command-line quoting); the working directory is taken
#: off ``sys.path`` first, as for a script.
_BOOTSTRAP = "; ".join(
    [
        "import importlib.util, sys",
        "sys.path and sys.path[0] in ('', '.') and sys.path.pop(0)",
        "spec = importlib.util.spec_from_file_location('ps_isolated_serve', sys.argv[1])",
        "module = importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name] = module",
        "spec.loader.exec_module(module)",
        "module.serve_adapter(sys.argv[2])",
    ]
)


class Worker:
    """One long-lived process of another environment, running an adapter file.

    Started (and the adapter loaded) on the first :meth:`call`, stopped by
    :meth:`close` or :meth:`kill` (the latter at interpreter exit too). One request is
    in flight at a time. The process leads its own process group, so stopping it also
    kills what it spawned (a forked data loader, say); on POSIX it therefore does not
    get the terminal's Ctrl-C, and if the host is killed outright it ends when its
    stdin closes, once the operation it is running returns.

    Args:
        python: The environment's interpreter; a relative path is resolved against
            the host's working directory, a bare name is looked up on ``PATH``.
        adapter: The adapter file (``load`` plus operations); a relative path is
            resolved against the host's working directory.
        init: Keyword arguments of the adapter's ``load`` (sent on start).
        env: Environment variables added to the host's (minus :data:`HOST_ONLY_VARS`).
        cwd: The working directory (default: an empty folder).
        stderr: Where the process's stderr goes: a log file path, or ``None`` for the
            host's stderr (model output reaches the terminal, errors carry no log
            tail). Default: a temporary log file, created on start, deleted on close.
        spawn: The process factory (seam; :class:`subprocess.Popen` signature).
    """

    def __init__(
        self,
        python: PathLike,
        adapter: PathLike,
        *,
        init: Optional[Mapping] = None,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[PathLike] = None,
        stderr: Any = _TEMP_LOG,
        spawn: Callable[..., subprocess.Popen] = subprocess.Popen,
    ):
        # absolute: the process runs in another directory (an empty one by default)
        python = os.fspath(python)
        # a path (not a bare name looked up on PATH) is resolved against the host's
        # directory, like the adapter: Popen would resolve it against ``cwd``
        has_sep = os.sep in python or (os.altsep and os.altsep in python)
        self.python = os.path.abspath(python) if has_sep else python
        self.adapter = os.path.abspath(os.fspath(adapter))
        self.name = Path(self.adapter).stem
        self.init = dict(init or {})
        self.env = dict(env or {})
        self.cwd = str(cwd) if cwd else None
        self._temp_log = stderr is _TEMP_LOG
        self.log_path = None if self._temp_log or stderr is None else Path(stderr)
        self._spawn = spawn
        self._proc = None
        # re-entrant: a failed load or a dead process stops the worker inside call()
        self._lock = threading.RLock()
        _LIVE_WORKERS.add(self)

    @property
    def running(self) -> bool:
        """Whether the process is running."""
        return self._proc is not None and self._proc.poll() is None

    def call(self, op: str, /, **request) -> Any:
        """Run ``op(state, **request)`` in the worker (starting it on first use).

        Raises:
            WorkerError: The operation failed in the worker, or the worker died.
            TypeError, ValueError: ``request`` is not JSON-serialisable (the worker is
                left as it was).
        """
        payload = json.dumps({"op": op, **request})
        with self._lock:
            if not self.running:
                self._start()
            return self._send(payload)

    def _start(self) -> None:
        if self._temp_log and (self.log_path is None or not self.log_path.exists()):
            fd, path = tempfile.mkstemp(prefix=f"{self.name}-", suffix=".log")
            os.close(fd)
            self.log_path = Path(path)
        argv = [self.python, "-u", "-c", _BOOTSTRAP, __file__, self.adapter]
        from ps.util import NEW_PROCESS_GROUP

        options = dict(
            **NEW_PROCESS_GROUP,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=self.cwd or _neutral_dir(),
            env={**clean_environ(), **self.env},
            text=True,
            bufsize=1,
        )
        if self.log_path is None:
            self._proc = self._spawn(argv, stderr=None, **options)
        else:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "ab") as log:
                self._proc = self._spawn(argv, stderr=log, **options)
        self._send(json.dumps({"op": "load", **self.init}), loading=True)

    def _send(self, payload: str, *, loading: bool = False) -> Any:
        proc = self._proc
        try:
            reply = self._exchange(proc, payload)
        except BaseException:
            # An interrupted or garbled exchange would leave a reply in the pipe that the
            # next call would read as its own: stop the worker (it restarts lazily).
            self._stop(proc)
            raise
        if not reply.get("ok"):
            if loading:
                self._stop(proc)  # a model that failed to load must not serve calls
            trace = (reply.get("traceback") or "")[-LOG_TAIL_CHARS:]
            raise WorkerError(
                f"{self.name} worker failed: {reply.get('error')}\n{trace}"
            )
        return reply.get("result")

    def _exchange(self, proc, payload: str) -> dict:
        try:
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        except (OSError, ValueError, AttributeError):  # a broken or closed pipe
            line = ""
        if not line:
            raise self._died(proc)
        try:
            reply = json.loads(line)
        except ValueError:
            reply = None
        if not isinstance(reply, dict):
            raise WorkerError(
                f"{self.name} worker sent a line that is not a reply: {line[:200]!r}"
            )
        return reply

    def _died(self, proc) -> WorkerError:
        try:
            code = proc.wait(timeout=DFLT_CLOSE_TIMEOUT)
        except subprocess.TimeoutExpired:
            code = None
        where = (
            f"log: {self.log_path}):\n{self.log_tail()}"
            if self.log_path
            else "its stderr went to this process's)"
        )
        return WorkerError(
            f"The {self.name} worker exited (code {code}) before answering ({where}"
        )

    def log_tail(self, chars: int = LOG_TAIL_CHARS) -> str:
        """The last ``chars`` characters of the process's log ("" without a log file)."""
        if self.log_path is None:
            return ""
        try:
            with open(self.log_path, "rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(
                    max(0, log.tell() - 4 * chars)
                )  # utf-8: at most 4 bytes a char
                return log.read().decode("utf-8", errors="replace")[-chars:]
        except OSError:
            return ""

    def _stop(self, proc) -> None:
        """Kill ``proc`` if it is still this worker's process (keeps the log)."""
        if self._proc is proc:
            self._proc = None
        _terminate(proc)

    def _remove_temp_log(self) -> None:
        if self._temp_log and self.log_path is not None:
            try:
                self.log_path.unlink()
            except OSError:
                pass
            self.log_path = None

    def kill(self) -> None:
        """Kill the process (and what it spawned) now, without waiting for a call in flight.

        That call then fails. The reply pipe is left to its reader to close.
        """
        proc, self._proc = self._proc, None
        _terminate(proc, close_stdout=False)
        self._remove_temp_log()

    def close(self, timeout: float = DFLT_CLOSE_TIMEOUT) -> None:
        """Stop the process: close its stdin (ending the serve loop), kill it after ``timeout``.

        From another thread, this waits for a call in flight to finish first.
        """
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=timeout)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            _terminate(proc)
        self._remove_temp_log()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - nothing to report to at collection time
            pass


def _terminate(proc, *, close_stdout: bool = True) -> None:
    """Kill ``proc`` and its process group, collect it, and close our ends of its pipes.

    The group is killed even when ``proc`` already exited, since what it spawned may
    live on. ``close_stdout=False`` leaves the reply pipe alone: closing it while
    another thread reads it would block until every writer is gone.
    """
    if proc is None:
        return
    if isinstance(proc, subprocess.Popen):
        from ps.util import kill_process_tree

        kill_process_tree(proc)
    elif proc.poll() is None:  # a spawn= stand-in: no group to kill
        proc.kill()
    proc.wait()
    streams = [getattr(proc, "stdin", None)]
    if close_stdout:
        streams.append(getattr(proc, "stdout", None))
    for stream in streams:
        try:
            stream.close()
        except Exception:  # noqa: BLE001 - already closed, or a fake without close
            pass


# --------------------------------------------------------------------------------------
# The environment side


def _protect_stdout():
    """Keep the real stdout for replies; send everything else written to fd 1 to stderr."""
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # C-level prints to fd 1 now land in stderr too
    sys.stdout = sys.stderr
    return replies


def _adapter_handler(adapter) -> Callable[[dict], Any]:
    state = None

    def handle(request: dict) -> Any:
        nonlocal state
        op = request.pop("op")
        if op == "load":
            state = adapter.load(**request)
            return {}
        func = getattr(adapter, op, None)
        if op.startswith("_") or not callable(func):
            raise ValueError(f"Unknown operation {op!r}")
        result = func(state, **request)
        return {} if result is None else result

    return handle


def serve_adapter(path: PathLike) -> None:
    """Load the adapter file at ``path`` and answer JSON requests on stdin (what a :class:`Worker` runs).

    stdout is protected before the adapter is loaded, so what it prints on import goes
    to stderr too; the adapter's folder is put first on ``sys.path``.
    """
    replies = _protect_stdout()
    path = os.path.abspath(os.fspath(path))
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("ps_isolated_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter  # what dataclasses and pickle look up
    spec.loader.exec_module(adapter)
    handle = _adapter_handler(adapter)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            reply = {"ok": True, "result": handle(json.loads(line))}
        except Exception as error:  # report every failure to the client, keep serving
            reply = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        replies.write(json.dumps(reply) + "\n")
        replies.flush()


# --------------------------------------------------------------------------------------
# One-shot


def run_process(
    cmd: Iterable,
    *,
    cwd: Optional[PathLike] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
) -> Tuple[int, str]:
    """Run ``cmd`` once; return ``(returncode, output)``, stdout and stderr interleaved.

    ``env`` is added to the host's environment (minus :data:`HOST_ONLY_VARS`); stdin is
    closed, so a program prompting for input fails instead of waiting. Past ``timeout``
    seconds the process and everything it spawned are killed, and
    :class:`subprocess.TimeoutExpired` is raised. The process leads its own process
    group for that, so on POSIX it does not get the terminal's Ctrl-C or SIGHUP, and
    it outlives a host that is killed outright.

    >>> import sys
    >>> run_process([sys.executable, "-c", "print('hi')"])
    (0, 'hi\\n')
    """
    from ps.util import NEW_PROCESS_GROUP, _kill_and_reap, kill_process_tree

    argv = [str(c) for c in cmd]
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env={**clean_environ(), **(env or {})},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **NEW_PROCESS_GROUP,
    )
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_and_reap(proc, kill=kill_process_tree)
        raise subprocess.TimeoutExpired(argv, timeout) from None
    except BaseException:
        _kill_and_reap(proc, kill=kill_process_tree)
        raise
    return proc.returncode, output
