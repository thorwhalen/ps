"""Run Python code inside another Python environment, from this one.

Some libraries cannot share an environment with yours (or with each other): a model
pins its own torch, numpy or MLX, an evaluation toolkit wants ``numpy<2``. This module
lets a host process drive such code anyway, with the standard library only:

- :class:`Runtime` says where a hand-installed environment lives
  (``$<env_var>``, else ``$<base_dir_var>/<name>``, else ``<default_dir>/<name>``) and how
  to install it; a missing one raises :class:`RuntimeNotInstalled` carrying the commands.
- :class:`EnvSpec` and :func:`ensure_env` build an environment with ``uv`` from a list
  of requirements, once: a marker holding the spec's digest makes a changed spec rebuild.
- :class:`Worker` keeps one process of that environment alive and talks to it in JSON
  lines, so a model loads once instead of once per call. Its stderr goes to a log file
  whose tail is put into the error when the process dies. :class:`WorkerPool` shares
  workers between threads, one per key.
- :func:`serve` (and :func:`serve_adapter`) is the loop on the other side.
- :func:`run_process` is the one-shot alternative: run a command, get
  ``(returncode, output)``.

Two driver styles
-----------------

A **module driver** is a module the environment can import, run as
``python -m <module>``; it calls ``serve(handle)``, ``handle(request) -> result``.
Pass ``expose=("mypackage", "ps")`` so the worker can import those packages (and only
those: a directory linking just them is put on its ``PYTHONPATH``).

An **adapter file** is a plain ``.py`` file that imports nothing from the host; it
defines ``load(**init) -> state`` and operations ``op(state, **request) -> dict``. The
worker loads this very module file standalone in the environment (``ps`` need not be
installed there), sends ``{"op": "load", **init}`` on start, and ``call("op", **kw)``
runs ``op(state, **kw)``. This side of the module loads on Python 3.8+.

>>> import sys, tempfile, pathlib
>>> adapter = pathlib.Path(tempfile.mkdtemp()) / "echo_adapter.py"
>>> _ = adapter.write_text(
...     "def load(**kw): return kw\\n"
...     "def echo(state, **kw): print('noise'); return {**state, **kw}\\n"
... )
>>> with Worker(sys.executable, adapter, init={"a": 1}) as w:
...     w.call("echo", b=2)
{'a': 1, 'b': 2}

Protocol: one JSON request per line on the worker's stdin, one reply per line on its
stdout, ``{"ok": true, "result": ...}`` or ``{"ok": false, "error": "Type: message",
"traceback": "..."}``. Anything the libraries print (Python or C level, to file
descriptor 1) is sent to stderr, so it cannot corrupt the replies. A failed request
leaves the worker serving; an interrupted or garbled exchange kills it (a stale reply
must never be read by the next call), and the next call starts a fresh one.

What the copies this module merges map to
-----------------------------------------

arioso (``arioso._isolated``):

- ``Runtime(name=, env_var=, setup=, python=, cwd=)`` -> :class:`Runtime`, with
  ``base_dir_var="ARIOSO_RUNTIMES_DIR"`` and ``default_dir=DFLT_RUNTIMES_DIR`` supplied
  by arioso; ``.root() .installed() .interpreter() .workdir() .hint()`` unchanged.
- ``RuntimeNotInstalled``, ``WorkerError``, ``Runner``, ``run_process`` -> same names.
- ``Worker(python, module, env=, cwd=, spawn=)``, ``.call(dict)``, ``.alive``,
  ``.close()`` -> :class:`Worker`, same signature; arioso adds
  ``expose=("arioso", "ps")`` in place of ``_arioso_only_dir``. Differences: stderr goes
  to ``log_path`` (its tail is in errors) instead of the host's stderr, and an omitted
  ``cwd`` is a neutral empty folder instead of the host's.
- ``serve(handle, stdin=)`` -> :func:`serve`, unchanged: the existing ``_driver``
  modules keep working.
- ``draw_seed``, ``read_wav``, ``is_apple_silicon``, ``song_from_wav``: arioso's own.

voxy (``voxy.envs``, ``voxy.worker``, ``voxy/workers/serve.py``):

- ``EnvSpec(name, requirements, python, torch_backend)`` -> :class:`EnvSpec` (same
  digest, so built environments are not rebuilt); voxy subclasses it to keep
  ``env_vars`` and its ``torch_backend="cpu"`` default.
- ``env_python(spec, rootdir=)`` -> ``venv_python(env_dir(spec, rootdir=rootdir))``.
- ``is_env_ready`` / ``ensure_env(spec, rootdir=, run=, log=)`` ->
  :func:`is_env_ready` / :func:`ensure_env` with ``path=env_dir(spec, rootdir=rootdir)``
  and ``marker=".voxy-env.json"``; ``_uv`` is internal here.
- ``worker_environ``: its ``PYTHONPATH``/``VIRTUAL_ENV``/``PYTHONHOME`` stripping is
  :func:`clean_environ` (every worker does it); threads and ``HF_HOME`` stay voxy's, as
  do ``env_dir``, ``models_dir``, ``num_threads``, ``remove_env``, ``env_worker``.
- ``Worker(python, adapter, init=, env=, log_path=, cwd=)``, ``.call(op, **kw)``,
  ``.running`` -> :class:`Worker`, same signature (a ``.py`` path is an adapter file).
- ``WorkerError``, ``LOG_TAIL_CHARS`` -> same names.
- ``WorkerPool(spec, adapter, make=)`` -> :class:`WorkerPool` (``make`` and the key are
  voxy's: a few-line subclass).
- ``workers/serve.py`` -> :func:`serve_adapter` (run by :class:`Worker`); its torch
  thread cap is voxy's to keep (in the adapters' ``load``).

foley (``foley/sources/stable_audio_3/adapter.py``):

- ``_run_process`` -> :func:`run_process` (stdout and stderr together, in order).
- ``_venv_python`` -> :func:`venv_python`; ``setup_hint``, ``runtime_ready`` ->
  ``Runtime.hint``, ``Runtime.installed``.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

#: How much of a worker's log (or a failed request's traceback) goes into an error.
LOG_TAIL_CHARS = 2000
#: Seconds :meth:`Worker.close` waits for a process to end before killing it.
DFLT_CLOSE_TIMEOUT = 10
#: The file, in a built environment, recording the digest of the spec it was built from.
DFLT_ENV_MARKER = ".ps-env.json"
#: Variables that point a Python at the host's packages, never passed to another env.
HOST_ONLY_VARS = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
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


def _shell_join(argv) -> str:
    """``argv`` as one command line for this platform's shell."""
    argv = [str(a) for a in argv]
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


# --------------------------------------------------------------------------------------
# Where environments live


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
        """Whether the runtime's interpreter exists."""
        return self._python().exists()

    def interpreter(self) -> Path:
        """The runtime's Python, or :class:`RuntimeNotInstalled` with the setup commands."""
        if not self.installed():
            raise RuntimeNotInstalled(self.hint())
        return self._python()

    def workdir(self) -> Path:
        """The directory the runtime's processes run in."""
        return self.root() / self.cwd

    def hint(self) -> str:
        """How to install this runtime, or point to an existing one."""
        root, python = self.root(), self._python()
        text = (
            f"The {self.name!r} runtime is not installed at {root} (no {python}). "
            "Install it with:\n\n" + self.setup.format(root=root, python=python)
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


def env_commands(spec: EnvSpec, path: PathLike, *, uv: str = "uv") -> list:
    """The commands that build ``spec``'s environment at ``path``: ``uv venv``, then the install."""
    commands = [[uv, "venv", "--python", spec.python, str(path)]]
    if spec.requirements:
        install = [uv, "pip", "install", "--python", str(venv_python(path))]
        if spec.torch_backend:
            install += ["--torch-backend", spec.torch_backend]
        commands.append(install + list(spec.requirements))
    return commands


def is_env_ready(
    spec: EnvSpec, path: PathLike, *, marker: str = DFLT_ENV_MARKER
) -> bool:
    """True if the environment at ``path`` (``~`` expanded) exists and was built from this exact spec."""
    path = Path(path).expanduser()
    try:
        built = json.loads((path / marker).read_text()).get("digest")
    except (OSError, ValueError):
        return False
    return built == spec.digest() and venv_python(path).exists()


def _find_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeNotInstalled(
            f"Building an isolated environment needs uv: {_UV_INSTALL} "
            "(see https://docs.astral.sh/uv/)"
        )
    return uv


def _log_to_stderr(message: str) -> None:
    print(message, file=sys.stderr)


def ensure_env(
    spec: EnvSpec,
    path: PathLike,
    *,
    build: bool = True,
    uv: Optional[str] = None,
    run: Callable = subprocess.run,
    log: Callable[[str], Any] = _log_to_stderr,
    marker: str = DFLT_ENV_MARKER,
) -> Path:
    """Build ``spec``'s environment at ``path`` if missing or stale; return its interpreter.

    The host's ``UV_CONSTRAINT`` is cleared for the build: its pins are for the host,
    and an isolated environment exists precisely to differ.

    Args:
        spec: What to install.
        path: The environment's folder.
        build: If False, a missing environment raises :class:`RuntimeNotInstalled`
            carrying the commands instead of running them.
        uv: The ``uv`` executable (default: found on ``PATH``).
        run: Runs each command (seam; ``subprocess.run`` signature).
        log: Receives a one-line notice before a build.
        marker: The marker file name (a host may keep its own).
    """
    path = Path(path).expanduser()
    python = venv_python(path)
    if is_env_ready(spec, path, marker=marker):
        return python
    if not build:
        commands = "\n".join(_shell_join(c) for c in env_commands(spec, path))
        raise RuntimeNotInstalled(
            f"The {spec.name!r} environment is not built at {path}. Build it with:"
            f"\n\n{commands}"
        )
    venv, *install = env_commands(spec, path, uv=uv or _find_uv())
    env = {**os.environ, "UV_CONSTRAINT": ""}
    log(f"Building the {spec.name!r} environment in {path} (one-time)")
    if not python.exists():
        run(venv, check=True, env=env)
    for command in install:
        run(command, check=True, env=env)
    path.mkdir(parents=True, exist_ok=True)
    (path / marker).write_text(
        json.dumps({"digest": spec.digest(), "spec": asdict(spec)}, indent=2)
    )
    return venv_python(path)


# --------------------------------------------------------------------------------------
# The host side: one long-lived process


_LIVE_WORKERS: "weakref.WeakSet[Worker]" = weakref.WeakSet()
_EXPOSED_DIRS: dict = {}
_NEUTRAL_DIR: list = []


@atexit.register
def _close_all() -> None:
    for worker in list(_LIVE_WORKERS):
        try:
            worker.close()
        except Exception:  # noqa: BLE001 - one stuck worker must not keep the rest
            pass


def _neutral_dir() -> str:
    """An empty folder, made once per host process: a default ``cwd`` that shadows nothing."""
    if not _NEUTRAL_DIR or not os.path.isdir(_NEUTRAL_DIR[0]):  # temp dirs get swept
        _NEUTRAL_DIR[:] = [tempfile.mkdtemp(prefix="ps-isolated-")]
    return _NEUTRAL_DIR[0]


def _package_path(package: Union[str, ModuleType, PathLike]) -> Path:
    """The directory (or file) of a top-level package given by name, module or path."""
    if isinstance(package, ModuleType):
        file = Path(package.__file__)
        return file.parent if hasattr(package, "__path__") else file
    if isinstance(package, str):
        spec = importlib.util.find_spec(package)
        if spec is None:
            raise ModuleNotFoundError(f"Cannot expose {package!r}: no such module")
        if spec.submodule_search_locations:
            return Path(list(spec.submodule_search_locations)[0])
        return Path(spec.origin)
    return Path(package)


def _exposed_dir(packages: Iterable) -> str:
    """A directory holding only links to ``packages``, for a worker's ``PYTHONPATH``.

    Putting a package's parent directory on the path would, for a regular install,
    expose the host's whole site-packages to an environment built to pin its own.
    Without symlink rights (Windows without developer mode) the packages are copied.
    """
    sources = tuple(_package_path(p).resolve() for p in packages)
    base = _EXPOSED_DIRS.get(sources)
    if base is None or not all(Path(base, s.name).exists() for s in sources):
        base = tempfile.mkdtemp(prefix="ps-isolated-path-")
        for source in sources:
            link = Path(base, source.name)
            try:
                link.symlink_to(source, target_is_directory=source.is_dir())
            except OSError:
                if source.is_dir():
                    ignore = shutil.ignore_patterns("__pycache__")
                    shutil.copytree(source, link, ignore=ignore)
                else:
                    shutil.copy2(source, link)
        _EXPOSED_DIRS[sources] = base
    return base


def _is_adapter_file(target) -> bool:
    return isinstance(target, os.PathLike) or str(target).endswith(".py")


#: Run in the worker for an adapter file: load this module standalone, serve the adapter.
#: One line (no newlines to survive Windows command-line quoting); the working
#: directory is taken off ``sys.path`` first, as for a script.
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
    """One long-lived process of another environment, answering JSON requests.

    Started on the first :meth:`call`, stopped by :meth:`close` (also at interpreter
    exit). One request is in flight at a time; :meth:`close` from another thread waits
    for it.

    Args:
        python: The environment's interpreter.
        target: A module name, run as ``python -m target`` (it calls :func:`serve`), or
            the path of an adapter file (a ``Path``, or a string ending in ``.py``),
            served by :func:`serve_adapter`.
        init: Sent as ``{"op": "load", **init}`` when the process starts; a failed load
            stops it. Adapter files are always loaded (default ``{}``); a module driver
            only when ``init`` is given.
        env: Environment variables added to the host's. The host's ``PYTHONPATH``,
            ``PYTHONHOME`` and ``VIRTUAL_ENV`` are never passed on; a ``PYTHONPATH``
            given here comes after the ``expose`` directory.
        cwd: The working directory (default: an empty folder, so nothing in the
            caller's folder shadows the environment's imports).
        log_path: Where the process's stderr goes (default: a temporary file).
        expose: Packages (names, modules or paths) the process may import from the
            host, e.g. ``("mypackage", "ps")`` for a module driver.
        spawn: The process factory (seam; :class:`subprocess.Popen` signature).
    """

    def __init__(
        self,
        python: PathLike,
        target: PathLike,
        *,
        init: Optional[Mapping] = None,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[PathLike] = None,
        log_path: Optional[PathLike] = None,
        expose: Iterable = (),
        spawn: Callable[..., subprocess.Popen] = subprocess.Popen,
    ):
        self.python, self.target = str(python), os.fspath(target)
        self.is_adapter = _is_adapter_file(target)
        self.name = Path(self.target).stem if self.is_adapter else self.target
        if init is None and self.is_adapter:
            init = {}
        self.init = None if init is None else dict(init)
        self.env = dict(env or {})
        self.cwd = str(cwd) if cwd else None
        if log_path is None:
            fd, log_path = tempfile.mkstemp(prefix=f"{self.name}-", suffix=".log")
            os.close(fd)
        self.log_path = Path(log_path)
        self.expose = tuple(expose)
        self._spawn = spawn
        self._proc = None
        # re-entrant: a failed load or a dead process closes the worker inside call()
        self._lock = threading.RLock()
        _LIVE_WORKERS.add(self)

    def _argv(self) -> list:
        if self.is_adapter:
            return [self.python, "-u", "-c", _BOOTSTRAP, __file__, self.target]
        return [self.python, "-u", "-m", self.target]

    def _environ(self) -> dict:
        env = {**clean_environ(), **self.env}
        path = [_exposed_dir(self.expose)] if self.expose else []
        path += [self.env["PYTHONPATH"]] if self.env.get("PYTHONPATH") else []
        env.pop("PYTHONPATH", None)
        if path:
            env["PYTHONPATH"] = os.pathsep.join(path)
        return env

    def _start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as log:
            self._proc = self._spawn(
                self._argv(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                cwd=self.cwd or _neutral_dir(),
                env=self._environ(),
                text=True,
                bufsize=1,
            )
        if self.init is not None:
            self._send(json.dumps({"op": "load", **self.init}), loading=True)

    @property
    def alive(self) -> bool:
        """Whether the process is running."""
        return self._proc is not None and self._proc.poll() is None

    running = alive

    def call(self, request: Union[str, Mapping, None] = None, /, **fields) -> Any:
        """Send a request; return the driver's result (starting the process if needed).

        ``call({"a": 1})`` sends that dict; ``call("op", **kw)`` sends
        ``{"op": "op", **kw}`` (the adapter-file form).

        Raises:
            WorkerError: The request failed in the worker, or the worker died.
        """
        if isinstance(request, str):
            request = {"op": request, **fields}
        else:
            request = {**(request or {}), **fields}
        payload = json.dumps(
            request
        )  # a request that cannot be sent is the caller's error
        with self._lock:
            if not self.alive:
                self._start()
            return self._send(payload)

    def _send(self, payload: str, *, loading: bool = False) -> Any:
        try:
            reply = self._exchange(payload)
        except BaseException:
            # An interrupted or garbled exchange would leave a reply in the pipe that the
            # next call would read as its own: stop the worker (it restarts lazily).
            self._kill()
            raise
        if not reply.get("ok"):
            trace = (reply.get("traceback") or "")[-LOG_TAIL_CHARS:]
            if loading:
                self._kill()  # a model that failed to load must not serve calls
            raise WorkerError(
                f"{self.name} worker failed: {reply.get('error')}\n{trace}"
            )
        return reply.get("result")

    def _exchange(self, payload: str) -> dict:
        proc = self._proc
        try:
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        except (OSError, ValueError):  # a broken pipe, or a stream already closed
            line = ""
        if not line:
            raise self._died()
        try:
            reply = json.loads(line)
        except ValueError:
            reply = None
        if not isinstance(reply, dict):
            raise WorkerError(
                f"{self.name} worker sent a line that is not a reply: {line[:200]!r}"
            )
        return reply

    def _died(self) -> WorkerError:
        try:
            code = self._proc.wait(timeout=DFLT_CLOSE_TIMEOUT)
        except subprocess.TimeoutExpired:
            code = None
        return WorkerError(
            f"The {self.name} worker exited (code {code}) before answering "
            f"(log: {self.log_path}):\n{self.log_tail()}"
        )

    def log_tail(self) -> str:
        """The end of the process's log (stderr)."""
        try:
            return self.log_path.read_text(errors="replace")[-LOG_TAIL_CHARS:]
        except OSError:
            return ""

    def _kill(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            _close_pipes(proc)

    def close(self, timeout: float = DFLT_CLOSE_TIMEOUT) -> None:
        """Stop the process (closing its stdin ends :func:`serve`'s loop); kill it after ``timeout``.

        From another thread, this waits for a call in flight to finish first.
        """
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=timeout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait()
        _close_pipes(proc)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - nothing to report to at collection time
            pass


def _close_pipes(proc) -> None:
    for stream in (getattr(proc, "stdin", None), getattr(proc, "stdout", None)):
        try:
            stream.close()
        except Exception:  # noqa: BLE001 - already closed, or a fake without close
            pass


class WorkerPool:
    """Workers shared across threads, one per key, each made once.

    ``get(*args, **kwargs)`` returns ``make(*args, **kwargs)``, made on the first ask
    for those arguments (they are the key, so they must be hashable). One lock covers
    the making, so two threads never build the same environment at once.

    >>> pool = WorkerPool(lambda size: object())
    >>> pool.get("small") is pool.get("small") is not pool.get("large")
    True
    """

    def __init__(self, make: Callable[..., Worker]):
        self.make = make
        self._workers: dict = {}
        self._lock = threading.Lock()

    def get(self, *args, **kwargs) -> Worker:
        """The worker for these arguments (made on first use)."""
        key = (args, tuple(sorted(kwargs.items())))
        worker = self._workers.get(key)
        if worker is not None:
            return worker
        with self._lock:
            if key not in self._workers:
                self._workers[key] = self.make(*args, **kwargs)
            return self._workers[key]

    def close(self) -> None:
        """Stop every worker; the next ``get`` starts afresh."""
        with self._lock:
            workers, self._workers = list(self._workers.values()), {}
        for worker in workers:
            try:
                worker.close()
            except Exception:  # noqa: BLE001 - one stuck worker must not keep the rest
                pass


# --------------------------------------------------------------------------------------
# The environment side


def _protect_stdout():
    """Keep the real stdout for replies; send everything else written to fd 1 to stderr."""
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # C-level prints to fd 1 now land in stderr too
    sys.stdout = sys.stderr
    return replies


def _serve_loop(handle: Callable[[Any], Any], lines, replies) -> None:
    for line in lines:
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


def serve(handle: Callable[[Any], Any], *, stdin=None) -> None:
    """The driver side of :class:`Worker`: answer each JSON line on stdin with ``handle``.

    ``handle(request)`` returns something JSON-serialisable; an exception is sent back
    (type, message, traceback) and the loop carries on. Library output to stdout,
    Python or C level, goes to stderr so it cannot corrupt the replies.
    """
    replies = _protect_stdout()
    _serve_loop(handle, stdin or sys.stdin, replies)


def adapter_handler(adapter: ModuleType) -> Callable[[dict], dict]:
    """A :func:`serve` handler for an adapter module: ``load(**init)``, then operations.

    ``{"op": "load", **init}`` sets the state to ``adapter.load(**init)``; any other
    ``{"op": name, **request}`` returns ``adapter.<name>(state, **request) or {}``.
    """
    state = None

    def handle(request: dict) -> dict:
        nonlocal state
        op = request.pop("op")
        if op == "load":
            state = adapter.load(**request)
            return {}
        func = getattr(adapter, op, None)
        if op.startswith("_") or not callable(func):
            raise ValueError(f"Unknown operation {op!r}")
        return func(state, **request) or {}

    return handle


def serve_adapter(path: PathLike, *, stdin=None) -> None:
    """Load the adapter file at ``path`` and :func:`serve` it (what a :class:`Worker` runs).

    The adapter is loaded after stdout is protected, so what it prints on import goes
    to the log too.
    """
    replies = _protect_stdout()
    spec = importlib.util.spec_from_file_location(
        "ps_isolated_adapter", os.fspath(path)
    )
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter  # what dataclasses and pickle look up
    spec.loader.exec_module(adapter)
    _serve_loop(adapter_handler(adapter), stdin or sys.stdin, replies)


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
    seconds the process is killed and :class:`subprocess.TimeoutExpired` raised.

    >>> import sys
    >>> run_process([sys.executable, "-c", "print('hi')"])
    (0, 'hi\\n')
    """
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        env={**clean_environ(), **(env or {})},
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout
