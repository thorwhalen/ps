"""
Misc tools for commands.

Utilities in this module include a small helper to run shell snippets from a
specified directory while keeping strict error propagation. See
`run_commands_in_dir` for a full, documented example — a short demo is shown
below.

Short demo
----------

>>> from ps.misc import run_commands_in_dir
>>> cp = run_commands_in_dir('.', "pwd; python -c 'print(40+2)'")
>>> cp.returncode
0
>>> '42' in cp.stdout
True
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Union


def run_commands_in_dir(
    directory: str | Path,
    commands: str,
    *,
    timeout: float | None = None,
    env: dict | None = None,
    shell_path: str = "/bin/bash",
) -> subprocess.CompletedProcess:
    """
    Run one or several shell commands *as if* they were executed from `directory`.

    This function spawns a fresh shell whose working directory is set to
    `directory`. It prepends `set -e -o pipefail` so the shell exits immediately
    on the first failing command; failures in pipelines also fail the script.

    Parameters
    ----------
    directory
        Target working directory. Does not affect the caller's CWD.
    commands
        A shell script snippet: one or more commands, possibly multi-line.
    timeout
        Optional timeout in seconds for the whole command block.
    env
        Optional environment overrides to pass to the child process.
    shell_path
        Path to the shell to use (defaults to /bin/bash). The shell is invoked
        as `shell_path -lc "<script>"`.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process with `stdout`, `stderr`, and `returncode` fields.

    Raises
    ------
    subprocess.CalledProcessError
        If any command fails. The exception includes `.returncode`, `.stdout`,
        and `.stderr` with full context for debugging.

    Notes
    -----
    - For Git-only sequences you *could* prefix each command with `git -C DIR`,
      but changing the working directory for the subprocess is more general and
      less error-prone when mixing tools.

    Examples
    --------
    Basic usage on any system:

    >>> cp = run_commands_in_dir(
    ...     ".", "pwd; python -c 'print(40+2)'"
    ... )
    >>> cp.returncode
    0
    >>> "42" in cp.stdout
    True

    Create a temporary Git repo, make a commit, and push to a bare repo:
    (This is a full integration test and may be skipped if Git isn't available.)

    >>> import tempfile, os, textwrap
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     tmp = Path(tmp)
    ...     bare = tmp / "remote.git"
    ...     work = tmp / "work"
    ...     bare.mkdir()
    ...     _ = run_commands_in_dir(
    ...         bare,
    ...         "git init --bare --initial-branch=main >/dev/null"
    ...     )
    ...     _ = run_commands_in_dir(
    ...         tmp,
    ...         f"git clone {bare} {work.name} >/dev/null"
    ...     )
    ...     # Create a file, add/commit, and push
    ...     script = textwrap.dedent('''
    ...         set -x
    ...         # Ensure a user is configured so commit doesn't fail in CI/tmp dirs
    ...         git config user.email "test@example.com"
    ...         git config user.name "Test User"
    ...         git status --porcelain
    ...         echo "hello" > hello.txt
    ...         git add -A
    ...         git commit -m "Initial commit"
    ...         # No need to pull in a fresh clone; push the new branch to the bare repo
    ...         git push -u origin HEAD
    ...     ''')
    ...     cp = run_commands_in_dir(work, script)
    ...     cp.returncode
    0

    Error propagation example (raises CalledProcessError):

    >>> _ = run_commands_in_dir(".", "false")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    CalledProcessError: ...
    """
    cwd = Path(directory).resolve()
    if not cwd.exists():
        raise FileNotFoundError(f"Directory does not exist: {cwd}")

    # Robust shell header: -e -> fail fast; pipefail -> propagate pipeline errors.
    # Avoid `set -u` to not break on benign unset variables in user scripts.
    script = f"set -e -o pipefail\n{commands}"

    # Use text mode to get str outputs; capture both streams.
    # check=True makes Python raise CalledProcessError on nonzero exit.
    cp = subprocess.run(
        [shell_path, "-lc", script],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return cp
