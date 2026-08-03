"""Regression tests for how ``ps`` discovers commands on ``PATH``.

Both tests here pin down bugs that CI caught only on Windows, where they
combined to make ``import ps`` raise ``NameError`` outright.
"""

import os
import subprocess
import sys

from ps.util import local_commands


def test_import_survives_an_empty_path():
    """Importing ``ps`` must work even when no command is discoverable.

    ``ps/__init__.py`` used to delete its loop variables after binding the
    commands. With nothing to iterate over, those names were never bound and
    the ``del`` raised ``NameError`` while importing the package.

    Run in a subprocess so the empty ``PATH`` cannot leak into this session,
    and so the failure surfaces as an import of a genuinely fresh interpreter.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import ps; print(len(ps.Commands()))"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": ""},
    )
    assert result.returncode == 0, f"importing ps failed:\n{result.stderr}"
    assert result.stdout.strip() == "0"


def test_path_is_split_on_the_platform_separator(tmp_path, monkeypatch):
    """``PATH`` entries must be split on ``os.pathsep``, not a hardcoded ":".

    On Windows the separator is ";", so a hardcoded ":" left ``PATH`` as one
    unsplit string, no entry survived the "is it a directory?" filter, and no
    command was ever found -- which is what made ``import ps`` blow up there.

    ``os.pathsep`` is patched to the Windows separator so this discriminates on
    POSIX too; otherwise the two separators coincide and the bug is invisible.
    """
    monkeypatch.setattr(os, "pathsep", ";")

    first, second = tmp_path / "first", tmp_path / "second"
    for directory, name in ((first, "alpha_cmd"), (second, "beta_cmd")):
        directory.mkdir()
        executable = directory / name
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)

    monkeypatch.setenv("PATH", ";".join(map(str, (first, second))))
    commands = local_commands()

    assert "alpha_cmd" in commands, "first PATH entry was not scanned"
    assert "beta_cmd" in commands, "second PATH entry was not scanned"
