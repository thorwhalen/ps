"""Regression tests for how ``ps`` discovers and identifies commands.

The first two pin down bugs that CI caught only on Windows, where they
combined to make ``import ps`` raise ``NameError`` outright. The rest pin the
safety and the semantics of turning a command name into a python identifier.
"""

import os
import subprocess
import sys

import pytest

from ps.util import (
    identifier_mapping,
    is_executable_according_to_which,
    local_commands,
)


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


@pytest.mark.skipif(sys.platform == "win32", reason="`which` is POSIX-only")
def test_which_does_not_execute_injected_commands(tmp_path):
    """The argument is a command name, not a shell fragment to evaluate.

    It used to be interpolated unquoted into a shell, so anything after a ";"
    was executed -- an "is this on PATH?" question with arbitrary side effects.
    """
    marker = tmp_path / "PWNED"

    is_executable_according_to_which(f"ls; touch {marker}")

    assert not marker.exists(), "the argument was executed as a shell fragment"


@pytest.mark.skipif(sys.platform == "win32", reason="`which` is POSIX-only")
def test_which_still_finds_real_commands():
    """Quoting the argument must not change the answer for real command names."""
    assert is_executable_according_to_which("ls")
    assert not is_executable_according_to_which("no_such_cmd_xyz")


def test_identifier_mapping_tolerates_repeated_commands():
    """A string repeated in the input is not in collision with itself."""
    assert identifier_mapping(["ls", "ls", "pwd"]) == {"ls": "ls", "pwd": "pwd"}


def test_identifier_mapping_still_rejects_real_collisions():
    """Two *distinct* strings mapping to one identifier is still an error."""
    with pytest.raises(ValueError, match="foo_bar"):
        identifier_mapping(["foo-bar", "foo.bar"])
