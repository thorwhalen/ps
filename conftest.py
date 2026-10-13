"""Skip the POSIX-only doctests on Windows (thorwhalen/ps#5).

These doctests demonstrate wrapping real system commands found on ``PATH``
(``pwd``, ``ls``) or a ``/bin/bash`` script (``ps.misc.run_commands_in_dir``).
On ``windows-latest`` the bash script cannot start (no ``/bin/bash``), and
``pwd`` resolves to Git for Windows' MSYS one, whose ``/d/a/...`` output is not
a Windows directory. Swapping them for ``python -c ...`` would stop
demonstrating what the package does, and Windows-native replacements need
verifying on Windows first, so they are skipped there by name, and run
everywhere else. ``tests/test_doctest_platforms.py`` fails if a name here stops
matching a doctest, so a rename cannot silently un-skip one or leave a stale
entry.
"""

import sys

import pytest

#: Doctest ids (``module.qualname``) that need a POSIX system.
POSIX_ONLY_DOCTESTS = frozenset(
    {
        "ps.base.Commands",  # wraps `pwd` and `ls -l` / `ls -la`
        "ps.misc",  # run_commands_in_dir: /bin/bash, `pwd`
        "ps.misc.run_commands_in_dir",  # same
        "ps.util.run",  # run('pwd')
    }
)


def pytest_collection_modifyitems(config, items):
    if sys.platform != "win32":
        return
    skip = pytest.mark.skip(
        reason="POSIX-only doctest: needs a POSIX pwd/ls or /bin/bash (ps#5)"
    )
    for item in items:
        # Only doctest items carry these dotted ids; matching by name alone keeps
        # this working on pytest < 7.2, which does not export pytest.DoctestItem.
        if item.name in POSIX_ONLY_DOCTESTS:
            item.add_marker(skip)
