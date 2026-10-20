"""Repo-level pytest hooks."""

import os

import pytest

#: Doctests that drive POSIX-only commands (``ls``, man pages, ``sh``).
_POSIX_ONLY_DOCTESTS = (
    "ps.base.Commands",
    "ps.misc",
    "ps.misc.run_commands_in_dir",
    "ps.util.run",
)


def pytest_collection_modifyitems(config, items):
    if os.name != "nt":
        return
    skip = pytest.mark.skip(reason="doctest drives POSIX-only commands")
    for item in items:
        if item.name in _POSIX_ONLY_DOCTESTS:
            item.add_marker(skip)
