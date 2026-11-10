"""Call any available system command from python.

Every executable found on your ``PATH`` is reachable as an attribute of ``ps``,
named after a python-safe version of the command name, so shell tools are
reachable as ordinary python functions::

    import ps
    ps.ls('-la')          # runs "ls -la", returns its output as bytes
    ps.ls.help()          # prints the command's man page / --help output

The building blocks live in :mod:`ps.base`:

- :class:`ps.base.Command` wraps a single executable as a callable object.
- :class:`ps.base.Commands` is a ``Mapping`` of many such commands, defaulting
  to everything discoverable on ``PATH``.

:mod:`ps.raw` exposes the same commands as plain module-level globals, and
:mod:`ps.misc` holds extra helpers such as :func:`ps.misc.run_commands_in_dir`.

Note that finding an executable on ``PATH`` does not mean calling it is safe, or
that it will work -- so use with care. In particular, a ``Command``'s ``__doc__``
is computed lazily by *running* the command (``man 1 <cmd>``, then
``<cmd> --help``), so introspecting these objects en masse -- as documentation
generators do -- will execute every binary on the machine. Each such probe gets
its stdin closed and is bounded by :data:`ps.base.DFLT_DOC_TIMEOUT`, so it cannot
hang, but it is still a real execution.

The ``PATH`` is scanned on the first such attribute access (``ps.ls``,
``from ps import ls``, ``dir(ps)``), not on ``import ps``, so importing a
submodule (e.g. :mod:`ps.isolated`) costs no scan.
"""

from pathlib import Path as _Path

from ps.base import Command, Commands

_COMMANDS = None  # the PATH's commands, found on first use


def _commands():
    global _COMMANDS
    if _COMMANDS is None:
        _COMMANDS = Commands()
    return _COMMANDS


def __getattr__(name):
    """``ps.<command>``: the command of that identifier found on ``PATH`` (PEP 562)."""
    # Private names and submodules not imported yet are never commands: no scan.
    if name.startswith("_") or (_Path(__file__).parent / f"{name}.py").exists():
        raise AttributeError(f"module 'ps' has no attribute {name!r}")
    try:
        return _commands()[name]
    except KeyError:
        raise AttributeError(f"module 'ps' has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(_commands()))
