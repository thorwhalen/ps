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
``from ps import ls``, ``from ps import *``, ``dir(ps)``), not on ``import ps``,
so importing a submodule (e.g. :mod:`ps.isolated`) costs no scan. The commands
are therefore no longer entries of the module's ``__dict__`` (``vars(ps)``).
"""

import pkgutil as _pkgutil
import threading as _threading

from ps.base import Command, Commands

#: Submodule names: never commands, so asking for one does not scan ``PATH``.
_SUBMODULES = frozenset(name for _, name, _ in _pkgutil.iter_modules(__path__))
_PUBLIC = ("Command", "Commands")
_COMMANDS = None  # the PATH's commands, found on first use
_COMMANDS_LOCK = _threading.Lock()


def _commands():
    global _COMMANDS
    if _COMMANDS is None:
        with _COMMANDS_LOCK:  # one scan, even when threads ask at once
            if _COMMANDS is None:
                _COMMANDS = Commands()
    return _COMMANDS


def __getattr__(name):
    """``ps.<command>``: the command of that identifier found on ``PATH`` (PEP 562)."""
    if name == "__all__":  # ``from ps import *``: the public names and every command
        return [*_PUBLIC, *_commands()]
    if name.startswith("_") or name in _SUBMODULES:
        raise AttributeError(f"module 'ps' has no attribute {name!r}")
    try:
        return _commands()[name]
    except KeyError:
        raise AttributeError(f"module 'ps' has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(_commands()))
