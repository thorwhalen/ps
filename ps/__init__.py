"""Call any available system command from python.

Importing ``ps`` binds one callable per executable found on your ``PATH``, named
after a python-safe version of the command name, so shell tools are reachable as
ordinary python functions::

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
generators do -- will execute every binary on the machine.
"""

from ps.base import Command, Commands

# Bind every locally available system command as a module-level callable.
# `Commands()` is a Mapping of {identifier: callable}, so updating the module
# namespace with it is enough -- and, unlike an explicit loop, it leaves behind
# no temporary names needing cleanup. (Deleting the loop variables raised
# NameError whenever no command was found, which is exactly what happened on
# Windows, where PATH was never parsed correctly.)
globals().update(Commands())
