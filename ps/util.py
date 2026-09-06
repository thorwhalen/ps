"""Utils"""

import os
import re
from math import isinf, isnan
from shlex import quote as shlex_quote, split as shlex_split
from subprocess import PIPE, Popen, TimeoutExpired
from itertools import chain
from collections import defaultdict
from typing import Dict, Union
from collections.abc import Iterable, Callable
from warnings import warn
from functools import partial

# How long to keep draining a killed process's pipes before giving up on them.
# A killed child can leave grandchildren holding the write end, in which case
# draining would block for as long as *they* live -- the very hang a timeout is
# there to prevent.
_REAP_TIMEOUT = 1.0

# Dropping our end of a pipe another thread may still be reading is safe on POSIX,
# where ``Popen.communicate`` drains in the calling thread, but not on Windows,
# where it drains in reader threads: ``close()`` would then block on the stream's
# buffer lock for as long as the read does -- exactly the hang we are avoiding.
# On Windows we therefore leave the pipes to those (daemon) reader threads.
_CAN_CLOSE_PIPES_MID_READ = os.name != "nt"

Identifier = str  # but satisfying str.isidentifier
IdentifierCommandDict = dict[Identifier, str]
IdentifiedCommands = Union[
    Iterable[Identifier], IdentifierCommandDict, Callable[..., IdentifierCommandDict]
]


class ProcessError(OSError):
    """To be raised when running a command yields an error"""


class ProcessTimeout(ProcessError):
    """To be raised when a command did not finish within its allotted time"""


def raise_process_error(stderr):
    raise ProcessError(stderr.decode())


def simple_run_command(cmd, *, strip_output=True):
    with os.popen(cmd) as stream:
        output = stream.read()
    if strip_output:
        output = output.strip()
    return output


def strip(x):
    return x.strip()


def _validated_timeout(timeout):
    """``timeout``, validated: a non-negative number of seconds, or ``None``.

    Both ``None`` and an infinite timeout mean "wait forever" -- ``float('inf')``
    is a natural way to spell "no limit", but ``subprocess`` cannot consume it.

    >>> _validated_timeout(None) is _validated_timeout(float('inf')) is None
    True
    >>> _validated_timeout(2.5)
    2.5

    Anything that could not bound a wait is rejected up front, before a process
    exists to be leaked by the failure:

    >>> _validated_timeout('5')
    Traceback (most recent call last):
      ...
    TypeError: timeout should be a number of seconds, or None. Got: '5'
    >>> _validated_timeout(-1)
    Traceback (most recent call last):
      ...
    ValueError: timeout should be a non-negative number of seconds, or None. Got: -1
    """
    if timeout is None:
        return None
    if not isinstance(timeout, (int, float)):
        raise TypeError(
            f"timeout should be a number of seconds, or None. Got: {timeout!r}"
        )
    if isnan(timeout) or timeout < 0:
        raise ValueError(
            f"timeout should be a non-negative number of seconds, or None. "
            f"Got: {timeout!r}"
        )
    return None if isinf(timeout) else timeout


def _kill_and_reap(process, *, reap_timeout=_REAP_TIMEOUT):
    """Kill ``process`` and let go of its pipes, without blocking the caller.

    ``Popen.communicate`` drains the pipes, but a killed child can leave
    grandchildren holding their write ends, in which case draining blocks for as
    long as *those* live -- the very hang a timeout is there to prevent. So the
    drain is bounded, and we fall back to dropping our own ends of the pipes
    (where that is safe -- see ``_CAN_CLOSE_PIPES_MID_READ``) and collecting the
    already-killed direct child.

    Note that only the direct child is killed: anything *it* spawned is not ours
    to find, and outlives the call.
    """
    process.kill()
    try:
        process.communicate(timeout=reap_timeout)
        return
    except TimeoutExpired:
        pass
    if _CAN_CLOSE_PIPES_MID_READ:
        for stream in filter(None, (process.stdin, process.stdout, process.stderr)):
            stream.close()
    try:
        process.wait(timeout=reap_timeout)
    except TimeoutExpired:
        pass


def run(
    *args,
    timeout=None,
    on_error=raise_process_error,
    egress=strip,
    stdout=PIPE,
    stderr=PIPE,
    **kwargs,
):
    """
    A parametrizable way to run shell commands.

    :param args: The "command" or "instruction" to run.
    It can be the same string you'd type in the console, or a tokenized sequence of
    its command and arguments.
    :param timeout: Maximum number of seconds to wait for the command to finish.
    ``None`` (the default) waits forever, as it always has, and so does ``inf``.
    When the command overruns, the command's own process is killed and a
    ``ProcessTimeout`` (a ``ProcessError``, therefore an ``OSError``) is raised.
    Note that processes the command itself spawned are not ours to find, so they
    are not killed with it.
    :param on_error: A function to be called on the stderr generated by running the
    command, if the stderr is not empty.
    :param egress: A function to call on the stdout. The output of this function is
    what will be returned to the user.
    :param stdout, stderr, kwargs: Extra ``subprocess.Popen`` arguments.
    :return: The output of running the command.

    Works somewhat like the `subprocess.run
    <https://docs.python.org/3/library/subprocess.html#subprocess.run>`_ function,
    but with different defaults, as well as the additional arguments `on_error` and
    `egress`.

    >>> output = run('pwd')
    >>> os.path.isdir(output)  # verify that output is indeed a valid directory path
    True

    Also very important difference with ``subprocess.run``:
    You don't specify a LIST of tokenized arguments here:
    You can specify the full (string) command or parts of it as a sequence of strings:

    >>> assert run('echo hello world') == run('echo', 'hello', 'world') == b'hello world'

    Note that ``run`` will return ``bytes`` of the output, stripped of extremal
    newlines. The argument that does the stripping is ``egress``.
    You can use this argument to do something else with the output.
    For example, if you want to to cast the output to a ``str``, strip it, then
    print it, you could specify this in the ``egress``:

    >>> run('echo hello world', egress=lambda x: print(x.decode().strip()))
    hello world

    ``run``'s purpose in life is designed to be curried.
    That is, you can use ``functools.partial`` to make your own specialized
    functions that use shell scripts as their backend.

    >>> from functools import partial
    >>> stripped_str = lambda x: x.decode().strip()
    >>> pwd = partial(run, 'pwd', egress=stripped_str)
    >>> ls_la = partial(run, 'ls', '-la', egress=lambda x: print(stripped_str(x)))
    >>> current_dir = pwd()
    >>> os.path.isdir(current_dir)
    True
    >>> ls_la(current_dir)  # doctest: +SKIP
    total 56
    drwxr-xr-x@  7 Thor.Whalen  staff   224 Sep 23 12:12 .
    drwxr-xr-x@ 11 Thor.Whalen  staff   352 Sep 23 11:33 ..
    -rw-r--r--@  1 Thor.Whalen  staff    48 Sep 22 12:47 __init__.py
    -rw-r--r--@  1 Thor.Whalen  staff  4649 Sep 23 11:33 base.py
    -rw-r--r--@  1 Thor.Whalen  staff   348 Sep 22 12:38 raw.py
    -rw-r--r--@  1 Thor.Whalen  staff  8980 Sep 23 12:12 util.py

    """
    timeout = _validated_timeout(timeout)
    args = list(chain.from_iterable(map(shlex_split, args)))
    process = Popen(args, stdout=stdout, stderr=stderr, **kwargs)
    try:
        output, error = process.communicate(timeout=timeout)
    except TimeoutExpired:
        _kill_and_reap(process)
        # ``from None``: the internal TimeoutExpired adds a second stack without
        # adding information -- the message below already says everything.
        raise ProcessTimeout(f"Timed out after {timeout}s: {' '.join(args)}") from None
    except BaseException:
        # Anything else out of ``communicate`` (a KeyboardInterrupt, say) would
        # otherwise leave the command running and its pipes open.
        _kill_and_reap(process)
        raise
    if error:
        return on_error(error)
    else:
        return egress(output)


def str_if_bytes(x, encoding="utf-8", errors="strict"):
    if isinstance(x, bytes):
        x = x.decode(encoding, errors)
    return x


def print_text_egress(
    output, *, encoding="utf-8", errors="strict", end="\n", file=None
):
    """
    Decodes output and prints it (with control on decoder and printing).
    A useful ``egress`` argument for the ``run`` function.
    """
    return print(str_if_bytes(output, encoding, errors), end=end, file=file)


is_executable_path = partial(os.access, mode=os.X_OK)
# directories are also executable, so could need:
is_executable_file = lambda path: os.path.isfile(path) and is_executable_path(path)


def is_executable_according_to_which(string: str):
    """
    Says if a string is an executable command according to the (linux) which command.

    That is, it will try resolving finding the executable file with a ``which COMMAND``
    command, deciding the ``COMMAND`` is indeed an executable if, and only if, ``which``
    comes back with something.


    Note that ``string`` is treated as a single command name, not as a shell
    fragment: it is quoted before being handed to the shell, so metacharacters in
    it are never interpreted (and never executed).

    See: https://linuxize.com/post/linux-which-command

    """
    return bool(simple_run_command(f"which {shlex_quote(string)}", strip_output=True))


# TODO: Generalize to DOS
# See options for getting available commands here:
# https://stackoverflow.com/questions/948008/linux-command-to-list-all-available-commands-and-aliases
def local_commands(verbose=False):
    """
    Get a list of available commands (strings).

    The function will look at all folders listed in the PATH environment variables,
    and gather all filenames of files therein (in first level of folder only) that
    are executable.

    Essentially do what the command:
    ``ls $(echo $PATH | tr ':' ' ') | grep -v '/' | grep . | sort``
    would, with deduplication.


    """

    def _keep_only_existing_paths(dirpaths, verbose=False):
        dirpaths = set(filter(None, dirpaths))
        existing_dirpaths = set(filter(os.path.isdir, dirpaths))
        if non_existing_dirs := (set(dirpaths) - existing_dirpaths):
            _non_existing_dirs = "\n\t" + "\n\t".join(non_existing_dirs)
            if verbose:
                warn(
                    "These paths were in your PATH environment variable, but were not "
                    f"found as directories:{_non_existing_dirs}"
                )
        return sorted(existing_dirpaths)

    def _executables_of_dir(dirpath):
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            if is_executable_file(filepath):
                yield filename

    # os.pathsep, not ":" -- PATH is separated by ";" on Windows, where a
    # hardcoded ":" yielded a single unsplit string, no valid directories, and
    # therefore no commands at all.
    dirpaths = os.environ.get("PATH", "").split(os.pathsep)
    dirpaths = _keep_only_existing_paths(dirpaths, verbose)

    def _commands():
        for dirpath in dirpaths:
            yield from _executables_of_dir(dirpath)

    return sorted(set(_commands()))


def str_to_identifier(string: str) -> Identifier:
    """
    Transforms a string into an identifier

    >>> str_to_identifier("a-string$with@non*identifier(characters)")
    'a_string_with_non_identifier_characters_'
    >>> str_to_identifier("123go")
    '_123go'
    """

    def _replace_all_non_alphnumerics_with_underscore(string: str):
        return re.sub(r"\W", "_", string)

    def _first_character_is_a_digit(string: str):
        if len(string) == 0:
            raise ValueError("string was empty")
        first_character, *_ = string
        return bool(re.match(r"\d", first_character))

    def _prefix_with_underscore_if_starts_with_digit(string: str):
        if _first_character_is_a_digit(string):
            return "_" + string
        else:
            return string

    identifier = _replace_all_non_alphnumerics_with_underscore(string)
    identifier = _prefix_with_underscore_if_starts_with_digit(identifier)
    return identifier


def _gather_duplicates(values, value_to_group_key):
    """
    >>> _gather_duplicates(['this', 'or', 'that'], len)
    {4: ['this', 'that']}
    """
    d = defaultdict(list)
    for value in values:
        d[value_to_group_key(value)].append(value)
    return {k: group for k, group in d.items() if len(group) > 1}


def _gather_collisions(values, value_to_group_key):
    """Groups of *distinct* values sharing a key -- i.e. genuine collisions.

    A value that simply appears twice is not in collision with itself:

    >>> _gather_collisions(['this', 'this'], len)
    {}
    >>> _gather_collisions(['this', 'that'], len)
    {4: ['this', 'that']}
    """
    return {
        key: group
        for key, group in _gather_duplicates(values, value_to_group_key).items()
        if any(value != group[0] for value in group)
    }


# TODO: Could resolve collisions (e.g. suffixing with _1, _2, etc.) instead of raising
def identifier_mapping(
    strings: Iterable[str], str_to_id=str_to_identifier
) -> dict[str, Identifier]:
    """
    Maps strings to identifiers, returning a map from identifiers to the strings,
    raising on any collision (when two *distinct* strings map to the same
    identifier).

    Note the direction: the keys are the identifiers, the values the strings.

    >>> identifier_mapping(['foo-bar', 'pwd'])
    {'foo_bar': 'foo-bar', 'pwd': 'pwd'}

    A string appearing twice is the same command twice, not a collision:

    >>> identifier_mapping(['ls', 'ls', 'pwd'])
    {'ls': 'ls', 'pwd': 'pwd'}

    Two distinct strings sharing an identifier, on the other hand, is one:

    >>> identifier_mapping(['foo-bar', 'foo.bar'])
    Traceback (most recent call last):
      ...
    ValueError: Some commands mapped to the same identifier: {'foo_bar': ['foo-bar', 'foo.bar']}
    """
    strings = list(strings)
    str_of_id = {str_to_id(string): string for string in strings}
    if len(str_of_id) != len(strings):
        # Fewer identifiers than strings, so *something* shares one. That is only
        # an error if the sharers are distinct strings; a repeat is just a repeat.
        if collisions := _gather_collisions(strings, str_to_id):
            raise ValueError(
                f"Some commands mapped to the same identifier: {collisions}"
            )
    return str_of_id


def local_identifier_command_dict(
    str_to_id=str_to_identifier, verbose=False
) -> dict[Identifier, str]:
    """
    A dict of ``{identifier: command, ...`` for all commands found in the local system.

    ``identifier`` is a python-valid name that uniquely identifies ``command``.
    When ``command`` is a valid identifier itself (as defined by ``str.isidentifier``),
    ``identifier`` is equal to ``command``. But when ``command`` is not
    (when it contains anything that is not alphanumeric or an underscore, for example;
    dots or dashes), the ``identifier`` saves the day to make a valid python function
    name
    """
    return identifier_mapping(local_commands(verbose), str_to_id)
