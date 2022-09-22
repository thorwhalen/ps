"""Utils"""

import os
import re
from collections import defaultdict
from typing import Iterable, Dict, Union, Callable
from warnings import warn
from itertools import chain
from functools import partial

Identifier = str  # but satisfying str.isidentifier
IdentifierCommandDict = Dict[Identifier, str]
IdentifiedCommands = Union[
    Iterable[Identifier], IdentifierCommandDict, Callable[..., IdentifierCommandDict]
]


def simple_run_command(cmd, *, strip_output=True):
    with os.popen(cmd) as stream:
        output = stream.read()
    if strip_output:
        output = output.strip()
    return output


is_executable_path = partial(os.access, mode=os.X_OK)
# directories are also executable, so could need:
is_executable_file = lambda path: os.path.isfile(path) and is_executable_path(path)


def is_executable_according_to_which(string: str):
    """
    Says if a string is an executable command according to the (linux) which command.

    That is, it will try resolving finding the executable file with a ``which COMMAND``
    command, deciding the ``COMMAND`` is indeed an executable if, and only if, ``which``
    comes back with something.


    See: https://linuxize.com/post/linux-which-command

    """
    return bool(simple_run_command(f'which {string}', strip_output=True))


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
            _non_existing_dirs = '\n\t' + '\n\t'.join(non_existing_dirs)
            if verbose:
                warn(
                    'These paths were in your PATH environment variable, but were not '
                    f'found as directories:{_non_existing_dirs}'
                )
        return sorted(existing_dirpaths)

    def _executables_of_dir(dirpath):
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            if is_executable_file(filepath):
                yield filename

    dirpaths = os.environ.get('PATH', '').split(':')
    dirpaths = _keep_only_existing_paths(dirpaths, verbose)

    def _commands():
        for dirpath in dirpaths:
            for command in _executables_of_dir(dirpath):
                yield command

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
        return re.sub('\W', '_', string)

    def _first_character_is_a_digit(string: str):
        if len(string) == 0:
            raise ValueError('string was empty')
        first_character, *_ = string
        return bool(re.match('\d', first_character))

    def _prefix_with_underscore_if_starts_with_digit(string: str):
        if _first_character_is_a_digit(string):
            return '_' + string
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


# TODO: Could resolve collisions (e.g. suffixing with _1, _2, etc.) instead of warning
def identifier_mapping(
    strings: Iterable[str], str_to_id=str_to_identifier
) -> Dict[str, Identifier]:
    """
    Maps strings to identifiers, returning a map from identifiers to the strings,
    warning about any collisions (when two distinct strings map to the same
    identifier
    """
    strings = list(strings)
    str_of_id = {str_to_id(string): string for string in strings}
    if len(str_of_id) != len(strings):
        duplicates = _gather_duplicates(strings, str_to_id)
        raise ValueError(f'Some commands mapped to the same identifier: {duplicates}')
    return str_of_id


def local_identifier_command_dict(
    str_to_id=str_to_identifier, verbose=False
) -> Dict[Identifier, str]:
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
