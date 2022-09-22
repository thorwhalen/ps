"""Base objects for ps"""

from typing import Iterable, Mapping, Callable
from ps.util import (
    simple_run_command,
    local_identifier_command_dict,
    IdentifierCommandDict,
    IdentifiedCommands,
)

from dataclasses import dataclass


def mk_raw_command_func(command: str):
    def run_command(command_args_str: str = ''):
        return simple_run_command(command_args_str + ' ' + command)

    return run_command


def join_if_not_string(iterable: Iterable) -> str:
    if not isinstance(iterable, str):
        iterable = ' '.join(iterable)
    return iterable


# TODO: Figure out the standard CLI language: command, instruction, argument, parameter


@dataclass
class Command:
    command: str

    def __call__(self, args: Iterable = ''):
        return self.raw_call(args)

    def raw_call(self, args: Iterable = ''):
        return simple_run_command(self.instruction_str(args))

    def instruction_str(self, args: Iterable):
        args_str = join_if_not_string(args)
        return f'{self.command} {args_str}'

    # TODO: Include 'intelligence' to find the appropriate help string
    def help_str(self):
        return self._man_page_str()

    def help(self):
        return print(self.help_str())

    def _man_page_str(self):
        return simple_run_command(f'man {self.command}')

    def _dash_h_str(self):
        return simple_run_command(f'{self.command} -h')

    def _dash_dash_help_str(self):
        return simple_run_command(f'{self.command} --help')


def _ensure_identifier_keyed_dict(
    commands: IdentifiedCommands,
) -> IdentifierCommandDict:
    if callable(commands):
        commands = commands()
    if isinstance(commands, Mapping):
        d = dict(commands)
    else:
        d = {x: x for x in commands}
    non_identifier_keys = list(filter(lambda x: not x.isidentifier(), d))
    if non_identifier_keys:
        raise ValueError(f'These were not identifiers: {non_identifier_keys}')
    return d


class Commands(Mapping):
    def __init__(
        self,
        commands: IdentifiedCommands = local_identifier_command_dict,
        command_runner_factory: Callable[[str], Callable] = Command,
    ):
        self._commands = _ensure_identifier_keyed_dict(commands)
        for name, command in self._commands.items():
            setattr(self, name, command_runner_factory(command))
        self._command_runner_factory = command_runner_factory

    def __getitem__(self, command):
        try:
            return getattr(self, command)
        except AttributeError:
            raise KeyError(f"This command was not found: '{command}'")

    def __iter__(self):
        yield from self._commands

    def __len__(self):
        return len(self._commands)

    def __contains__(self, command):
        return command in self._commands

