
# ps
Call any available system command from python

To install:	```pip install ps```

# Quick Start

`Commands()` will gather all available commands it can find
(searching through the same `PATH` environment variable that your system does)
and makes them available for execution. 

```
>>> from ps import Commands
>>> c = Commands()
>>> c.echo('hello world')
'hello world'
>>> c.ls('-la')
total 6271272
drwxr--r--@ 316 Thor.Whalen  staff      10112 Sep 22 14:01 .
drwxr-xr-x@  17 Thor.Whalen  staff        544 Mar  8  2022 ..
...
```

A `Commands` instance is also a `Mapping`, so you can do things like:
```python
len(c)
# 3097
list(c)
# ['_2to3',
#  'Activate_ps1',
#  'AssetCacheLocatorUtil'
#  ...
# ]
'ls' in c
# True
'this_command_does_not_exist' in c
# False
py = c.get('python3.9', c.get('python3.8', None))
c['ls']
# Command(command='ls')
```

You can get help like so:

```
>>> c.ls.help()
LS(1)                     BSD General Commands Manual                    LS(1)

NAME
     ls -- list directory contents
...
```

Or just get the help string like so:

```
>>> help_string = c.ls.help_str()
```

# More control

`Commands` is just a collection of `Command` instances, which itself is 
just a callable that will run a shell script with you, in a manner 
specified by the `run` function. 

So let's have a quick look at those three objects to understand 
better what powers we have at our disposal.

## run

A parametrizable way to run shell commands.

Works somewhat like the [subprocess.run](
<https://docs.python.org/3/library/subprocess.html#subprocess.run>) function.
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

## Command

A `Command` runs a specific shell script for you in a specific manner. 
The `run` function is the general function to do that, and we saw 
that you can curry `run` to specify what and how to run.
`Command` just wraps such a curried `run` function
(or any compliant run function you provide),
and specifies what executable (the `command` argument) to actually run.

So not much over a curried `run`. 

But what it does do as well is set up the ability to do other things 
that may be specific to the executable you're running, such as 
giving your (callable) command instance a signature, some docs, or 
a help method. 

    >>> import os
    >>> pwd = Command('pwd')
    >>> os.path.isdir(pwd())
    True
    >>> assert pwd.__doc__  # docs exist (and are non-empty)!
    >>> # To print the docs:
    >>> pwd.help()  # doctest: +SKIP
    PWD(1)                    BSD General Commands Manual                   PWD(1)

    NAME
         pwd -- return working directory name

    SYNOPSIS
         pwd [-L | -P]
    ...

## Commands

A collection of commands.

The general usage is that you can specify a mapping between valid python identifiers
(alphanumerical strings (and underscores) that don't start with a number) and
functions. If instead of functions you specify a string, a ``factory`` comes
in play to make a function based on your string.
By default, it will consider it as a console command and give you a function that
runs it.

    >>> import os
    >>>
    >>> c = Commands({
    ...     'current_dir': 'pwd',
    ...     'sys_listdir': 'ls -l',
    ...     'listdir': os.listdir,
    ...     'echo': Command('echo', egress=lambda x: print(x.decode().strip())),
    ... })
    >>>
    >>> list(c)
    ['current_dir', 'sys_listdir', 'listdir', 'echo']
    >>> current_folder = c.current_dir()
    >>> os.path.isdir(current_folder)
    True
    >>> b = c.sys_listdir()
    >>> b[:40]  # doctest: +SKIP
    b'total 56\n-rw-r--r--@ 1 Thor.Whalen  staf'
    >>> a_list_of_filenames = c.listdir()
    >>> isinstance(a_list_of_filenames, list)
    True
    >>> c.echo('hello world')
    hello world

If you don't specify any commands, it will gather all executable names it can find in
your local system (according to your ``PATH`` environment variable),
map those to valid python identifiers if needed, and use that.

Important: Note that finding executable in the ``PATH`` doesn't mean that it will
work, or is safe -- so use with care!

    >>> c = Commands()
    >>> assert len(c) > 0
    >>> 'ls' in c
    True

You can access the 'ls' command as a key or an attribute

    >>> assert c['ls'] == c.ls

You can print the ``.help()`` (docs) of any command, or just get the help string:

    >>> man_page = c.ls.help_str()

Let's see if these docs have a few things we expect it to have for ``ls``:

    >>> assert 'BSD General Commands Manual' in man_page
    >>> assert 'list directory contents' in man_page

Let's see what the output of ``ls`` gives us:

    >>> output = c.ls('-la').decode()  # execute "ls -la"
    >>> assert 'total' in output  # "ls -l" output usually starts with "total"
    >>> assert '..' in output  # the "-a" flag includes '..' as a file

Note that we needed to decode the output here.
That's because by default the output of a command will be captured in bytes.
If you want to apply a decoder to (attempt to) convert all outputs into strings,
you can specify a ``factory`` that will do this for you automatically.

The default ``factory`` is ``Command``, which  has a ``run``
argument that defines how an instruction should be run. 
The default of ``run`` is ``run_command``, which conveniently has an ``egress`` 
argument where you can specify a function to call on the output. 

So one solution to define a ``Commands`` instance that will (attempt to) output 
strings systematically is to do this:

    >>> from functools import partial
    >>> from ps.util import run
    >>> run_and_cast_to_str = partial(run, egress=bytes.decode)
    >>> factory = partial(Command, run=run_and_cast_to_str)
    >>> cc = Commands(factory=factory)

So now we have:

    >>> output = cc.ls('-la')
    >>> isinstance(output, str)  # it's already decoded for us!
    True

# Notes

## Auto docs

Trying to figure out a robust way to get doc strings.
Unfortunately, `man <command>` and `<command> --help` don't always exist,
and the former isn't even always correct
(i.e. linked to the same executable as what `which <command>` says)

See: https://stackoverflow.com/questions/73814043/universal-help-for-terminal-commands

## Misc references

[Tutorial on options for calling shell commands from python](https://janakiev.com/blog/python-shell-commands/)


