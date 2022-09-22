
# ps
Call any available system command from python

To install:	```pip install ps```

# Examples

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

A `Command` instance is also a `Mapping`, so you can do things like:
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

If there's a man page, you can get help like so:

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

# Notes

## Auto docs

Trying to figure out a robust way to get doc strings.
Unfortunately, `man <command>` and `<command> --help` don't always exist,
and the former isn't even always correct
(i.e. linked to the same executable as what `which <command>` says)

See: https://stackoverflow.com/questions/73814043/universal-help-for-terminal-commands

