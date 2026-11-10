"""A module driver for tests: echoes requests, prints noise to stdout, fails on demand."""

import os
import sys

from ps.isolated import serve


def handle(request):
    print("library noise on stdout")  # must not reach the protocol channel
    os.write(1, b"C-level noise on fd 1\n")  # nor this
    if request.get("fail"):
        raise ValueError("asked to fail")
    if request.get("die"):
        sys.stderr.write("dying clue in the log\n")
        os._exit(3)
    import ps

    return {
        "echo": request,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "ps_file": ps.__file__,
    }


if __name__ == "__main__":
    serve(handle)
