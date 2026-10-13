"""Every doctest the root conftest skips on Windows must exist (ps#5).

The skip list is by name, so a renamed function would otherwise silently start
running (and failing) on Windows again, or leave a stale entry behind.
"""

import doctest
import importlib
import importlib.util
from pathlib import Path

CONFTEST = Path(__file__).resolve().parent.parent / "conftest.py"


def _posix_only_doctests() -> frozenset:
    spec = importlib.util.spec_from_file_location("_ps_root_conftest", CONFTEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.POSIX_ONLY_DOCTESTS


def _doctest_names(module_name: str) -> set:
    module = importlib.import_module(module_name)
    return {t.name for t in doctest.DocTestFinder().find(module) if t.examples}


def test_every_skipped_doctest_exists():
    names = _posix_only_doctests()
    modules = {name.split(".")[0] + "." + name.split(".")[1] for name in names}
    found = set().union(*(_doctest_names(m) for m in modules))
    assert names <= found, f"not doctests any more: {sorted(names - found)}"
