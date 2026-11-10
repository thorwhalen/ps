"""ps.isolated.ensure_ca_bundle: a worker whose Python's OpenSSL has no CA bundle gets one."""

import os
import sys
import types

import pytest

from ps import isolated
from ps.isolated import Worker, ensure_ca_bundle


def _paths(cafile=None, capath=None):
    return lambda: types.SimpleNamespace(openssl_cafile=cafile, openssl_capath=capath)


def test_a_python_with_a_bundle_is_left_alone(tmp_path):
    bundle = tmp_path / "cert.pem"
    bundle.write_text("x")
    env = {}
    assert ensure_ca_bundle(environ=env, verify_paths=_paths(str(bundle))) is None
    assert env == {}
    folder = tmp_path / "certs"
    folder.mkdir()
    (folder / "abc.0").write_text("x")
    assert ensure_ca_bundle(environ=env, verify_paths=_paths(None, str(folder))) is None


def test_an_empty_openssl_gets_certifi_or_a_system_bundle(tmp_path, monkeypatch):
    empty = tmp_path / "certs"
    empty.mkdir()
    env = {}
    got = ensure_ca_bundle(environ=env, verify_paths=_paths(str(tmp_path / "none.pem"), str(empty)))
    try:
        import certifi
        assert got == certifi.where()
    except ImportError:
        assert got in isolated.SYSTEM_CA_BUNDLES or got is None
    assert env.get("SSL_CERT_FILE") == got


def test_without_certifi_the_system_bundle_is_used(tmp_path, monkeypatch):
    system = tmp_path / "system.pem"
    system.write_text("x")
    monkeypatch.setattr(isolated, "SYSTEM_CA_BUNDLES", (str(tmp_path / "missing.pem"), str(system)))
    monkeypatch.setitem(sys.modules, "certifi", None)  # import certifi -> ImportError
    env = {}
    assert ensure_ca_bundle(environ=env, verify_paths=_paths()) == str(system)
    assert env == {"SSL_CERT_FILE": str(system)}


def test_a_bundle_the_user_chose_wins():
    env = {"SSL_CERT_FILE": "/mine.pem"}
    assert ensure_ca_bundle(environ=env, verify_paths=_paths()) is None
    assert env == {"SSL_CERT_FILE": "/mine.pem"}
    assert ensure_ca_bundle(environ={"SSL_CERT_DIR": "/d"}, verify_paths=_paths()) is None


ADAPTER = """
import os
def load(**kw):
    return {}
def cert_file(state):
    return os.environ.get("SSL_CERT_FILE")
"""

MACPORTS = "/opt/local/bin/python3.12"


@pytest.mark.skipif(not os.path.exists(MACPORTS), reason="needs a Python whose OpenSSL has no CA bundle")
def test_a_worker_on_a_python_without_a_bundle_gets_one(tmp_path, monkeypatch):
    import subprocess

    probe = subprocess.run([MACPORTS, "-c", "import ssl, os; p = ssl.get_default_verify_paths(); "
                            "print(os.path.isfile(p.openssl_cafile))"], capture_output=True, text=True)
    if probe.stdout.strip() != "False":
        pytest.skip("that Python has a CA bundle after all")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)  # the host's must not hide the case
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    adapter = tmp_path / "ca_adapter.py"
    adapter.write_text(ADAPTER)
    with Worker(MACPORTS, adapter) as w:
        assert w.call("cert_file") in isolated.SYSTEM_CA_BUNDLES  # no certifi there
