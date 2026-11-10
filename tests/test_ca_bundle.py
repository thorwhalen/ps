"""ps.isolated.ensure_ca_bundle: a worker whose Python's OpenSSL has no CA bundle gets one."""

import os
import sys
import types

import pytest

from ps import isolated
from ps.isolated import Worker, ensure_ca_bundle

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="Windows uses the system store")


def _paths(cafile=None, capath=None):
    return lambda: types.SimpleNamespace(
        openssl_cafile=cafile, openssl_capath=capath,
        openssl_cafile_env="SSL_CERT_FILE", openssl_capath_env="SSL_CERT_DIR")


@pytest.fixture
def system_bundle(tmp_path, monkeypatch):
    system = tmp_path / "system.pem"
    system.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setattr(isolated, "SYSTEM_CA_BUNDLES", (str(tmp_path / "missing.pem"), str(system)))
    return str(system)


@posix_only
def test_a_python_with_a_bundle_is_left_alone(tmp_path):
    bundle = tmp_path / "cert.pem"
    bundle.write_text("x")
    env = {}
    assert ensure_ca_bundle(environ=env, verify_paths=_paths(str(bundle))) is None
    folder = tmp_path / "certs"
    folder.mkdir()
    (folder / "5ad8a5d6.0").write_text("x")  # a hashed certificate name
    assert ensure_ca_bundle(environ=env, verify_paths=_paths(None, str(folder))) is None
    assert env == {}


@posix_only
def test_an_empty_file_or_a_folder_without_hashed_certs_counts_as_none(tmp_path, system_bundle):
    empty = tmp_path / "cert.pem"
    empty.write_text("")
    folder = tmp_path / "certs"
    folder.mkdir()
    (folder / ".keepme").write_text("")  # Homebrew's openssl@3/certs
    (folder / "ca-bundle.crt").write_text("x")  # not a name OpenSSL reads there
    env = {}
    assert ensure_ca_bundle(environ=env, verify_paths=_paths(str(empty), str(folder))) == system_bundle
    assert env == {"SSL_CERT_FILE": system_bundle}


@posix_only
def test_the_system_bundle_comes_before_certifi(system_bundle, monkeypatch):
    monkeypatch.setitem(sys.modules, "certifi", types.SimpleNamespace(where=lambda: "/certifi.pem"))
    assert ensure_ca_bundle(environ={}, verify_paths=_paths()) == system_bundle


@posix_only
def test_certifi_when_no_system_bundle(monkeypatch):
    monkeypatch.setattr(isolated, "SYSTEM_CA_BUNDLES", ())
    monkeypatch.setitem(sys.modules, "certifi", types.SimpleNamespace(where=lambda: "/certifi.pem"))
    env = {}
    assert ensure_ca_bundle(environ=env, verify_paths=_paths()) == "/certifi.pem"
    assert env == {"SSL_CERT_FILE": "/certifi.pem"}
    monkeypatch.setitem(sys.modules, "certifi", None)  # import certifi -> ImportError
    env = {}
    assert ensure_ca_bundle(environ=env, verify_paths=_paths()) is None and env == {}


def test_a_bundle_the_user_chose_wins(system_bundle):
    env = {"SSL_CERT_FILE": "/mine.pem"}
    assert ensure_ca_bundle(environ=env, verify_paths=_paths()) is None
    assert env == {"SSL_CERT_FILE": "/mine.pem"}
    assert ensure_ca_bundle(environ={"SSL_CERT_DIR": "/d"}, verify_paths=_paths()) is None


def test_windows_is_left_to_its_system_store(system_bundle, monkeypatch):
    monkeypatch.setattr(isolated.sys, "platform", "win32")
    env = {}
    assert ensure_ca_bundle(environ=env, verify_paths=_paths()) is None and env == {}


def test_this_python_is_left_alone_when_it_verifies():
    import ssl

    paths = ssl.get_default_verify_paths()
    if not (paths.openssl_cafile and os.path.isfile(paths.openssl_cafile)):
        pytest.skip("this Python has no default CA file")
    env = {}
    assert ensure_ca_bundle(environ=env) is None and env == {}


ADAPTER = """
import os, ssl
def load(**kw):
    return {}
def cas(state):
    return {"file": os.environ.get("SSL_CERT_FILE"),
            "loaded": ssl.create_default_context().cert_store_stats()["x509_ca"]}
"""

MACPORTS = "/opt/local/bin/python3.12"


@pytest.mark.skipif(not os.path.exists(MACPORTS), reason="needs a Python whose OpenSSL has no CA bundle")
def test_a_worker_on_a_python_without_a_bundle_verifies(tmp_path, monkeypatch):
    import subprocess

    probe = subprocess.run([MACPORTS, "-c", "import ssl; print(ssl.create_default_context()"
                            ".cert_store_stats()['x509_ca'])"], capture_output=True, text=True,
                           env={k: v for k, v in os.environ.items() if not k.startswith("SSL_CERT")})
    if probe.stdout.strip() != "0":
        pytest.skip("that Python has CA certificates after all")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)  # the host's must not hide the case
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    adapter = tmp_path / "ca_adapter.py"
    adapter.write_text(ADAPTER)
    with Worker(MACPORTS, adapter) as w:
        got = w.call("cas")
    assert got["file"] and got["loaded"] > 0  # certificates are loaded: HTTPS can verify
