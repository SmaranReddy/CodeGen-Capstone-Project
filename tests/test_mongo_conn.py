"""MongoDB connection helper: TLS only where the connection uses TLS, env precedence, safe errors.

No network: MongoClient is mocked everywhere except the malformed-URI case, which pymongo
rejects while parsing (before any connection). No +srv URI ever reaches a real MongoClient.
"""
import os
import sys
from pathlib import Path
from unittest import mock

import certifi
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/spider1"))
import mongo_conn as M  # noqa: E402
import spider1_kit as K  # noqa: E402

LOCAL = "mongodb://127.0.0.1:27017"
ATLAS = "mongodb+srv://someuser:s3cretpw@cluster0.example.mongodb.net/?retryWrites=true&w=majority"
CA = {"tlsCAFile": certifi.where()}


@pytest.fixture
def clean_env():
    """os.environ snapshot restored afterwards (load_dotenv writes straight into it)."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MONGODB_URI", None)
        yield


@pytest.fixture
def fake_client():
    with mock.patch("pymongo.MongoClient") as mc:
        yield mc


# ---------------------------------------------------------------- 1. local: no TLS kwarg
@pytest.mark.parametrize("uri", [LOCAL, "mongodb://localhost:27017/", "mongodb://127.0.0.1:27017/codegen?directConnection=true",
                                 "mongodb://db.internal.example:27017"])
def test_plain_mongodb_uri_gets_no_tls_kwarg(uri):
    assert M.tls_kwargs(uri) == {}


def test_local_connect_does_not_pass_tlscafile(clean_env, fake_client, tmp_path):
    os.environ["MONGODB_URI"] = LOCAL
    M.connect(tmp_path / ".env", 30000)
    fake_client.assert_called_once_with(LOCAL, serverSelectionTimeoutMS=30000)


# ---------------------------------------------------------------- 2. Atlas +srv: unchanged CA behaviour
@pytest.mark.parametrize("uri", [ATLAS, "MONGODB+SRV://cluster0.example.mongodb.net", "mongodb+srv://c.example.net/db?tls=true"])
def test_srv_uri_keeps_certifi_ca(uri):
    assert M.tls_kwargs(uri) == CA


def test_atlas_connect_kwargs_match_previous_behaviour(clean_env, fake_client, tmp_path):
    os.environ["MONGODB_URI"] = ATLAS
    M.connect(tmp_path / ".env", 30000)
    # identical to the old MongoClient(uri, serverSelectionTimeoutMS=30000, tlsCAFile=certifi.where())
    fake_client.assert_called_once_with(ATLAS, serverSelectionTimeoutMS=30000, tlsCAFile=certifi.where())


# ---------------------------------------------------------------- 3. explicit URI TLS settings respected
@pytest.mark.parametrize("uri,want", [
    ("mongodb://h1:27017,h2:27017/db?replicaSet=rs0&tls=true", CA),        # TLS on a standard URI
    ("mongodb://h1:27017/?ssl=true", CA),                                  # legacy spelling
    ("mongodb://h1:27017/?TLS=True", CA),                                  # case-insensitive
    ("mongodb://h1:27017/?tlsCertificateKeyFile=%2Fc.pem", CA),            # other tls* option implies TLS
    ("mongodb://h1:27017/?tls=true&tlsCAFile=%2Fmy%2Fca.pem", {}),         # URI CA file not overridden
    ("mongodb+srv://c.example.net/?tlsCAFile=%2Fmy%2Fca.pem", {}),
    ("mongodb+srv://c.example.net/?tls=false", {}),                        # explicit off not overridden
    ("mongodb://h1:27017/?ssl=false", {}),
    ("mongodb://u:p%40ss@h1:27017/?tls=true", CA),                          # credentials ignored by parser
])
def test_explicit_uri_tls_settings(uri, want):
    assert M.tls_kwargs(uri) == want


# ---------------------------------------------------------------- 4. env var beats .env
def test_process_env_overrides_dotenv(clean_env, fake_client, tmp_path):
    (tmp_path / ".env").write_text(f"MONGODB_URI={ATLAS}\n", encoding="utf-8")
    os.environ["MONGODB_URI"] = LOCAL
    M.connect(tmp_path / ".env", 30000)
    assert fake_client.call_args.args[0] == LOCAL
    assert "tlsCAFile" not in fake_client.call_args.kwargs


def test_dotenv_used_when_env_unset(clean_env, fake_client, tmp_path):
    (tmp_path / ".env").write_text(f"MONGODB_URI={LOCAL}\n", encoding="utf-8")
    M.connect(tmp_path / ".env", 30000)
    assert fake_client.call_args.args[0] == LOCAL


def test_spider1_kit_wiring_reads_its_root_env(clean_env, fake_client, tmp_path, monkeypatch):
    monkeypatch.setattr(K, "ROOT", tmp_path)  # never load the real .env
    (tmp_path / ".env").write_text(f"MONGODB_URI={ATLAS}\n", encoding="utf-8")
    K.mongo_client()
    fake_client.assert_called_once_with(ATLAS, serverSelectionTimeoutMS=30000, tlsCAFile=certifi.where())
    os.environ["MONGODB_URI"] = LOCAL
    K.mongo_client()
    assert fake_client.call_args == mock.call(LOCAL, serverSelectionTimeoutMS=30000)


def test_conversion_kit_wiring(clean_env, fake_client, tmp_path, monkeypatch):
    import conversion_kit as CK
    monkeypatch.setattr(CK, "ROOT", tmp_path)
    os.environ["MONGODB_URI"] = LOCAL
    CK.mongo_client()
    fake_client.assert_called_once_with(LOCAL, serverSelectionTimeoutMS=30000)


@pytest.mark.parametrize("rel", ["scripts/conversion_kit.py", "scripts/spider1/spider1_kit.py",
                                 "scripts/prototypes/harness.py", "scripts/pilot_local041.py"])
def test_no_unconditional_tlscafile_left(rel):
    # pilot_local041 / harness run work at import time, so they are checked by source only
    src = (ROOT / rel).read_text(encoding="utf-8")
    assert "tlsCAFile" not in src and "certifi" not in src
    assert "connect(ROOT / \".env\"" in src


# ---------------------------------------------------------------- 5. clear errors, no secrets
@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_uri_is_clear(clean_env, fake_client, tmp_path, value):
    if value is not None:
        os.environ["MONGODB_URI"] = value
    with pytest.raises(RuntimeError, match="MONGODB_URI is not set"):
        M.connect(tmp_path / ".env", 30000)
    fake_client.assert_not_called()


@pytest.mark.parametrize("bad", ["postgres://admin:s3cretpw@db:5432", "127.0.0.1:27017", "s3cretpw"])
def test_bad_scheme_is_clear_and_hides_value(clean_env, fake_client, tmp_path, bad):
    os.environ["MONGODB_URI"] = bad
    with pytest.raises(ValueError, match="must start with mongodb:// or mongodb\\+srv://") as ei:
        M.connect(tmp_path / ".env", 30000)
    assert "s3cretpw" not in str(ei.value) and bad not in str(ei.value)
    fake_client.assert_not_called()


def test_malformed_uri_error_hides_credentials(clean_env, tmp_path):
    # unescaped '@' in the password: pymongo rejects it while parsing, before any connection
    bad = "mongodb://admin:p@ss5ecret@127.0.0.1:27017"
    os.environ["MONGODB_URI"] = bad
    with pytest.raises(ValueError, match="invalid MONGODB_URI") as ei:
        M.connect(tmp_path / ".env", 1000)
    msg = str(ei.value)
    assert "ss5ecret" not in msg and bad not in msg
    assert ei.value.__cause__ is None and ei.value.__suppress_context__
