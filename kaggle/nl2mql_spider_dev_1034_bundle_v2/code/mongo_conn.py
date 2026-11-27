"""Shared MongoDB connection helper (Atlas and local MongoDB).

TLS rule -- the certifi CA bundle is passed only when the connection uses TLS:
  * `mongodb+srv` URIs (Atlas): TLS is on by default -> tlsCAFile=certifi.where(), as before
  * `mongodb` URIs with tls=true / ssl=true (or any other tls*/ssl* option): same
  * `mongodb` URIs with no TLS option (e.g. local Docker at 127.0.0.1:27017): no TLS kwarg,
    so pymongo connects in plain text (passing tlsCAFile would itself switch TLS on)
  * an explicit tls=false / ssl=false, or a tlsCAFile already in the URI, is never overridden

MONGODB_URI comes from the process environment, else from .env (load_dotenv does not
override variables already set). The URI is never printed or put in an error message.
"""
import os
from urllib.parse import unquote

_SCHEMES = ("mongodb", "mongodb+srv")


def _split_uri(uri):
    """(scheme, {lower-cased option name: lower-cased value}) -- no DNS lookup, no validation."""
    scheme, sep, rest = uri.partition("://")
    if not sep or scheme.lower() not in _SCHEMES:
        raise ValueError("MONGODB_URI must start with mongodb:// or mongodb+srv:// (value not shown)")
    hosts_and_path = rest.rpartition("@")[2]  # drop user:password@
    query = hosts_and_path.partition("/")[2].partition("?")[2]
    opts = {}
    for part in query.replace(";", "&").split("&"):
        if part:
            k, _, v = part.partition("=")
            opts[unquote(k).lower()] = unquote(v).lower()
    return scheme.lower(), opts


def tls_kwargs(uri):
    """MongoClient kwargs for TLS: {"tlsCAFile": certifi.where()} or {}."""
    scheme, opts = _split_uri(uri)
    explicit = opts.get("tls", opts.get("ssl"))
    if explicit == "false" or "tlscafile" in opts:
        return {}
    implied = any(k.startswith(("tls", "ssl")) for k in opts)
    if scheme == "mongodb+srv" or explicit == "true" or implied:
        import certifi
        return {"tlsCAFile": certifi.where()}
    return {}


def mongo_uri(env_path):
    from dotenv import load_dotenv

    load_dotenv(env_path)
    uri = os.environ.get("MONGODB_URI", "")
    if not uri.strip():
        raise RuntimeError("MONGODB_URI is not set (process environment or .env)")
    return uri


def _redact(msg, uri):
    msg = msg.replace(uri, "<MONGODB_URI>")
    creds = uri.partition("://")[2].rpartition("@")[0]
    if creds:
        msg = msg.replace(creds, "<credentials>")
        for piece in creds.split(":"):
            if piece:
                msg = msg.replace(piece, "<credentials>")
    return msg


def connect(env_path, timeout_ms):
    """MongoClient for MONGODB_URI with the TLS rule above."""
    from pymongo import MongoClient
    from pymongo.errors import ConfigurationError

    uri = mongo_uri(env_path)
    kwargs = tls_kwargs(uri)
    try:
        return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms, **kwargs)
    except ConfigurationError as e:
        raise ValueError(f"invalid MONGODB_URI ({type(e).__name__}): {_redact(str(e), uri)}") from None
