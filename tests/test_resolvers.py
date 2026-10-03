"""Registry semantics for mnemosyne.core.resolvers.

These cover the contract independent of any filesystem: scheme dispatch,
registration lifecycle, the built-in default, and the promise that a failing
resolver degrades to None rather than raising into a caller.

Blob-reading behaviour lives in tests/test_blob_resolver.py.
"""

import pytest

from mnemosyne.core.resolvers import (
    BlobResolver,
    CallableContentResolver,
    ResolvedMeta,
    clear_content_resolvers,
    get_resolver,
    resolve_head,
    resolve_open,
    resolve_presign,
    scheme_of,
    set_content_resolver,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Belt and braces: conftest already clears this, but these tests register
    aggressively and should not depend on the global fixture's ordering."""
    clear_content_resolvers()
    yield
    clear_content_resolvers()


def _archive_resolver(**kwargs):
    return CallableContentResolver(name="fake-archive", schemes=frozenset({"archive"}), **kwargs)


# --- scheme_of --------------------------------------------------------------


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("blob://sha256/abc", "blob"),
        ("archive://host/thing", "archive"),
        ("HTTPS://Example.com/x", "https"),
        ("s3+custom://bucket/key", "s3+custom"),
        ("", None),
        ("noscheme", None),
        ("://missing", None),
        ("/plain/path", None),
        (None, None),
    ],
)
def test_scheme_of(uri, expected):
    assert scheme_of(uri) == expected


# --- registration lifecycle -------------------------------------------------


def test_get_resolver_returns_none_for_unhandled_scheme():
    """The supported absence path: no resolver, no error, no exception."""
    assert get_resolver("archive") is None
    assert get_resolver("archive://anything") is None
    assert resolve_head("archive://anything") is None
    assert resolve_open("archive://anything") is None
    assert resolve_presign("archive://anything") is None


def test_set_and_get_by_scheme_and_by_uri():
    r = _archive_resolver()
    set_content_resolver(r)
    assert get_resolver("archive") is r
    assert get_resolver("archive://host/thing") is r


def test_schemes_override_claims_extra_schemes():
    r = _archive_resolver()
    set_content_resolver(r, schemes=["archive", "vault"])
    assert get_resolver("vault://x") is r


def test_last_registration_wins():
    first = _archive_resolver()
    second = _archive_resolver()
    set_content_resolver(first)
    set_content_resolver(second)
    assert get_resolver("archive") is second


def test_set_none_unregisters_named_schemes():
    r = _archive_resolver()
    set_content_resolver(r)
    set_content_resolver(None, schemes=["archive"])
    assert get_resolver("archive") is None


def test_set_none_without_schemes_is_an_error():
    """Silently clearing everything would be a nasty way to spell 'unregister'."""
    with pytest.raises(ValueError):
        set_content_resolver(None)


# --- the built-in default ---------------------------------------------------


def test_blob_resolver_is_available_with_zero_configuration():
    """The whole point of the two-tier registry: nobody has to register
    BlobResolver for the blob store to become readable."""
    assert isinstance(get_resolver("blob"), BlobResolver)
    assert isinstance(get_resolver("blob://sha256/" + "a" * 64), BlobResolver)


def test_explicit_registration_overrides_the_builtin():
    custom = CallableContentResolver(name="custom-blob", schemes=frozenset({"blob"}))
    set_content_resolver(custom)
    assert get_resolver("blob") is custom


def test_clear_restores_builtin_rather_than_removing_blob_access():
    """conftest calls clear_content_resolvers() before every test. If clearing
    dropped the built-in, that reset would silently disable blob reads for the
    entire suite -- so this asserts the reset is safe to run."""
    custom = CallableContentResolver(name="custom-blob", schemes=frozenset({"blob"}))
    set_content_resolver(custom)
    assert get_resolver("blob") is custom

    clear_content_resolvers()

    restored = get_resolver("blob")
    assert restored is not custom
    assert isinstance(restored, BlobResolver)


# --- failure degrades to None ----------------------------------------------


def test_wrappers_swallow_resolver_exceptions():
    """A resolver talking to a network archive will raise eventually. No caller
    of recall or doctor should have to care. Mirrors call_host_llm."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("archive unreachable")

    set_content_resolver(
        _archive_resolver(head_func=boom, open_func=boom, presign_func=boom)
    )

    assert resolve_head("archive://x") is None
    assert resolve_open("archive://x") is None
    assert resolve_presign("archive://x") is None


def test_callable_resolver_methods_default_to_none():
    """A resolver that only knows how to head is a legal resolver."""
    set_content_resolver(
        _archive_resolver(head_func=lambda uri: ResolvedMeta(exists=True))
    )
    assert resolve_head("archive://x") == ResolvedMeta(exists=True)
    assert resolve_open("archive://x") is None
    assert resolve_presign("archive://x") is None


def test_presign_ttl_is_forwarded():
    seen = {}

    def presign(uri, *, ttl_s):
        seen["uri"] = uri
        seen["ttl_s"] = ttl_s
        return "https://example.invalid/signed"

    set_content_resolver(_archive_resolver(presign_func=presign))
    assert resolve_presign("archive://x", ttl_s=60) == "https://example.invalid/signed"
    assert seen == {"uri": "archive://x", "ttl_s": 60}


# --- cross-test isolation ---------------------------------------------------
# These two run in definition order within the file and together assert that
# the autouse reset actually works.


def test_isolation_a_registers_a_resolver():
    set_content_resolver(
        CallableContentResolver(name="leaky", schemes=frozenset({"leak"}))
    )
    assert get_resolver("leak") is not None


def test_isolation_b_does_not_see_the_previous_registration():
    assert get_resolver("leak") is None
