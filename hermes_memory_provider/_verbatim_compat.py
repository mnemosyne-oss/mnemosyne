"""Optional core capability, bundled with each independently installed provider.

Keep both provider copies byte-identical (enforced by contract tests). A fallback
in a new core module would itself be absent on the older cores this supports.
"""


class DisabledVerbatimLedger:
    """Ordinary recall and capture when optional self-echo support is absent."""

    enabled = False

    def reset_session(self, key):
        return None

    def begin(self, key, messages):
        return None

    def release(self, key, messages):
        return None

    def snapshot_for(self, key):
        return None

    def capture(self, key, ticket, beam, raw, **kwargs):
        return beam.remember(**kwargs)


def make_verbatim_ledger():
    """Import only during provider construction, never during CLI discovery."""
    try:
        from mnemosyne.core.verbatim_ledger import VerbatimLedger
    except ImportError:
        return DisabledVerbatimLedger()
    return VerbatimLedger()
