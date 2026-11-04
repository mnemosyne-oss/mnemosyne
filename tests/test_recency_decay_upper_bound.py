"""`_recency_decay` must never exceed 1.0 — decay is a multiplier, not a bonus.

A stored timestamp can never be in the future, so the factor is bounded above by 1.0. It can
nonetheless *compute* above 1.0: the producer contract notes that pre-round-4 producers stamped
naive LOCAL wall time while readers treat naive values as UTC. On a non-UTC host those legacy rows
land `utcoffset` hours in the future — measured 1.0549 at UTC+9, i.e. a 5.5% bonus on the recency
arm for a row that is simply misdated. Where decay is blended as
`score = base_score * (rc_share + (1 - rc_share) * decay)` the bonus survives into the final score,
so a row can outrank a correct one purely because of the skew.

These tests pin the bound without requiring a non-UTC host: the condition is constructed
explicitly rather than by manipulating TZ.
"""
from datetime import datetime, timedelta, timezone

import pytest

from mnemosyne.core.beam import _recency_decay


def _naive_utc(dt):
    """Render as the naive-UTC form every producer writes."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat()


def test_present_timestamp_is_at_most_one():
    assert _recency_decay(_naive_utc(datetime.now(timezone.utc))) <= 1.0


def test_future_timestamp_is_clamped():
    """A timestamp ahead of now must not produce a bonus."""
    future = datetime.now(timezone.utc) + timedelta(days=365)
    assert _recency_decay(_naive_utc(future)) == 1.0


@pytest.mark.parametrize("offset_hours", [1, 5, 9, 14])
def test_legacy_local_naive_row_is_clamped(offset_hours):
    """A row stamped local wall-clock on a UTC+N host, then read as UTC, looks N hours ahead."""
    skewed = datetime.now(timezone.utc) + timedelta(hours=offset_hours)
    value = _recency_decay(_naive_utc(skewed))
    assert value == 1.0, (
        f"a row misdated +{offset_hours}h must saturate at exactly 1.0, not merely stay under it; "
        f"got {value:.6f} (before the clamp this was exp(+{offset_hours}/168) > 1; a regression "
        "returning the 0.5 fallback or capping below 1.0 would also pass a <= assertion)"
    )


def test_overflow_sized_future_timestamp_still_saturates():
    """A far-future timestamp must saturate at 1.0, not fall through to the 0.5 handler.

    ``math.exp`` raises OverflowError once the exponent exceeds ~709, so clamping the *result*
    is too late: 20 years ahead is 175,200 h, i.e. exp(1043) on the default halflife. Clamping
    the age saturates before the exponent is ever computed.
    """
    far_future = datetime.now(timezone.utc) + timedelta(days=365 * 20)
    assert _recency_decay(_naive_utc(far_future)) == 1.0
    # ...and it holds for a short halflife too, where the exponent is even larger.
    assert _recency_decay(_naive_utc(far_future), halflife_hours=1.0) == 1.0


def test_decay_is_monotone_non_increasing_and_bounded():
    ages = [0, 1, 6, 24, 72, 168, 720, 8760]
    values = [_recency_decay(_naive_utc(datetime.now(timezone.utc) - timedelta(hours=h))) for h in ages]
    assert all(0.0 <= v <= 1.0 for v in values), values
    for older, newer in zip(values[1:], values[:-1]):
        assert older <= newer + 1e-12, f"decay increased with age: {values}"


def test_ordinary_curve_unchanged():
    """The clamp must not disturb the ordinary curve.

    Note the implementation is e-folding, not a true half-life: exp(-age/halflife) is 1/e ~= 0.368
    at age == halflife, while the docstring says "~0.5 = one halflife old". Pinning the actual
    behaviour so a clamp can never quietly change it (changing the formula itself would move every
    stored score, so this test deliberately does not ask for that).
    """
    one_halflife = _recency_decay(
        _naive_utc(datetime.now(timezone.utc) - timedelta(hours=168)), halflife_hours=168
    )
    assert 0.36 <= one_halflife <= 0.38, one_halflife


def test_unknown_timestamp_is_still_neutral():
    assert _recency_decay("") == 0.5
    assert _recency_decay("not-a-timestamp") == 0.5
