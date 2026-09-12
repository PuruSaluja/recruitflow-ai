"""Unit tests for the pipeline transition rules. No database needed."""

from __future__ import annotations

import pytest

from app.models.enums import ApplicationStatus as S
from app.services.pipeline import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    TransitionError,
    can_transition,
    check_transition,
    next_statuses,
)

LIVE_STATUSES = [status for status in S if status not in TERMINAL_STATUSES]


def test_every_status_has_a_rule() -> None:
    """A status missing from the table would raise KeyError at request time."""
    assert set(ALLOWED_TRANSITIONS) == set(S)


def test_terminal_statuses_are_exactly_the_end_states() -> None:
    assert set(TERMINAL_STATUSES) == {S.HIRED, S.REJECTED, S.WITHDRAWN}


def test_no_rule_points_at_an_unknown_status() -> None:
    for onward in ALLOWED_TRANSITIONS.values():
        assert onward <= set(S)


def test_a_status_never_lists_itself_as_a_move() -> None:
    """Staying put is handled by can_transition, not by the table."""
    for status, onward in ALLOWED_TRANSITIONS.items():
        assert status not in onward


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (S.APPLIED, S.SCREENING),
        (S.SCREENING, S.INTERVIEW),
        (S.INTERVIEW, S.OFFER),
        (S.OFFER, S.HIRED),
    ],
)
def test_the_happy_path_is_allowed(current: S, target: S) -> None:
    assert can_transition(current, target)
    check_transition(current, target)


def test_the_whole_happy_path_runs_end_to_end() -> None:
    status = S.APPLIED
    for nxt in (S.SCREENING, S.INTERVIEW, S.OFFER, S.HIRED):
        check_transition(status, nxt)
        status = nxt
    assert status in TERMINAL_STATUSES


@pytest.mark.parametrize("current", LIVE_STATUSES)
@pytest.mark.parametrize("target", [S.REJECTED, S.WITHDRAWN])
def test_rejection_and_withdrawal_are_available_from_every_live_stage(
    current: S, target: S
) -> None:
    check_transition(current, target)


@pytest.mark.parametrize("status", sorted(S, key=str))
def test_staying_on_the_same_status_is_a_no_op(status: S) -> None:
    """Re-sending the current status must not error, so retries are safe."""
    assert can_transition(status, status)
    check_transition(status, status)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (S.APPLIED, S.INTERVIEW),
        (S.APPLIED, S.OFFER),
        (S.APPLIED, S.HIRED),
        (S.SCREENING, S.OFFER),
        (S.SCREENING, S.HIRED),
        (S.INTERVIEW, S.HIRED),
    ],
)
def test_skipping_a_stage_is_refused(current: S, target: S) -> None:
    assert not can_transition(current, target)
    with pytest.raises(TransitionError):
        check_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (S.SCREENING, S.APPLIED),
        (S.INTERVIEW, S.SCREENING),
        (S.OFFER, S.INTERVIEW),
    ],
)
def test_moving_backwards_is_refused(current: S, target: S) -> None:
    with pytest.raises(TransitionError):
        check_transition(current, target)


@pytest.mark.parametrize("current", sorted(TERMINAL_STATUSES, key=str))
def test_a_finished_application_cannot_move(current: S) -> None:
    assert next_statuses(current) == frozenset()
    for target in S:
        if target == current:
            continue
        with pytest.raises(TransitionError, match="can no longer change status"):
            check_transition(current, target)


def test_the_refusal_message_lists_what_is_allowed_instead() -> None:
    with pytest.raises(TransitionError) as caught:
        check_transition(S.APPLIED, S.HIRED)
    message = str(caught.value)
    assert "from applied to hired" in message
    for expected in ("screening", "rejected", "withdrawn"):
        assert expected in message


def test_the_terminal_message_names_the_status() -> None:
    with pytest.raises(TransitionError, match="hired"):
        check_transition(S.HIRED, S.REJECTED)
