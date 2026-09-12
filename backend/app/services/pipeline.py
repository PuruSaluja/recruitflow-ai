"""Legal moves through the hiring pipeline.

The rules live here rather than in the endpoint so they can be unit-tested
without a database, and so the same graph can later drive the UI's "what can I
do next" affordances.

Two deliberate choices:

* **Terminal states stay terminal.** Once an application is hired, rejected or
  withdrawn it cannot move again. Re-opening a candidate is a new application,
  which keeps the pipeline history honest rather than rewriting it in place.
* **No backwards moves.** Real recruiters do occasionally walk a candidate back
  a stage, but allowing it silently would make "how long did screening take"
  unanswerable from the data. If that need turns up, it should arrive as an
  explicit reopen action with its own audit trail, not as a loosened rule here.
"""

from __future__ import annotations

from app.models.enums import ApplicationStatus

_S = ApplicationStatus

#: Where each status may go next. Rejection and withdrawal stay available from
#: every live stage, because either can happen at any point in a real process.
ALLOWED_TRANSITIONS: dict[ApplicationStatus, frozenset[ApplicationStatus]] = {
    _S.APPLIED: frozenset({_S.SCREENING, _S.REJECTED, _S.WITHDRAWN}),
    _S.SCREENING: frozenset({_S.INTERVIEW, _S.REJECTED, _S.WITHDRAWN}),
    _S.INTERVIEW: frozenset({_S.OFFER, _S.REJECTED, _S.WITHDRAWN}),
    _S.OFFER: frozenset({_S.HIRED, _S.REJECTED, _S.WITHDRAWN}),
    _S.HIRED: frozenset(),
    _S.REJECTED: frozenset(),
    _S.WITHDRAWN: frozenset(),
}

#: Statuses an application can never leave.
TERMINAL_STATUSES = frozenset(
    status for status, onward in ALLOWED_TRANSITIONS.items() if not onward
)


class TransitionError(ValueError):
    """Raised when a status change is not a legal move.

    The endpoint surfaces the message verbatim, so it is written for the
    recruiter looking at the pipeline, not for a log reader.
    """


def next_statuses(current: ApplicationStatus) -> frozenset[ApplicationStatus]:
    """Statuses reachable from ``current`` in one move."""
    return ALLOWED_TRANSITIONS[current]


def can_transition(current: ApplicationStatus, target: ApplicationStatus) -> bool:
    """Whether ``current`` may move to ``target``.

    Staying put counts as allowed: re-sending the current status is a no-op, so
    a client that retries a request does not get an error for it.
    """
    return target == current or target in ALLOWED_TRANSITIONS[current]


def check_transition(current: ApplicationStatus, target: ApplicationStatus) -> None:
    """Raise :class:`TransitionError` unless the move is legal."""
    if can_transition(current, target):
        return

    onward = ALLOWED_TRANSITIONS[current]
    if not onward:
        raise TransitionError(
            f"This application is {current.value} and can no longer change status."
        )
    allowed = ", ".join(sorted(status.value for status in onward))
    raise TransitionError(
        f"Cannot move an application from {current.value} to {target.value}. "
        f"Allowed from {current.value}: {allowed}."
    )
