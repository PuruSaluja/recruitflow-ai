"""Legal moves through the hiring pipeline.

The rules live here rather than in the endpoint so they can be unit-tested
without a database, and so the same graph can later drive the UI's "what can I
do next" affordances.

Three deliberate choices:

* **Terminal states stay terminal for ordinary moves.** Once an application is
  hired, rejected or withdrawn, no status change can move it.
* **No backwards moves.** Real recruiters do occasionally walk a candidate back
  a stage, but allowing it silently would make "how long did screening take"
  unanswerable from the data.
* **Reopening is its own action, not a loosened rule.** A rejected or withdrawn
  application can be put back at the first stage through a separate, explicit
  reopen. It reopens the same application rather than starting a new one: the
  stage history records the reopen like any other move, so the finished stint
  keeps its duration instead of being rewritten, and a candidate still holds
  one application per job. A hired application cannot be reopened.
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

#: Statuses no ordinary move can leave. Rejected and withdrawn can still be
#: reopened; see :func:`check_reopen`.
TERMINAL_STATUSES = frozenset(
    status for status, onward in ALLOWED_TRANSITIONS.items() if not onward
)

#: Finished statuses a reopen may start from. Hired is left out on purpose:
#: undoing a hire is not the same as giving a turned-down candidate another look.
REOPENABLE_STATUSES = frozenset({_S.REJECTED, _S.WITHDRAWN})

#: Where a reopened application lands: back at the start of the pipeline.
REOPEN_TARGET = _S.APPLIED


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
        message = f"This application is {current.value} and can no longer change status."
        if current in REOPENABLE_STATUSES:
            message += " Reopen it to put it back into the pipeline."
        raise TransitionError(message)
    allowed = ", ".join(sorted(status.value for status in onward))
    raise TransitionError(
        f"Cannot move an application from {current.value} to {target.value}. "
        f"Allowed from {current.value}: {allowed}."
    )


def check_reopen(current: ApplicationStatus) -> None:
    """Raise :class:`TransitionError` unless an application at ``current`` may be reopened."""
    if current in REOPENABLE_STATUSES:
        return

    allowed = " or ".join(sorted(status.value for status in REOPENABLE_STATUSES))
    if current in TERMINAL_STATUSES:
        raise TransitionError(
            f"A {current.value} application cannot be reopened; only {allowed} ones can."
        )
    raise TransitionError(
        f"This application is at {current.value} and has not finished, "
        "so there is nothing to reopen."
    )
