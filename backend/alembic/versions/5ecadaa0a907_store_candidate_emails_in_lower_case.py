"""store candidate emails in lower case

The API now stores every email in lower case, so the unique index refuses one
already on file in other capitals. Emails stored before that are folded here
too, or "Ana@Example.com" on file would still let "ana@example.com" in.

Two rows that fold to the same email are the same person entered twice; the
upgrade stops and names them rather than choosing which to keep. Emails are
folded with Python's str.lower(), as the API folds them, not PostgreSQL's
lower(): the two differ on a few non-ASCII letters (U+0130, for one), and a
row folded one way would not clash with a new one folded the other.

Revision ID: 5ecadaa0a907
Revises: 125593aa4a2f
Create Date: 2026-11-11 00:05:18.496085

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5ecadaa0a907'
down_revision: Union[str, Sequence[str], None] = '125593aa4a2f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT id, email FROM candidates ORDER BY id")).all()
    by_email: dict[str, list[int]] = {}
    for row in rows:
        by_email.setdefault(row.email.lower(), []).append(row.id)
    clashes = {email: ids for email, ids in sorted(by_email.items()) if len(ids) > 1}
    if clashes:
        listed = "; ".join(f"{email}: candidates {ids}" for email, ids in clashes.items())
        raise RuntimeError(
            "These candidates share an email in different capitals; merge or delete "
            f"the duplicates, then upgrade again: {listed}"
        )
    changed = [
        {"id": row.id, "email": row.email.lower()}
        for row in rows
        if row.email != row.email.lower()
    ]
    if changed:
        bind.execute(sa.text("UPDATE candidates SET email = :email WHERE id = :id"), changed)


def downgrade() -> None:
    """Downgrade schema."""
    # The capitals each email was entered with are not kept, so there is
    # nothing to restore; the lower-case emails stay valid as they are.
    pass
