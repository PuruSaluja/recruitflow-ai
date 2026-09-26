"""check match_scores job foreign key at commit

A scoring run inserts its new rows 1,000 to a statement, and each statement's
foreign key check took FOR KEY SHARE on the job, so a run held that lock from
its first batch while a job delete's last statement waited for it. A later
batch waiting on a row the delete had removed then deadlocked with it. Checked
at commit, the lock is taken after the run's last insert. See ``MatchScore``.

``ALTER CONSTRAINT`` rather than dropping and recreating it: the name stays
the same, and the existing rows are not checked again.

Revision ID: 11285e2656ce
Revises: 6bc37e0ed601
Create Date: 2026-09-26 14:22:00.413581

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '11285e2656ce'
down_revision: Union[str, Sequence[str], None] = '6bc37e0ed601'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        'ALTER TABLE match_scores ALTER CONSTRAINT match_scores_job_id_fkey '
        'DEFERRABLE INITIALLY DEFERRED'
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        'ALTER TABLE match_scores ALTER CONSTRAINT match_scores_job_id_fkey NOT DEFERRABLE'
    )
