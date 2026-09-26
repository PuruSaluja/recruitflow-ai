"""check match_scores job foreign key per statement again

Undoes 11285e2656ce. Deferring the check closed one deadlock between a scoring
run and a job delete but made another: the delete's cascade could reach a
score the run had rescored while the run waited to lock the job at commit.
A scoring run now holds its job FOR SHARE while it writes (see
``services.matching.score_pool``), so the two take turns and the check's
timing no longer matters. Checked per statement, as it was before.

Revision ID: 0d2e3f968842
Revises: 11285e2656ce
Create Date: 2026-09-26 15:32:32.839193

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '0d2e3f968842'
down_revision: Union[str, Sequence[str], None] = '11285e2656ce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        'ALTER TABLE match_scores ALTER CONSTRAINT match_scores_job_id_fkey NOT DEFERRABLE'
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        'ALTER TABLE match_scores ALTER CONSTRAINT match_scores_job_id_fkey '
        'DEFERRABLE INITIALLY DEFERRED'
    )
