"""store enum values not names

By default SQLAlchemy uses Python enum member *names* as PostgreSQL enum
labels, so the database held ``FULL_TIME`` while the API serialised
``full_time``. A raw SQL query and an API response therefore disagreed about
the same row. The models now pass ``values_callable``; this revision brings
the existing enum types into line.

``ALTER TYPE ... RENAME VALUE`` (PostgreSQL 10+) preserves every dependent
column, so there is no table rewrite and no data migration. None of these
columns carry a server-side default, so nothing else needs re-pointing.

Revision ID: 33d350d8d3e2
Revises: 7aaff9b12dde
Create Date: 2026-09-08 05:40:54.003173

"""

from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "33d350d8d3e2"
down_revision: Union[str, Sequence[str], None] = "7aaff9b12dde"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: enum type name -> [(python member name, python member value), ...]
ENUM_LABELS: dict[str, list[tuple[str, str]]] = {
    "employment_type": [
        ("FULL_TIME", "full_time"),
        ("PART_TIME", "part_time"),
        ("CONTRACT", "contract"),
        ("INTERNSHIP", "internship"),
    ],
    "seniority_level": [
        ("INTERN", "intern"),
        ("JUNIOR", "junior"),
        ("MID", "mid"),
        ("SENIOR", "senior"),
        ("STAFF", "staff"),
        ("PRINCIPAL", "principal"),
    ],
    "education_level": [
        ("HIGH_SCHOOL", "high_school"),
        ("ASSOCIATE", "associate"),
        ("BACHELOR", "bachelor"),
        ("MASTER", "master"),
        ("DOCTORATE", "doctorate"),
    ],
    "job_status": [
        ("DRAFT", "draft"),
        ("OPEN", "open"),
        ("PAUSED", "paused"),
        ("CLOSED", "closed"),
    ],
    "application_status": [
        ("APPLIED", "applied"),
        ("SCREENING", "screening"),
        ("INTERVIEW", "interview"),
        ("OFFER", "offer"),
        ("HIRED", "hired"),
        ("REJECTED", "rejected"),
        ("WITHDRAWN", "withdrawn"),
    ],
}


def _rename(type_name: str, old: str, new: str) -> None:
    op.execute(f"ALTER TYPE {type_name} RENAME VALUE '{old}' TO '{new}'")


def upgrade() -> None:
    """Rename every enum label from the member name to the member value."""
    for type_name, labels in ENUM_LABELS.items():
        for name, value in labels:
            if name != value:
                _rename(type_name, name, value)


def downgrade() -> None:
    """Restore member names as the enum labels."""
    for type_name, labels in ENUM_LABELS.items():
        for name, value in labels:
            if name != value:
                _rename(type_name, value, name)
