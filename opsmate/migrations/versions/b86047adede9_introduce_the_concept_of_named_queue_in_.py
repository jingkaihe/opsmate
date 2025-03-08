"""introduce the concept of named-queue in db queue

Revision ID: b86047adede9
Revises: 79fe7d287ba8
Create Date: 2025-03-06 12:33:48.109644

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "b86047adede9"
down_revision: Union[str, None] = "79fe7d287ba8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "taskitem",
        sa.Column(
            "queue_name",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("taskitem", "queue_name")
