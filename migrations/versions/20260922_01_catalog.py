"""PostgreSQL catalogue metadata.

Revision ID: 20260922_01
"""

from pathlib import Path

from alembic import op

revision = "20260922_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Keep the SQL form as the canonical, inspectable schema and make Alembic the normal runner.
    sql = (Path(__file__).parents[1] / "0001_postgresql_catalog.sql").read_text(encoding="utf-8")
    for statement in sql.split(";"):
        statement = "\n".join(line for line in statement.splitlines() if not line.lstrip().startswith("--"))
        if statement.strip():
            op.execute(statement)


def downgrade() -> None:
    for table in ("index_builds", "image_derivatives", "wine_images", "wine_grapes", "grape_varieties", "wines"):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
