"""normalize documents into documents, policy_data, field_verifications

Revision ID: e64c528fbe19
Revises: 69b88f2dba30
Create Date: 2026-09-24 09:41:20.887251

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e64c528fbe19'
down_revision: Union[str, Sequence[str], None] = '69b88f2dba30'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # The prior 'documents' table (single JSONB 'data' column, composite PK)
    # has 0 rows in every known environment, so this replaces it outright
    # rather than migrating a composite PK into a surrogate one in place.
    op.drop_table('documents')

    op.create_table('documents',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('store', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('source_file', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('extracted_date', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('extraction_method', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('extraction_strategy', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('error', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('store', 'source_file', name='uq_documents_store_source_file'),
    )

    op.create_table('policy_data',
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('policy_number', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('insurance_type', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('insurance_company', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('broker', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('coverholder', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('insured_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('insured_address', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('period_from', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('period_to', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('period_from_iso', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('period_to_iso', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('document_role', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('premium_amount', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('coverage_limit', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('deductible', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('key_coverages', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('exclusions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('notes', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('document_id'),
    )

    op.create_table('field_verifications',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('field_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('status', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('quote', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('original_value', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('flagged', sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('document_id', 'field_name', name='uq_field_verifications_document_field'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('field_verifications')
    op.drop_table('policy_data')
    op.drop_table('documents')

    op.create_table('documents',
    sa.Column('store', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('source_file', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.PrimaryKeyConstraint('store', 'source_file'),
    )
