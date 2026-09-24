from typing import Any, List, Optional

from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class Document(SQLModel, table=True):
    """One row per extracted source file -- the source table. 'store'
    separates the default pipeline's results from the professional
    (classify-then-target) pipeline's, matching the two side-by-side JSON
    files this replaces. 'error' is set instead of a PolicyData/
    FieldVerification row when extraction failed for this file."""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("store", "source_file", name="uq_documents_store_source_file"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    store: str
    source_file: str
    extracted_date: str
    extraction_method: Optional[str] = None
    extraction_strategy: Optional[str] = None
    error: Optional[str] = None


class PolicyData(SQLModel, table=True):
    """The extracted policy fields for one Document. Split out from
    Document (1:1) rather than added as columns there, so a failed
    extraction (Document.error set) simply has no PolicyData row instead of
    an all-null one."""

    __tablename__ = "policy_data"

    document_id: int = Field(
        sa_column=Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    )
    policy_number: Optional[str] = None
    insurance_type: Optional[str] = None
    insurance_company: Optional[str] = None
    broker: Optional[str] = None
    coverholder: Optional[str] = None
    insured_name: Optional[str] = None
    insured_address: Optional[str] = None
    period_from: Optional[str] = None
    period_to: Optional[str] = None
    period_from_iso: Optional[str] = None
    period_to_iso: Optional[str] = None
    document_role: Optional[str] = None
    premium_amount: Optional[str] = None
    coverage_limit: Optional[str] = None
    deductible: Optional[str] = None
    key_coverages: Optional[List[Any]] = Field(default=None, sa_column=Column(JSONB))
    exclusions: Optional[List[Any]] = Field(default=None, sa_column=Column(JSONB))
    notes: Optional[str] = None


class FieldVerification(SQLModel, table=True):
    """The grounding audit trail for one field of one Document (1:many).
    'quote' and 'original_value' are only populated for the verification
    statuses that carry them (see insurance_loader.verify_grounded_fields) --
    left null otherwise rather than storing empty strings."""

    __tablename__ = "field_verifications"
    __table_args__ = (
        UniqueConstraint("document_id", "field_name", name="uq_field_verifications_document_field"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(sa_column=Column(Integer, ForeignKey("documents.id", ondelete="CASCADE")))
    field_name: str
    status: str
    quote: Optional[str] = None
    original_value: Optional[str] = None
    flagged: bool = False
