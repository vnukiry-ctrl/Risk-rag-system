import logging
from typing import Dict

from sqlmodel import delete, select

from db import get_session
from models import Document, FieldVerification, PolicyData

logger = logging.getLogger(__name__)

DEFAULT_STORE = "default"
PROFESSIONAL_STORE = "professional"

_POLICY_FIELDS = [
    "policy_number", "insurance_type", "insurance_company", "broker",
    "coverholder", "insured_name", "insured_address", "period_from",
    "period_to", "period_from_iso", "period_to_iso", "document_role",
    "premium_amount", "coverage_limit", "deductible", "key_coverages",
    "exclusions", "notes",
]


def save_documents(documents_db: Dict, store: str = DEFAULT_STORE) -> None:
    """Persist the full documents_db snapshot, replacing any previous rows
    for this store. documents_db is produced as a full replace on every
    /extract (not a stream of discrete events), so a delete-then-insert per
    save matches how it's actually produced. Deleting a store's Document
    rows cascades (ON DELETE CASCADE) to its PolicyData/FieldVerification
    rows automatically."""
    with get_session() as session:
        session.exec(delete(Document).where(Document.store == store))

        for source_file, data in documents_db.items():
            data = dict(data)
            error = data.pop("error", None)
            doc = Document(
                store=store,
                source_file=source_file,
                extracted_date=data.pop("extracted_date", None),
                extraction_method=data.pop("extraction_method", None),
                extraction_strategy=data.pop("extraction_strategy", None),
                error=error,
            )
            session.add(doc)
            session.flush()  # populate doc.id for the FK rows below

            if error is not None:
                continue

            data.pop("source_file", None)
            field_verification = data.pop("field_verification", None) or {}
            flagged_fields = set(data.pop("flagged_fields", None) or [])

            session.add(PolicyData(
                document_id=doc.id,
                **{field: data.pop(field, None) for field in _POLICY_FIELDS},
            ))

            for field_name, verification in field_verification.items():
                session.add(FieldVerification(
                    document_id=doc.id,
                    field_name=field_name,
                    status=verification.get("status"),
                    quote=verification.get("quote"),
                    original_value=verification.get("original_value"),
                    flagged=field_name in flagged_fields,
                ))

        session.commit()
    logger.info("Saved %d document(s) to store '%s'", len(documents_db), store)


def load_documents(store: str = DEFAULT_STORE) -> Dict:
    """Load the last saved documents_db snapshot for this store, rebuilding
    each entry into the same flat dict shape insurance_loader.py produces.
    Empty dict if none exists yet."""
    with get_session() as session:
        docs = session.exec(select(Document).where(Document.store == store)).all()

        documents_db = {}
        for doc in docs:
            if doc.error is not None:
                documents_db[doc.source_file] = {
                    "error": doc.error,
                    "source_file": doc.source_file,
                    "extracted_date": doc.extracted_date,
                }
                continue

            policy = session.exec(
                select(PolicyData).where(PolicyData.document_id == doc.id)
            ).one()
            verifications = session.exec(
                select(FieldVerification).where(FieldVerification.document_id == doc.id)
            ).all()

            field_verification = {}
            flagged_fields = []
            for v in verifications:
                entry = {"status": v.status}
                if v.quote is not None:
                    entry["quote"] = v.quote
                if v.original_value is not None:
                    entry["original_value"] = v.original_value
                field_verification[v.field_name] = entry
                if v.flagged:
                    flagged_fields.append(v.field_name)

            documents_db[doc.source_file] = {
                **{field: getattr(policy, field) for field in _POLICY_FIELDS},
                "extracted_date": doc.extracted_date,
                "source_file": doc.source_file,
                "extraction_method": doc.extraction_method,
                "extraction_strategy": doc.extraction_strategy,
                "field_verification": field_verification,
                "flagged_fields": flagged_fields,
            }

    logger.info("Loaded %d document(s) from store '%s'", len(documents_db), store)
    return documents_db
