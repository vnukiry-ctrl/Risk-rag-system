import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, Tuple, Optional
from pypdf import PdfReader
from llm_client import llm_complete, LLM_PROVIDER, MODEL
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FOLDER = os.path.join(BASE_DIR, "data")

_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _error_record(filename: str, error: str) -> Dict:
    """Build the standard error-metadata shape used whenever a document can't be processed."""
    return {
        "error": error,
        "source_file": filename,
        "extracted_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def normalize_text(text: str) -> str:
    """Collapse PDF layout whitespace noise without touching real content.

    PyMuPDF preserves the source page layout, so extracted text is full of
    lines that are empty except for spaces (column/table artifacts). Left
    in place, that noise eats into the 8000-char sample sent to the LLM and
    dilutes the chunks used for embedding search.
    """
    # PyMuPDF emits U+FFFD for glyphs it can't map (seen with custom/subset
    # fonts encoding smart quotes and dashes) -- drop rather than keep, since
    # a missing apostrophe ("Lloyds Underwriters") reads better than a
    # visible replacement-character glitch in extracted metadata and answers.
    text = text.replace("�", "")
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


class InsuranceDocumentParserLLM:
    def extract_metadata_with_llm(self, text: str, filename: str) -> Dict:
        """Use Groq to intelligently extract metadata from insurance document"""
        
        # DECISION (UNIVERSAL, value needs re-checking per project): hard
        # truncation before the fields-of-interest are guaranteed to appear.
        # A doc whose key fields sit past char 8000 (long riders/addenda)
        # silently loses them here -- verify against your actual document
        # lengths rather than reusing 8000.
        text_sample = text[:8000]

        # DECISION (DOMAIN-SPECIFIC): entire prompt below -- field list,
        # few-shot hints (policy-number formats, insurer/broker phrasing) --
        # is insurance vocabulary. Rewrite completely for a new domain; the
        # pattern worth keeping is "explicit field list + null-if-missing +
        # return-only-JSON", not the specific fields or hints.
        prompt = f"""You are an insurance document expert. Extract ONLY valid JSON from this insurance document. Return the JSON object only, no other text.

DOCUMENT FILENAME: {filename}
DOCUMENT TEXT:
{text_sample}

Extract these fields. Return ONLY this JSON structure (no markdown, no explanation):
{{
    "policy_number": "Policy number/ID (look for 'Policy No', 'Policy #', 'ALCOG', 'BW', 'XLKR' formats) or null",
    "insurance_type": "Type of insurance (e.g., Personal Accident, General Liability, Kidnap & Ransom) or null",
    "insurance_company": "Name of the INSURER/Insurance Company/Underwriter/Syndicate or null",
    "broker": "Name of BROKER/Agent who arranged the policy or null",
    "coverholder": "Name of COVERHOLDER (may be same as broker) or null",
    "insured_name": "Name of policyholder/insured party or null",
    "insured_address": "Address of insured or null",
    "period_from": "Start date of policy or null",
    "period_to": "End/expiration date of policy or null",
    "premium_amount": "Annual premium amount (numbers only, no currency) or null",
    "coverage_limit": "Main coverage limit amount or null",
    "deductible": "Deductible amount or null",
    "key_coverages": ["coverage1", "coverage2", "coverage3"],
    "exclusions": ["exclusion1", "exclusion2"],
    "notes": "Any other important details or null"
}}

CRITICAL RULES:
- Return ONLY the JSON object, absolutely nothing else
- Use null for missing fields (not "N/A" or "Unknown")
- For POLICY NUMBER: Look for "Policy No", "Policy #", "ALCOG", "BW", "XLKR", "Assigned:", policy numbers in boxes/headers
- For INSURANCE COMPANY: Look for "Insurance Company", "Underwriter", "Insured with", "Effected with", "Lloyd's", "AXA", "Aviva", "Allianz", or similar
- For BROKER: Look for "Broker", "Agent", "Through", "Arranged by", "Via", or company names that arrange insurance
- For COVERHOLDER: Look for "Lloyd's Approved Coverholder", "Coverholder", "Administrator"
- Be accurate - only extract text that actually appears in the document
- Return valid JSON that can be parsed
- For amounts, return only numbers (e.g., "25000" not "$25,000")"""

        try:
            response_text = llm_complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1500,
            ).strip()
            
            # Remove markdown if present
            response_text = re.sub(r'```json\n?', '', response_text)
            response_text = re.sub(r'```\n?', '', response_text)
            response_text = response_text.strip()
            
            # Validate JSON before parsing
            if not response_text.startswith('{'):
                logger.warning("%s: LLM response doesn't start with JSON", filename)
                return _error_record(filename, "Invalid LLM response format")
            
            # Parse JSON
            metadata = json.loads(response_text)
            
            # Ensure all expected fields exist
            default_fields = {
                "policy_number": None,
                "insurance_type": None,
                "insurance_company": None,
                "broker": None,
                "coverholder": None,
                "insured_name": None,
                "insured_address": None,
                "period_from": None,
                "period_to": None,
                "premium_amount": None,
                "coverage_limit": None,
                "deductible": None,
                "key_coverages": [],
                "exclusions": [],
                "notes": None
            }
            
            # Merge with defaults
            for key in default_fields:
                if key not in metadata:
                    metadata[key] = default_fields[key]
            
            # Add metadata
            metadata["extracted_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata["source_file"] = filename
            metadata["extraction_method"] = f"{LLM_PROVIDER} LLM ({MODEL})"
            
            return metadata
            
        except json.JSONDecodeError as e:
            logger.error("%s: JSON parse error: %s | response was: %s", filename, e, response_text[:200])
            return _error_record(filename, f"JSON parse error: {str(e)}")
        except Exception as e:
            logger.error("%s: LLM API call failed", filename, exc_info=True)
            return _error_record(filename, str(e))
    
    def parse_document(self, text: str, filename: str) -> Tuple[Dict, str]:
        """Parse insurance document using Groq LLM"""
        metadata = self.extract_metadata_with_llm(text, filename)
        return metadata, text


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF (better CID font handling)"""
    try:
        import fitz
        
        text = ""
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        logger.debug("Reading %d pages from %s", num_pages, pdf_path)

        for page_num in range(num_pages):
            page = doc[page_num]
            extracted = page.get_text()
            if extracted:
                text += extracted

        doc.close()
        return text
    except Exception as e:
        logger.error("Failed to read PDF %s", pdf_path, exc_info=True)
        return ""
def load_insurance_documents(data_folder: Optional[str] = None) -> Dict:
    """Load and parse all insurance documents using Groq LLM."""
    if data_folder is None:
        data_folder = DEFAULT_DATA_FOLDER
    elif not os.path.isabs(data_folder):
        data_folder = os.path.join(BASE_DIR, data_folder)

    parser = InsuranceDocumentParserLLM()
    results = {
        "metadata": [],
        "parent_chunks": []
    }

    if not os.path.exists(data_folder):
        logger.warning("Data folder not found: %s", data_folder)
        return results

    files = [f for f in os.listdir(data_folder) if f.endswith('.txt') or f.endswith('.pdf')]

    if not files:
        logger.warning("No TXT or PDF files found in %s", data_folder)
        return results

    logger.info("Parsing %d documents from %s", len(files), data_folder)

    for filename in files:
        filepath = os.path.join(data_folder, filename)

        try:
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
                if not text.strip():
                    logger.warning("%s: no text could be extracted from PDF", filename)
                    results["metadata"].append(
                        _error_record(filename, "No text could be extracted from PDF")
                    )
                    continue
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()

            text = normalize_text(text)

            metadata, parent = parser.parse_document(text, filename)

            results["metadata"].append(metadata)

            results["parent_chunks"].append({
                "content": parent,
                "metadata": metadata,
                "source_file": filename,
                "type": "parent"
            })

            if "error" in metadata:
                logger.warning("%s: metadata extraction failed: %s", filename, metadata["error"])
            else:
                logger.info(
                    "Extracted %s: policy=%s type=%s insurer=%s",
                    filename,
                    metadata.get('policy_number', 'N/A'),
                    metadata.get('insurance_type', 'N/A'),
                    metadata.get('insurance_company', 'N/A'),
                )

        except Exception as e:
            logger.error("Failed to parse %s", filename, exc_info=True)
            results["metadata"].append(_error_record(filename, str(e)))

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("INSURANCE DOCUMENT PARSER - GROQ LLM VERSION")
    print("Using Groq for intelligent metadata extraction")
    print("=" * 70)
    
    results = load_insurance_documents()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nPolicies Loaded: {len(results['metadata'])}")
    print(f"Parent Chunks: {len(results['parent_chunks'])}")
    
    if len(results['metadata']) > 0:
        print("\n" + "=" * 70)
        print("EXTRACTED METADATA")
        print("=" * 70)
        
        for meta in results['metadata']:
            print(f"\n{'='*70}")
            print(f"File: {meta.get('source_file', 'Unknown')}")
            print(f"{'='*70}")
            print(f"  Policy #: {meta.get('policy_number', 'N/A')}")
            print(f"  Type: {meta.get('insurance_type', 'N/A')}")
            print(f"  Insurer: {meta.get('insurance_company', 'N/A')}")
            print(f"  Broker: {meta.get('broker', 'N/A')}")
            print(f"  Coverholder: {meta.get('coverholder', 'N/A')}")
            print(f"  Insured: {meta.get('insured_name', 'N/A')}")
            print(f"  Address: {meta.get('insured_address', 'N/A')}")
            print(f"  Period: {meta.get('period_from', 'N/A')} to {meta.get('period_to', 'N/A')}")
            print(f"  Premium: ${meta.get('premium_amount', 'N/A')}")
            print(f"  Coverage Limit: {meta.get('coverage_limit', 'N/A')}")
            print(f"  Deductible: {meta.get('deductible', 'N/A')}")
            if meta.get('key_coverages'):
                print(f"  Coverages: {', '.join(meta.get('key_coverages', []))}")
            if meta.get('exclusions'):
                print(f"  Exclusions: {', '.join(meta.get('exclusions', []))}")
            print(f"  Method: {meta.get('extraction_method', 'N/A')}")
    
    print("\n" + "=" * 70)