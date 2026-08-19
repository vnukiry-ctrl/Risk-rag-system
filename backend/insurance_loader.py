import os
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class InsuranceDocumentParserLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = "openai/gpt-oss-120b"
        #self.model = "mixtral-8x7b-32768" 
    
    def extract_metadata_with_llm(self, text: str, filename: str) -> Dict:
        """Use Groq to intelligently extract metadata from insurance document"""
        
        text_sample = text[:8000]
        
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
            message = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1500,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.choices[0].message.content.strip()
            
            # Remove markdown if present
            response_text = re.sub(r'```json\n?', '', response_text)
            response_text = re.sub(r'```\n?', '', response_text)
            response_text = response_text.strip()
            
            # Validate JSON before parsing
            if not response_text.startswith('{'):
                print(f"  Warning: Response doesn't start with JSON")
                return {
                    "error": "Invalid LLM response format",
                    "source_file": filename,
                    "extracted_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
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
            metadata["extraction_method"] = "Groq LLM (openai/gpt-oss-120b)"
            
            return metadata
            
        except json.JSONDecodeError as e:
            print(f"  Error parsing JSON: {str(e)}")
            print(f"  Response was: {response_text[:200]}")
            return {
                "error": f"JSON parse error: {str(e)}",
                "source_file": filename,
                "extracted_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"  Error calling Groq API: {str(e)}")
            return {
                "error": str(e),
                "source_file": filename,
                "extracted_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def chunk_by_sections(self, text: str) -> List[Dict]:
        """Chunk document by SECTION headers"""
        chunks = []
        sections = re.split(r'(SECTION\s+[\d\w]+[:\s]+[^\n]+)', text)
        current_section_title = None
        
        for i, part in enumerate(sections):
            if re.match(r'SECTION\s+[\d\w]+', part, re.IGNORECASE):
                current_section_title = part.strip()
            elif part.strip() and current_section_title:
                chunk = {
                    "content": part.strip(),
                    "section": current_section_title,
                    "type": "child"
                }
                chunks.append(chunk)
        
        return chunks
    
    def parse_document(self, text: str, filename: str) -> Tuple[Dict, str, List[Dict]]:
        """Parse insurance document using Groq LLM"""
        
        metadata = self.extract_metadata_with_llm(text, filename)
        parent_content = text
        child_chunks = self.chunk_by_sections(text)
        
        for chunk in child_chunks:
            chunk["metadata"] = metadata
            chunk["policy_number"] = metadata.get("policy_number")
            chunk["insurance_company"] = metadata.get("insurance_company")
            chunk["insurance_type"] = metadata.get("insurance_type")
        
        return metadata, parent_content, child_chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF (better CID font handling)"""
    try:
        import fitz
        
        text = ""
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        print(f"      (Reading {num_pages} pages...)")
        
        for page_num in range(num_pages):
            page = doc[page_num]
            
            # Extract text
            extracted = page.get_text()
            if extracted:
                text += extracted
            
            # Also try to extract text with different methods for CID PDFs
            blocks = page.get_text("blocks")
            for block in blocks:
                if isinstance(block, dict) and "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if "text" in span:
                                text += span["text"] + " "
            
            if (page_num + 1) % 5 == 0:
                print(f"      (Processed {page_num + 1}/{num_pages} pages...)")
        
        doc.close()
        return text
    except Exception as e:
        print(f"  Error reading PDF: {str(e)}")
        return ""
def load_insurance_documents(data_folder: str = "./data") -> Dict:
    """Load and parse all insurance documents using Groq LLM"""
    
    parser = InsuranceDocumentParserLLM()
    results = {
        "metadata": [],
        "parent_chunks": [],
        "child_chunks": []
    }
    
    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
        return results
    
    files = [f for f in os.listdir(data_folder) if f.endswith('.txt') or f.endswith('.pdf')]
    
    if not files:
        print(f"No TXT or PDF files found in {data_folder}")
        return results
    
    for filename in files:
        filepath = os.path.join(data_folder, filename)
        print(f"\nParsing: {filename}")
        
        try:
            # Extract text
            if filename.endswith('.pdf'):
                print("  PDF detected - extracting text...")
                text = extract_text_from_pdf(filepath)
                if not text.strip():
                    print("  Could not extract text from PDF")
                    continue
            else:
                print("  TXT file detected - reading...")
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Parse with Groq LLM
            print("  Analyzing with Groq...")
            metadata, parent, children = parser.parse_document(text, filename)
            
            results["metadata"].append(metadata)
            
            results["parent_chunks"].append({
                "content": parent,
                "metadata": metadata,
                "source_file": filename,
                "type": "parent"
            })
            
            for child in children:
                child["source_file"] = filename
                results["child_chunks"].append(child)
            
            # Print results
            print(f"  ✓ Policy #: {metadata.get('policy_number', 'N/A')}")
            print(f"  ✓ Type: {metadata.get('insurance_type', 'N/A')}")
            print(f"  ✓ Insurer: {metadata.get('insurance_company', 'N/A')}")
            print(f"  ✓ Broker: {metadata.get('broker', 'N/A')}")
            print(f"  ✓ Coverholder: {metadata.get('coverholder', 'N/A')}")
            print(f"  ✓ Insured: {metadata.get('insured_name', 'N/A')}")
            print(f"  ✓ Period: {metadata.get('period_from', 'N/A')} to {metadata.get('period_to', 'N/A')}")
            print(f"  ✓ Premium: ${metadata.get('premium_amount', 'N/A')}")
            print(f"  ✓ Sections: {len(children)}")
            
        except Exception as e:
            print(f"  Error parsing {filename}: {str(e)}")
    
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
    print(f"Child Chunks: {len(results['child_chunks'])}")
    
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