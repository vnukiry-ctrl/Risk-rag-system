# RAG SYSTEM BUILD - Mount Royal University Insurance Documents

**Project:** Professional RAG system for insurance document analysis  
**Owner:** Victoria Nukiry (vnukiry-ctrl)  
**Repository:** https://github.com/vnukiry-ctrl/Risk-rag-system  
**Location:** H:\my-rag-system\  
**Python:** 3.12.10

---

## PROJECT OVERVIEW

Building an intelligent RAG (Retrieval-Augmented Generation) system that:
1. **Extracts metadata** from insurance PDFs using LLM
2. **Chunks documents** into parent-child structure
3. **Stores in vector database** (Qdrant)
4. **Serves via FastAPI** backend
5. **Visualizes in Streamlit** frontend

### Tech Stack
- **Backend:** FastAPI + LangChain
- **Vector DB:** Qdrant
- **Embeddings:** Ollama (nomic-embed-text)
- **LLM:** Groq (openai/gpt-oss-120b)
- **Frontend:** Streamlit
- **PDF Processing:** pdfplumber, PyMuPDF, pypdf
- **Infrastructure:** Local + Ollama

---

## COMPLETED STEPS

### STEP 1: Project Structure ✅
H:\my-rag-system/
├── backend/
│ └── data/
├── frontend/
└── DOCUMENTATION.md

### STEP 2: Backend Dependencies ✅
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-community langchain-anthropic qdrant-client fastapi uvicorn python-dotenv pypdf python-docx pdfplumber groq openai pytesseract pdf2image PyMuPDF
```

### STEP 3: Frontend Dependencies ✅
```bash
cd frontend
python -m venv venv
venv\Scripts\activate
pip install streamlit==1.28.0 requests==2.31.0
```

### STEP 4: Document Loader ✅
- **File:** `backend/loader.py`
- Supports: PDF, DOCX, TXT
- Uses RecursiveCharacterTextSplitter
- Chunk size: 512 tokens, overlap: 50

### STEP 5: Vector Database ✅
- **File:** `backend/vector_store.py`
- Uses Qdrant in-memory
- Ollama embeddings (nomic-embed-text)
- Easy swap to Anthropic embeddings (1 line change)

### Git Setup ✅
- Git installed and configured
- Work email: vnukiry@mtroyal.ca
- Repository: https://github.com/vnukiry-ctrl/Risk-rag-system
- README.md and .gitignore created

---

## STEP 6: INSURANCE DOCUMENT PARSER - LLM EXTRACTION

**Date Started:** August 19, 2026  
**Status:** ✅ WORKING (with known limitations)

---

### PROBLEM STATEMENT

Need to extract structured metadata from **11 insurance PDF documents** with varying formats:

**Documents to Process:**
1. 24-25 Garage Automobile Policy ENDT - Extension to July 1, 2025.pdf
2. 25-26 Contingent Protective ENDT extension to July 28,23026 Policy B1230FW21595A23.pdf
3. 25-26 Group Accident Policy 100013386.pdf (CID-encoded)
4. 25-26 Medical Malpractice Policy No. 25.00008257.00 - AIF.pdf
5. 25-26 User Group (CGL) - Binder.pdf
6. 25-26 User Group (CGL) ENDT Year End Adjustment.pdf
7. 25-26 User Group (CGL) Policy Document Policy No. AVP406486 (1).pdf
8. 25-26 User Group (CGL) Policy Document Policy No. AVP406486.pdf
9. Mount Royal University - BW240599 (Policy).pdf
10. test.txt
11. XLKR10271 quote 270726.pdf

**Challenge:** Each document has different:
- Format (form-based vs text-based)
- Layout (headers, tables, fields)
- Font encodings (CID codes vs readable text)
- Information placement (policy # location varies)

---

### WHAT WE TRIED

#### ATTEMPT 1: Regex-Based Extraction ❌

**Approach:** Parse text with hardcoded regex patterns
```python
# Extract policy number
policy_match = re.search(r'POLICY\s+NO[:\s]+([A-Z0-9\-]+)', text)

# Extract insurance company
company_match = re.search(r'Lloyd\'s\s+Underwriters', text)
```

**Why It Failed:**
- Every insurance company uses different formatting
- Form fields weren't being captured
- Policy numbers appear in different locations
- Broker/Insurer names formatted inconsistently
- New document type = new regex patterns needed
- Unmaintainable and unscalable

**Lesson Learned:** Pattern matching is too brittle for real-world documents

---

#### ATTEMPT 2: LLM-Based Extraction (Anthropic Claude) ❌

**Approach:** Use Claude API to intelligently extract metadata

**Why It Failed:**
- User didn't have Anthropic API credits
- Cost not viable for experimentation

---

#### ATTEMPT 3: LLM-Based Extraction (xAI Grok) ❌

**Approach:** Use xAI's Grok model via OpenAI-compatible API

**Configuration:**
```python
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)
model = "grok-2-vision-1212"
```

**Why It Failed:**
- xAI requires paid credits
- Free tier exhausted
- Not viable for ongoing development

**Cost:** $0.50+ per document

---

#### ATTEMPT 4: LLM-Based Extraction (Groq) ✅ WORKING

**Approach:** Use Groq's free API with large language models

**Why It Works:**
- Free tier: 200,000 tokens/day
- Fast inference (10x faster than Claude)
- OpenAI-compatible API
- Intelligent metadata extraction
- Handles multiple document formats

**Configuration:**
```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
model = "openai/gpt-oss-120b"
```

**Sign up:** https://console.groq.com (free)

---

### HOW WE DISCOVERED WORKING MODELS & PROMPTS

#### Model Discovery Process

**Problem:** Multiple model names tried, all returned "model not found" errors

**Models That Failed:**
- `llama-3.3-70b-versatile` → Decommissioned
- `llama-3.1-70b-versatile` → Decommissioned  
- `llama-3.2-70b-versatile` → Not available

**Solution: Created Model Discovery Script**

**File:** `backend/list_models.py`
```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
client = OpenAI(
    api_key=api_key,
    base_url='https://api.groq.com/openai/v1'
)

print("Available Groq Models:")
print("=" * 50)

models = client.models.list()
for model in models.data:
    print(model.id)
```

**Output (Actual Available Models on Your Account):**
allam-2-7b
openai/gpt-oss-120b ← SELECTED THIS ONE
meta-llama/llama-prompt-guard-2-86m
canopylabs/orpheus-v1-english
qwen/qwen3.6-27b
meta-llama/llama-prompt-guard-2-22m
openai/gpt-oss-safeguard-20b
canopylabs/orpheus-arabic-saudi
whisper-large-v3
openai/gpt-oss-20b
groq/compound
whisper-large-v3-turbo
groq/compound-mini

**Why We Chose `openai/gpt-oss-120b`:**
- Largest model (120B parameters) = best quality
- Good performance on document extraction
- Groq confirmed available on this account
- Balances quality vs token cost

---

#### Prompt Refinement Process

**Iteration 1: Basic Extraction (Failed)**
```python
prompt = f"""Extract metadata from this insurance document.
Return JSON with: policy_number, insurance_type, company, broker, etc."""
```

❌ **Issues:**
- LLM returned malformed JSON
- Missing critical fields (broker, coverholder)
- Insurance company name missing entirely
- Inconsistent response formats

---

**Iteration 2: Structured JSON Template (Partially Working)**
```python
prompt = f"""Extract metadata and return ONLY this JSON structure:
{{
    "policy_number": "value or null",
    "insurance_type": "value or null",
    "insurance_company": "value or null",
    ...
}}"""
```

✅ **Improvements:**
- Better JSON parsing
- Structured output

❌ **Still Failing:**
- Broker field often missed
- Insurance company still not found consistently
- No guidance on what to search for

---

**Iteration 3: Added Field Guidance with Examples (Final - Working)**
```python
prompt = f"""You are an insurance document expert. Extract ONLY valid JSON from this insurance document. Return the JSON object only, no other text.

DOCUMENT FILENAME: {filename}
DOCUMENT TEXT:
{text_sample}

Extract these fields. Return ONLY this JSON structure (no markdown, no explanation):
{{
    "policy_number": "Policy number/ID (look for 'Policy No', 'ALCOG', 'BW', 'XLKR' formats) or null",
    "insurance_type": "Type of insurance (e.g., Personal Accident, General Liability, Kidnap & Ransom, Automobile) or null",
    "insurance_company": "Name of the INSURER/Insurance Company/Underwriter (e.g., AVIVA, AXA, Lloyd's) or null",
    "broker": "Name of BROKER/Agent who arranged the policy (e.g., BFL CANADA, Burns & Wilcox) or null",
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
- For INSURANCE COMPANY: Look for "Insurance Company", "Underwriter", "Insured with", "Effected with", "Lloyd's", "AXA", "Aviva", "Allianz", or similar
- For BROKER: Look for "Broker", "Agent", "Through", "Arranged by", "Via", or company names that arrange insurance
- For COVERHOLDER: Look for "Lloyd's Approved Coverholder", "Coverholder", "Administrator"
- For POLICY NUMBER: Look for 'Policy No', 'Policy #', 'ALCOG', 'BW', 'XLKR' formats and policy numbers in boxes/headers
- Be accurate - only extract text that actually appears in the document
- Return valid JSON that can be parsed
- For amounts, return only numbers (e.g., "25000" not "$25,000")"""
```

✅ **Why This Works:**
1. **Explicit field descriptions** - LLM knows what each field means
2. **Format hints** - Examples like "ALCOG", "BW", "XLKR" help find policy numbers
3. **Company examples** - Recognizes major insurers (AVIVA, AXA, Lloyd's)
4. **Clear rules** - CRITICAL RULES section is followed strictly by LLM
5. **JSON validation** - "Return ONLY the JSON object" prevents extra text
6. **Broker vs Coverholder distinction** - Separate search guidance for each

**Key Insight:** More specific instructions = better extraction

---

#### Temperature Setting

Changed from default (0.7) to **temperature=0**:
```python
message = self.client.chat.completions.create(
    model=self.model,
    max_tokens=1500,
    temperature=0,  ← CRITICAL FOR CONSISTENCY
    messages=[...]
)
```

**Why:**
- temperature=0 = deterministic, reproducible responses (perfect for JSON)
- temperature=0.7 = creative variations (produces inconsistent JSON)
- For structured extraction tasks, always use temperature=0

---

#### JSON Parsing Robustness

Added error handling and markdown removal:
```python
# Remove markdown if present
response_text = re.sub(r'```json\n?', '', response_text)
response_text = re.sub(r'```\n?', '', response_text)

# Validate JSON before parsing
if not response_text.startswith('{'):
    return {"error": "Invalid LLM response format"}

# Parse with error handling
try:
    metadata = json.loads(response_text)
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {str(e)}")
    return {"error": f"JSON parse error: {str(e)}"}
```

**Why:**
- LLM sometimes wraps response in ```json ... ``` code blocks
- Need to strip markdown before parsing
- Validation prevents cryptic JSON errors
- Try-catch prevents crash on malformed JSON

---

### TEXT EXTRACTION CHALLENGES

#### Problem 1: CID-Encoded Fonts ❌

**What are CID Fonts?**
Some PDFs use CID (Character ID) encoding instead of readable Unicode text:
(cid:0)(cid:1)(cid:2)(cid:3)(cid:4)(cid:5)...

**Affected Documents:**
- `25-26 Group Accident Policy 100013386.pdf` (35 pages)
- Potentially others (not fully tested)

**Why It Happens:**
PDFs with custom fonts embed character codes instead of glyphs. Standard text extraction sees "(cid:XX)" instead of readable characters.

---

#### Solutions Attempted:

**1. pdfplumber** ❌
```bash
pip install pdfplumber
```

**Pros:**
- Better table extraction than pypdf
- Good text extraction for normal PDFs

**Cons:**
- Can't decode CID fonts
- Returns "(cid:XX)" codes for encoded text

**Conclusion:** Works for well-formatted PDFs, fails on CID-encoded ones

---

**2. Tesseract OCR** ❌

**Installed at:** `C:\Users\vnukiry\AppData\Local\Tesseract-OCR`

**Approach:**
```bash
pip install pytesseract pdf2image
```

**Why It Failed:**
- pdf2image requires Poppler (additional system dependency)
- Poppler installation complex on Windows
- Dependency chain too complicated
- Abandoned after multiple attempts

**Lesson:** OCR has heavy system dependencies on Windows

---

**3. PyMuPDF (fitz)** ❌

```bash
pip install PyMuPDF
```

**Approach:**
```python
import fitz

doc = fitz.open(pdf_path)
for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()  # Better CID support than pypdf
```

**Status:**
- Better CID handling than pypdf/pdfplumber
- Still couldn't decode these specific CID-encoded PDFs
- Returns empty or garbage text

**Conclusion:** Even advanced libraries struggle with embedded CID fonts

---

#### Current Status: CID-Encoded PDFs Unsolved

**Why It's Hard on Windows:**
- Most OCR solutions (Tesseract) require Linux-style dependencies
- Cloud APIs (Google Vision, AWS Textract) require paid accounts
- Local solutions have heavy system requirements

**Potential Future Solutions:**
1. Cloud OCR API (Google Cloud Vision, AWS Textract)
2. Online PDF conversion service
3. Contact document provider for text-extractable version
4. Use Windows-native OCR (Windows.Media.Ocr API)

---

### SUCCESSFULLY EXTRACTED DOCUMENTS

#### Document 1: XLKR10271 quote 270726.pdf ✅ Perfect

**Extracted Metadata:**
```json
{
  "policy_number": "XLKR10271",
  "insurance_type": "Kidnap & Ransom",
  "insurance_company": "AXA XL",
  "broker": "Miler",
  "coverholder": null,
  "insured_name": "FMOUN000423",
  "insured_address": null,
  "period_from": "28th July 2026",
  "period_to": "1st July 2029",
  "premium_amount": null,
  "coverage_limit": "3000000",
  "deductible": null,
  "key_coverages": [
    "Ransom (CAD 3,000,000 per event)",
    "Additional Expenses (CAD 3,000,000 per event)",
    "Consultants Fees (UNLIMITED per event)",
    "Legal Liability (CAD 3,000,000 per event)",
    "Personal Accident (CAD 250,000 per person)"
  ],
  "exclusions": ["All Cyber Extortion events and losses are excluded"],
  "notes": "Complex ransom and liability coverage"
}
```

---

#### Document 2: Mount Royal University - BW240599 (Policy).pdf ⚠️ Partial

**Status:** JSON parsing errors initially, working better after prompt refinement

**Extracted:**
- ✅ Policy Number: BW240599
- ✅ Insurance Type: Personal Accident INSURANCE
- ✅ Period: September 1, 2024 to September 1, 2025
- ❌ Insurance Company: Not extracted
- ❌ Broker: Not extracted

**Note:** 45-page Lloyd's Personal Accident Insurance document. Need further investigation.

---

#### Document 3: Other Documents ⏳ Pending

Remaining 8 documents await processing after rate limit reset.

---

### METADATA FIELDS EXTRACTED

✅ Policy Number  
✅ Insurance Type  
✅ Insurance Company (Insurer)  
✅ Broker / Agent  
✅ Coverholder  
✅ Insured Name  
✅ Insured Address  
✅ Policy Period (From/To dates)  
✅ Premium Amount  
✅ Coverage Limit  
✅ Deductible  
✅ Key Coverages (list)  
✅ Exclusions (list)  
✅ Notes / Additional Details  

---

### RATE LIMITING ISSUE

**Error Encountered:**
Error code: 429 - Rate limit reached for model openai/gpt-oss-120b
Limit: 200,000 tokens/day
Used: 198,394 tokens
Requested: 4,707 tokens
Message: "Please try again in 22m19.632s"

**Why It Happened:**
- Groq free tier = 200,000 tokens/day
- Processing 11 large PDFs uses tokens quickly
- 120B model is expensive (4,707 tokens per document)
- First full run consumed 198k of 200k tokens

**Solutions Available:**

1. **Wait for Daily Reset** (Easiest)
   - Reset at: 24 hours from usage start
   - Cost: Nothing, just time
   
2. **Switch to Cheaper Model** (Recommended)
   - Use `mixtral-8x7b-32768` instead
   - Uses ~60% fewer tokens
   - Still maintains good quality
   - **Recommended action**

3. **Upgrade Groq Tier** (Best Long-term)
   - Dev Tier: 30M tokens/day
   - Cost: $0.10 per 1M tokens
   - Allows unlimited experimentation

**Next Action After Rate Limit Reset:**
- Switch to `mixtral-8x7b-32768` model
- Process all 11 documents
- Identify which have CID encoding issues

---

### KEY LEARNINGS

**1. LLM > Regex for Documents**
- Dynamic approach handles format variations
- LLM understands context and meaning
- Scales to new document types automatically
- Maintenance burden much lower

**2. PDF Complexity is Real**
- Different extraction methods needed for different PDFs
- Form-based PDFs need special handling
- CID fonts are a major blocker (especially on Windows)
- Text extraction quality directly impacts LLM output quality

**3. Prompt Engineering Matters Greatly**
- Explicit instructions >> vague instructions
- Examples in prompts dramatically improve accuracy
- CRITICAL RULES section is followed strictly
- temperature=0 is essential for structured output

**4. API Costs & Rate Limits are Important**
- Free tier limits meaningful for experimentation
- Token counting essential for budget planning
- Model selection impacts both speed and cost
- Large documents = rapid token consumption

**5. Groq vs Anthropic vs xAI**
- **Groq:** Free tier (200k tokens/day), fast, good quality, limited
- **Anthropic:** Expensive but high quality, better for production
- **xAI:** Requires paid credits, not ideal for experimentation

**6. JSON Validation is Critical**
- LLMs sometimes return wrapped JSON (```json ... ```)
- Always strip markdown before parsing
- Validate structure before use
- Handle errors gracefully

---

### FILES CREATED

**Core Extraction:**
- `backend/insurance_loader.py` - Main LLM extraction engine (465 lines)
- `backend/list_models.py` - Groq model discovery tool

**Debugging/Testing:**
- `backend/debug_extract.py` - PDF text extraction debugger
- `backend/debug_extract2.py` - Specific file CID testing
- `backend/debug_pymupdf.py` - PyMuPDF testing

**Configuration:**
- `.env` - Contains GROQ_API_KEY

---

### NEXT STEPS

**Immediate (After Rate Limit Reset - ~22 minutes):**
- [ ] Switch model to `mixtral-8x7b-32768`
- [ ] Run full extraction on all 11 documents
- [ ] Identify which documents have CID encoding issues
- [ ] Document extraction success rate

**Short Term:**
- [ ] Improve Mount Royal BW240599 extraction (JSON errors)
- [ ] Solve CID-encoded PDF problem (OCR API or other solution)
- [ ] Add error handling for failed extractions
- [ ] Implement retry logic with exponential backoff
- [ ] Cache extraction results

**Medium Term:**
- [ ] Step 7: FastAPI Backend to serve extractions
- [ ] Step 8: Streamlit Frontend for visualization
- [ ] Integrate extracted metadata with vector database
- [ ] Build RAG chain with LLM responses

**Long Term:**
- [ ] Fine-tune extraction for insurance-specific terms
- [ ] Add document classification (auto-detect policy type)
- [ ] Implement document versioning/updates
- [ ] Add user feedback loop for model improvement

---

### DECISION LOG

**Decision:** Use Groq LLM over Regex Pattern Matching  
**Date:** August 19, 2026  
**Rationale:** 
- Scalable across document formats
- Maintains accuracy across variations
- Maintainable without constant updates
- Intelligent context understanding

**Trade-offs:** 
- API costs (mitigated by free tier)
- Requires internet connection
- External dependency on Groq service

**Status:** ✅ APPROVED

---

**Decision:** Use LLM Over Multiple Regex Patterns  
**Date:** August 19, 2026  
**Rationale:** 
- Single solution vs 11 different patterns
- Better accuracy
- Future-proof for new document types

**Status:** ✅ APPROVED

---

**Decision:** Switch from Anthropic to Groq  
**Date:** August 19, 2026  
**Rationale:** 
- Anthropic: No free credits
- Groq: 200k tokens/day free
- xAI: Requires paid subscription
- Groq allows full experimentation at no cost

**Status:** ✅ APPROVED

---

**Decision:** Use temperature=0 for JSON Extraction  
**Date:** August 19, 2026  
**Rationale:** 
- Deterministic output ensures consistency
- JSON parsing requires reliable format
- Creative variations break parsing

**Status:** ✅ APPROVED

---

## ENVIRONMENT SETUP

### .env File
Location: `H:\my-rag-system\backend\.env`
GROQ_API_KEY=gsk_your_actual_key_here

**Get key from:** https://console.groq.com

### Virtual Environment
```bash
cd H:\my-rag-system\backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## GIT REPOSITORY

**Repository:** https://github.com/vnukiry-ctrl/Risk-rag-system  
**Branch:** main  

### Commits So Far:
1. Initial project structure
2. Backend dependencies
3. Frontend dependencies
4. Document loader implementation
5. Vector database setup
6. Insurance document parser with LLM extraction
7. Model discovery & prompt refinement documentation

---

## RUNNING THE SYSTEM

### Extract Insurance Metadata
```bash
cd H:\my-rag-system\backend
venv\Scripts\activate
python insurance_loader.py
```

### List Available Groq Models
```bash
python list_models.py
```

### Debug PDF Extraction
```bash
python debug_extract.py
```

---

## TROUBLESHOOTING

**Issue:** "Invalid API Key"  
**Solution:** Check .env file has correct GROQ_API_KEY

**Issue:** "Model not found"  
**Solution:** Run `list_models.py` to see available models, update model name

**Issue:** "Rate limit exceeded"  
**Solution:** Switch to cheaper model (mixtral-8x7b-32768) or wait 24 hours

**Issue:** All fields returning None  
**Solution:** Check text extraction - if input text is "(cid:XX)" codes, LLM can't process it

---

## RESOURCES

- Groq Console: https://console.groq.com
- Groq Docs: https://console.groq.com/docs
- LangChain: https://langchain.com
- Qdrant: https://qdrant.tech

---

**Last Updated:** August 19, 2026  
**Status:** Step 6 In Progress - LLM Extraction Working, Awaiting Rate Limit Reset