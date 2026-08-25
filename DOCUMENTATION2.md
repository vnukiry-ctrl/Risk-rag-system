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

#### Current Status: CID-Encoded PDFs Unsolved (SKIPPED FOR NOW)

**Why It's Hard on Windows:**
- Most OCR solutions (Tesseract) require Linux-style dependencies
- Cloud APIs require external accounts
- Local solutions have heavy system requirements
- Poppler/pdf2image dependency chain complex

**Document(s) Affected:**
- `25-26 Group Accident Policy 100013386.pdf` (35 pages)
- Potentially others with embedded CID fonts

**Decision:** Skip CID-encoded PDFs for now. Process 10 working documents first, solve this later.

---

### FUTURE SOLUTIONS FOR CID-ENCODED PDFS

**To be implemented in future iterations. Options ranked by recommendation:**

#### Option 1: Google Cloud Vision API ⭐ RECOMMENDED
**Difficulty:** Easy  
**Cost:** Free tier (1,000 requests/month)  
**Setup Time:** 15 minutes

**Pros:**
- Excellent OCR quality
- No system dependencies (cloud-based)
- Free tier sufficient for our use
- Easy Python integration

**Cons:**
- Requires Google Cloud account
- Internet required
- API quota limits

**Implementation:**
```bash
pip install google-cloud-vision pdf2image
```

```python
from google.cloud import vision
from pdf2image import convert_from_path

def extract_with_google_ocr(pdf_path: str) -> str:
    """Extract text from PDF using Google Cloud Vision OCR"""
    client = vision.ImageAnnotatorClient()
    images = convert_from_path(pdf_path)
    text = ""
    
    for image in images:
        # Convert PIL image to bytes
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Call Vision API
        response = client.document_text_detection(
            image=vision.Image(content=img_byte_arr.read())
        )
        text += response.full_text_annotation.text
    
    return text
```

**Estimated Cost:** $0 (free tier) to $3-5 (all 11 PDFs at $0.60 per 1,000 requests)

---

#### Option 2: Online PDF Converter (Simplest) ⭐ IMMEDIATE WORKAROUND
**Difficulty:** Manual  
**Cost:** Free  
**Setup Time:** 2 minutes per PDF

**Best For:** Quick one-time extraction

**Steps:**
1. Go to: https://www.ilovepdf.com/pdf_to_text
2. Upload CID-encoded PDF
3. Download extracted text
4. Save as `.txt` file in data/ folder
5. System automatically processes `.txt` file

**Pros:**
- No coding
- No dependencies
- Instant results
- Free

**Cons:**
- Manual process
- Not scalable

---

#### Option 3: Azure Computer Vision API
**Difficulty:** Medium  
**Cost:** Free tier (5,000 requests/month)  
**Setup Time:** 20 minutes

**Pros:**
- Good OCR quality
- Free tier sufficient
- Microsoft backing
- Reliable

**Cons:**
- Requires Azure account
- Slightly more complex setup
- Internet required

**Documentation:** https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/

---

#### Option 4: AWS Textract
**Difficulty:** Medium  
**Cost:** $0.01 per page (first 100 pages free/month)  
**Setup Time:** 20 minutes

**Pros:**
- Enterprise-grade OCR
- Very accurate
- AWS ecosystem integration

**Cons:**
- Paid after free tier
- More complex setup
- Slower response times

---

#### Option 5: Windows Native OCR (Windows.Media.Ocr)
**Difficulty:** Hard  
**Cost:** Free (built-in)  
**Setup Time:** 30 minutes

**Pros:**
- No external dependencies
- Free
- Runs locally

**Cons:**
- Windows API (requires ctypes/pywin32)
- Complex implementation
- Lower accuracy than cloud options

---

### RECOMMENDATION FOR NEXT ITERATION

**Short Term (Next Week):**
- Use Option 2 (Online PDF Converter) for quick results
- Manually extract CID PDFs
- Add `.txt` versions to data/ folder

**Medium Term (Next Sprint):**
- Implement Option 1 (Google Cloud Vision)
- Automate CID PDF detection
- Integrate into extraction pipeline
- Budget: ~$5 for all documents

**Implementation Steps:**
1. Create Google Cloud project
2. Enable Vision API
3. Create service account key
4. Update `insurance_loader.py` to detect CID fonts
5. Route CID PDFs to Google Vision
6. Fall back to pdfplumber for normal PDFs

---
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

---

## STEP 7: FASTAPI BACKEND

**Date Completed:** August 19, 2026  
**Status:** ✅ WORKING

### What We Built

REST API backend to serve insurance documents and enable RAG queries.

**File:** `backend/main.py`

### Endpoints Implemented
GET / - Health check
POST /extract - Extract metadata from all documents
GET /documents - List all extracted documents
GET /document/{id} - Get specific document
GET /health - Detailed health status

### Technology Stack

- **Framework:** FastAPI
- **Server:** Uvicorn
- **LLM:** Groq (mixtral-8x7b-32768)
- **Port:** 8000
- **CORS:** Enabled (all origins)

### Running the Backend

```bash
cd backend
venv\Scripts\activate.bat
python main.py
```

Server starts on: **http://localhost:8000**  
API Docs: **http://localhost:8000/docs**

### API Documentation

FastAPI auto-generates interactive documentation at `/docs` endpoint.
- Try all endpoints in browser
- See request/response schemas
- Test with real data

### Data Models

```python
class ExtractionResult(BaseModel):
    policy_number: Optional[str]
    insurance_type: Optional[str]
    insurance_company: Optional[str]
    broker: Optional[str]
    coverholder: Optional[str]
    insured_name: Optional[str]
    insured_address: Optional[str]
    period_from: Optional[str]
    period_to: Optional[str]
    premium_amount: Optional[str]
    coverage_limit: Optional[str]
    deductible: Optional[str]
    key_coverages: List[str]
    exclusions: List[str]
    notes: Optional[str]
    source_file: str
    extracted_date: str
```

### Known Issues

- `/query` endpoint needs refinement (422 errors)
- Vector store integration pending
- RAG chain not fully connected yet

### Next Iteration

- Fix query endpoint with proper JSON validation
- Integrate vector database for semantic search
- Connect Groq LLM for intelligent answering

---

## STEP 8: STREAMLIT FRONTEND

**Date Completed:** August 19, 2026  
**Status:** ✅ RUNNING

### What We Built

Interactive web dashboard for insurance document analysis.

**File:** `frontend/app.py`

### Features Implemented

**🏠 Home Page**
- Health status indicator
- Quick start extraction button
- System overview and features
- Technology stack info

**📄 Documents Page**
- List all extracted documents
- Expandable cards for each policy
- Displays:
  - Policy information (number, type, company, broker)
  - Coverage and dates
  - Insured details
  - Coverages and exclusions
- Refresh button

**🔍 Search Page**
- Keyword/phrase search
- Adjustable result count (1-10)
- Shows top matching documents
- Preview snippets

**❓ Ask Questions Page**
- Natural language question input
- Adjustable context documents (1-10)
- AI-powered answers from Groq LLM
- Source attribution

### Technology Stack

- **Framework:** Streamlit
- **HTTP Client:** Requests
- **Port:** 8501
- **UI:** Interactive widgets

### Running the Frontend

```bash
cd frontend
venv\Scripts\activate.bat
streamlit run app.py
```

Dashboard available at: **http://localhost:8501**

### Architecture
Streamlit Frontend (8501)
↓
HTTP Requests
↓
FastAPI Backend (8000)
↓
Insurance Loader
↓
Groq LLM (Extraction & Answering)

### Streamlit Pages

1. **Home** - Overview and quick actions
2. **Documents** - Browse extracted metadata
3. **Search** - Find documents by keyword
4. **Ask Questions** - Interactive Q&A with LLM

### Navigation

- Sidebar radio buttons for page selection
- Back/forward browser navigation supported
- Session state management

### UI Components

- Metrics display (documents loaded, API status)
- Expandable sections with `st.expander()`
- Text input for search and questions
- Sliders for parameter adjustment
- Column layouts for responsive design
- Error/info/success message notifications

### Data Flow
User Input (Streamlit)
↓
HTTP Request to FastAPI
↓
Insurance Loader (Extract)
↓
Groq LLM (Process)
↓
JSON Response
↓
Display in Streamlit

### Known Issues

- `/query` endpoint not fully functional (422 errors)
- Search endpoint needs vector database
- Q&A feature pending backend fix

### Future Enhancements

- Document filtering by type/company
- Advanced search with date ranges
- Export documents to PDF/Excel
- User preferences/bookmarks
- Document comparison tool
- Policy recommendation engine

---

## TWO-TERMINAL SETUP

### Terminal 1: Backend

```bash
cd H:\my-rag-system\backend
venv\Scripts\activate.bat
python main.py
```

**Keep running!** Shows: `Uvicorn running on http://127.0.0.1:8000`

### Terminal 2: Frontend

```bash
cd H:\my-rag-system\frontend
venv\Scripts\activate.bat
streamlit run app.py
```

**Keep running!** Shows: `Local URL: http://localhost:8501`

### Access Points

- **API:** http://localhost:8000/
- **API Docs:** http://localhost:8000/docs
- **Dashboard:** http://localhost:8501

---

## SYSTEM ARCHITECTURE
┌─────────────────────────────────────────┐
│ Streamlit Frontend (8501) │
│ ┌────────────────────────────────────┐ │
│ │ Home | Documents | Search | Q&A │ │
│ └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
↓ HTTP Requests
┌─────────────────────────────────────────┐
│ FastAPI Backend (8000) │
│ ┌────────────────────────────────────┐ │
│ │ /extract /documents /query │ │
│ │ /search/metadata /health │ │
│ └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
↓
┌─────────────────────────────────────────┐
│ Insurance Loader + LLM │
│ ┌────────────────────────────────────┐ │
│ │ PDF Extraction (pdfplumber) │ │
│ │ LLM Processing (Groq) │ │
│ │ Metadata Extraction │ │
│ └────────────────────────────────────┘ │
└─────────────────────────────────────────┘

---

## DEPLOYMENT STATUS

**Current:** Local development  
**Backend:** Running on localhost:8000  
**Frontend:** Running on localhost:8501  
**Database:** In-memory (documents_db dict)  
**Vector Store:** Qdrant (initialized but not integrated)  

**To Deploy:**
1. Move to cloud server
2. Update connection strings
3. Add authentication
4. Configure HTTPS
5. Set up persistent database
6. Add logging and monitoring

---

## FILES IN STEPS 7-8

**Backend:**
- `backend/main.py` - FastAPI application (simplified)
- `backend/test.py` - Test server (currently running)

**Frontend:**
- `frontend/app.py` - Streamlit dashboard

**Configuration:**
- `.env` - Environment variables (GROQ_API_KEY)

---

## TESTING CHECKLIST

✅ Backend starts and listens on 8000  
✅ Frontend starts and loads on 8501  
✅ `/extract` endpoint works  
✅ `/documents` endpoint works  
✅ Home page displays correctly  
✅ Documents page shows metadata  
⚠️ `/query` endpoint needs fixing  
⚠️ Search/Q&A features pending  

---

## NEXT STEPS

**Immediate:**
- [ ] Fix `/query` endpoint (JSON validation)
- [ ] Integrate vector store for search
- [ ] Connect RAG chain

**Short Term:**
- [ ] Add authentication
- [ ] Implement document upload
- [ ] Add export functionality
- [ ] Create admin dashboard

**Long Term:**
- [ ] Deploy to cloud
- [ ] Add persistent database
- [ ] Scale to handle more documents
- [ ] Fine-tune extraction for specific domains

---

**Last Updated:** August 19, 2026  
**Status:** Steps 1-8 Complete (Core Functionality Working)

---
---

## STEP 9: MILESTONE 1 HARDENING — Cleaning, Chunking, Error Handling, Logging

**Date Completed:** August 24, 2026  
**Status:** ✅ COMPLETE

### What We Built

Milestone 1 (Core Data Pipeline) had working LLM extraction from Step 6, but no text cleaning, an unused/unreliable chunking strategy, silent failures, and print-only diagnostics. This step closed all four gaps.

---

#### 1. Text Normalization

**Problem:** PyMuPDF preserves the original PDF page layout, so extracted text is full of lines that are empty except for spaces (column/table artifacts). That noise ate into the 8,000-character sample sent to the LLM for metadata extraction, and diluted the chunks used for embedding search.

**Fix:** Added `normalize_text()` in `insurance_loader.py` — strips whitespace-only lines, collapses blank-line runs, collapses repeated spaces. Applied to every document (PDF and TXT) right after extraction, before both LLM extraction and chunking.

**Also removed:** a dead code path in `extract_text_from_pdf()` that called `page.get_text("blocks")` and tried to iterate it as if it returned dicts. Confirmed via testing that `"blocks"` mode returns tuples, so the `isinstance(block, dict)` check was always `False` — it ran on every page and did nothing.

**Measured effect** (45-page Mount Royal policy PDF): overall text reduced 111,430 → 107,050 characters (3.9% noise removed); non-whitespace content in the first 8,000 characters (the LLM's extraction window) increased from 6,285 to 6,657 characters (~6% more real content in the same budget).

---

#### 2. Chunking Strategy: Parent-Child (Size-Based)

**Problem:** `insurance_loader.py` had a `chunk_by_sections()` method that split documents on `SECTION X:` headers — but it was never actually wired into indexing (`vector_store.py` re-split the raw text independently), and testing against all 9 real documents showed the regex was unreliable: 4 of 9 documents had zero `SECTION` matches at all, and most "matches" that did occur were the word "section" appearing mid-sentence in prose, not real headers.

**Decision:** Since more documents (of varying formats) will be added later, we chose not to build retrieval around document structure that isn't consistently present. Deleted `chunk_by_sections()` entirely.

**Fix:** Implemented size-based parent-child ("small-to-big") chunking in `vector_store.py`'s `index_documents()`:
- Each document is split into **parent chunks** (~2,000 characters) for context.
- Each parent is further split into **child chunks** (~400 characters), which are what get embedded — small chunks retrieve more precisely.
- Each child's stored payload carries its own text and its parent's full text.
- `semantic_search()` matches against child embeddings but returns the parent text, so a precise match still comes back with surrounding context instead of an isolated fragment. Results are deduplicated by parent (multiple children of one parent can all match the same query).

**Verified:** indexed 3 real documents → 432 child chunks; a test query returned deduplicated ~1,900–2,000 character parent passages (not tiny fragments), confirming the pipeline works end-to-end with Qdrant + Ollama.

---

#### 3. Error Handling

**Problem:** Failures were being silently dropped. An unreadable PDF (`extract_text_from_pdf()` returning `""`) triggered a bare `continue` with no record anywhere — the caller had no way to know the file was even attempted. `/extract`'s response mixed successful and failed documents into one list. A failure in `index_documents()` (e.g. Ollama not running) would 500 the entire `/extract` call and discard already-completed LLM extraction work.

**Fix:**
- Added `_error_record()` helper (reused across all failure paths) producing a consistent `{"error", "source_file", "extracted_date"}` shape — the same shape `extract_metadata_with_llm()` already used for its own error returns.
- Unreadable PDFs and unexpected per-file exceptions now append an error record instead of vanishing.
- `/extract`'s response splits into `successful` / `failed` lists.
- Indexing failures are caught separately from extraction failures, so a broken embedding service no longer discards extraction work — the response includes `chunks_indexed: 0` and an `indexing_error` message instead.
- `vector_store.py`'s Ollama HTTP call now raises a clear, actionable `RuntimeError` (e.g. "Could not reach Ollama at http://localhost:11434 — is it running?") instead of surfacing a raw `requests` traceback.

**Verified:** tested with a corrupted PDF (confirmed it now produces a visible error record) and an unreachable Ollama endpoint (confirmed the clear `RuntimeError` message).

---

#### 4. Logging

**Problem:** All diagnostics were `print()` statements — no timestamps, no severity levels, no way to distinguish routine progress from a real failure, and nothing captured if this ran outside an interactive terminal. `extract_metadata_with_llm()`'s exception handlers only logged `str(e)`, discarding the actual traceback.

**Fix:**
- Configured Python's standard `logging` module once in `main.py` (`logging.basicConfig`, level from `LOG_LEVEL` env var, default `INFO`).
- Replaced all `print()` calls in `insurance_loader.py` and `vector_store.py` with `logger.info` / `logger.warning` / `logger.error(..., exc_info=True)` as appropriate. Condensed the previous 8-line-per-document field dump into a single `INFO` line.
- Added logging to `/extract` and `/query` request handling in `main.py`, including `logger.exception(...)` before every `HTTPException`.
- `vector_store.py`'s `index_documents()` — previously silent — now logs chunk counts on start and finish.

**Real-world validation:** while testing, repeated extraction runs exhausted the Groq API's daily token quota (200,000 TPD). This surfaced a real bug: `load_insurance_documents()` was logging `"Extracted <file>: policy=N/A..."` (implying success) even when metadata extraction had actually failed with a `429 Rate Limit` error. Fixed by checking for `"error"` in the returned metadata and logging a `WARNING` instead of a false `INFO` success line. Also confirmed the per-file exception handling correctly let the loop continue through all 9 documents despite every one failing on that run — the pipeline degrades gracefully under a real, non-code failure (API quota) rather than crashing.

---

### Decision Log

**Decision:** Size-based parent-child chunking over header-based chunking  
**Date:** August 24, 2026  
**Rationale:** Section-header regex only matched 5 of 9 real documents reliably, and more documents (potentially other formats) will be added in deployment. Structure-agnostic chunking works the same regardless of document formatting.  
**Status:** ✅ APPROVED

**Decision:** Reuse the existing error-dict convention instead of introducing exceptions or a new error framework  
**Date:** August 24, 2026  
**Rationale:** `extract_metadata_with_llm()` already returned `{"error": ...}` dicts on failure; extending that same shape to PDF-read and per-file failures keeps the codebase consistent without adding a new pattern for a project this size.  
**Status:** ✅ APPROVED

---

### Files Changed
- `backend/insurance_loader.py` — normalization, error records, logging, removed dead chunking code
- `backend/vector_store.py` — parent-child chunking, indexing error handling, logging
- `backend/main.py` — logging config, request-level logging, split successful/failed response, isolated indexing failures

---

**Last Updated:** August 24, 2026  
**Status:** Milestone 1 (Core Data Pipeline) Complete

---
---

## STEP 10: MILESTONE 2 — VECTOR SEARCH FOUNDATION

**Date Completed:** August 25, 2026
**Status:** ✅ COMPLETE

### What We Built

Milestone 1 left the vector search pieces (embeddings, Qdrant, semantic search, parent-child retrieval) implemented but unverified against real data, running on an in-memory store, with no performance baseline and a misleadingly-named endpoint. This step closed those gaps.

---

#### 1. Persistent Vector Storage

**Problem:** `vector_store.py` used `QdrantClient(":memory:")`. Every backend restart silently wiped the entire vector index — all 1,800+ indexed chunks gone, requiring a full re-run of `/extract` (LLM extraction + embedding) just to get back to a working state.

**Fix:** Switched to Qdrant's on-disk local mode: `QdrantClient(path=QDRANT_PATH)`, where `QDRANT_PATH` defaults to `./qdrant_data` and is overridable via env var. `index_documents()` and `semantic_search()` needed no changes — same client API either way.

**Migration path (documented, not yet implemented):** for deployment, swap to `QdrantClient(url=...)` pointing at a real Qdrant server/cloud instance. One-line change; everything downstream is unaffected. On-disk local mode also only supports one process at a time (file lock) — fine for a single dev backend, but a reason to make that swap before running multiple workers.

**Verified:** wrote a test point with one Python process, opened a fresh process pointing at the same path, confirmed the collection was still there. `backend/qdrant_data/` added to `.gitignore` (regenerated data, not source — same treatment as the PDF data files).

---

#### 2. Real Index Rebuild + End-to-End Verification

**Problem:** Milestone 1's testing used 3 sample documents; Milestone 2's retrieval logic had never been run against all 9 real insurance PDFs or through the actual running API.

**What happened:** Found and killed a stale `uvicorn` process left running from the previous day (still on the old in-memory code) that was silently blocking port 8000. Started a fresh server on the new persistent-storage code and ran `/extract` for real.

**Result:** 8 of 9 documents extracted successfully by the LLM. One (`25-26 Group Accident Policy 100013386.pdf`) failed with a Groq `413` — the document's text is too large for `openai/gpt-oss-120b`'s per-request token budget (8,000 TPM limit, document requested 9,293). This is independent of indexing: **all 9 documents' text was still chunked and embedded** — 1,810 child chunks indexed with no indexing errors.

**Verified `/query` end-to-end:** asked "What is the coverage limit for the Mount Royal University Commercial General Liability policy?" — got the correct answer ($5,000,000, matching the extracted `AVP406486` policy data) with the right source document ranked first, in 3.4s.

**Discovered along the way:** `documents_db` (the structured metadata dict used to build the "authoritative summary" in `/query` prompts) is in-memory only, unlike the Qdrant index. After restarting the server to pick up the `/search` rename (below), a query that previously answered correctly ("what's the deductible") degraded to "not found in the excerpts" because `documents_db` was empty — the vector index survived the restart, the metadata dict didn't. Re-running `/extract` restored correct answers. **This is a real gap**: only half of the system's state is actually persistent right now.

---

#### 3. Performance Smoke Test

**Problem:** Milestone 1/2 checklists claimed "performance testing (latency, accuracy)" was done. It wasn't — no test script or results existed anywhere in the repo.

**Fix:** Added `backend/perf_smoke_test.py` — 5 known-answer questions (one per distinct policy), run against the live `/query` endpoint. For each, checks (a) the expected policy number appears in the returned sources, and (b) records latency.

**Result:** 5/5 retrieval hit rate; all spot-checked answers factually correct against the extracted data. Latency ranged 2.8s–22.9s (avg 12.6s) — **not a retrieval problem** (Qdrant search is local and fast), but Groq's free-tier rate limiting causing retries on the answer-generation call, same 429 behavior seen during `/extract`. Retrieval quality is solid; LLM answer-generation latency is the real bottleneck, and it's tied to the Groq free tier rather than anything in the vector search pipeline.

**Scope note:** this is a lightweight sanity check, not the full k-tuning evaluation methodology (golden Q&A set, Recall@k/MRR, generation-quality curves across k values). That's deliberately deferred until real production-scale documents and a proper golden dataset exist — see the "top-k tuning" discussion in this milestone's planning notes.

---

#### 4. Fixed the `/search` Endpoint Name

**Problem:** `/search` in `main.py` did plain substring matching over extracted metadata fields (`documents_db`) — not vector search. Its name made it indistinguishable from `/query` (the actual semantic search + LLM answer endpoint), which could mislead anyone integrating against the API.

**Fix:** Renamed to `/search/metadata` with a docstring clarifying it's keyword search over structured fields, not semantic search — "for vector-based retrieval over document content, use `/query`." Updated the one caller (`frontend/app.py`'s Search page) and the architecture diagram in this document.

**Verified:** old `/search` path now 404s; new `/search/metadata` path works identically to the old behavior.

---

#### 5. Removed Dead Code

`backend/rag_chain.py` was an empty leftover file from an earlier build step, never imported anywhere. Deleted after confirming no references in the app code.

---

### Known Gaps (Not Fixed This Milestone)

- **`documents_db` doesn't persist** across backend restarts (see item 2) — only the Qdrant vector index does. A restart requires re-running `/extract` to restore full `/query` answer quality.
- **One document can't be extracted** (`25-26 Group Accident Policy 100013386.pdf`) due to Groq's per-request token limit — its content is still searchable via `/query`, just without structured metadata (policy number, premium, etc.).
- **Top-k is still a fixed default (5)**, not tuned. Deliberately deferred: proper tuning needs a golden Q&A evaluation set built from real production documents, which don't fully exist yet (see planning discussion for the intended methodology: separate retrieval metrics from generation metrics, sweep k, account for the parent-child dedup mechanic, pick the elbow).
- **Groq LLM answer-generation latency** is inconsistent (free-tier rate limits) — not something to fix in this codebase, but worth knowing before treating `/query` latency numbers as representative of production behavior.
- The stubbed Anthropic embeddings option in `vector_store.py` remains unimplemented/unverified — intentionally kept for future use, not touched this milestone.

---

### Decision Log

**Decision:** On-disk local Qdrant mode now, server/cloud mode later at deployment
**Date:** August 25, 2026
**Rationale:** Same client API either way — `index_documents()`/`semantic_search()` don't change. Local mode needs no extra service for development; a real Qdrant server is more appropriate once there's a deployment target and possibly multiple worker processes (local mode only supports one process at a time via its file lock).
**Status:** ✅ APPROVED

**Decision:** Rename `/search` to `/search/metadata` rather than delete it
**Date:** August 25, 2026
**Rationale:** The keyword/metadata search is a legitimately different, still-useful capability (e.g. "find policies from Aviva") — the problem was only the misleading name next to `/query`, which does the real semantic search.
**Status:** ✅ APPROVED

**Decision:** Defer full top-k tuning until real documents + a golden eval set exist
**Date:** August 25, 2026
**Rationale:** Rigorous k-tuning requires ground-truth Q&A pairs across representative document types; the current 9-document set is a starting point, not the production corpus. Tuning now would optimize for the wrong distribution.
**Status:** ✅ APPROVED (deferred, not skipped)

---

### Files Changed
- `backend/vector_store.py` — persistent on-disk Qdrant client (`QdrantClient(path=...)` instead of `:memory:`)
- `backend/main.py` — `/search` renamed to `/search/metadata` with clarifying docstring
- `frontend/app.py` — updated to call `/search/metadata`
- `backend/perf_smoke_test.py` — new, latency + retrieval-accuracy smoke test
- `backend/rag_chain.py` — deleted (dead file)
- `.gitignore` — added `backend/qdrant_data/`

---

**Last Updated:** August 25, 2026
**Status:** Milestone 2 (Vector Search Foundation) Complete