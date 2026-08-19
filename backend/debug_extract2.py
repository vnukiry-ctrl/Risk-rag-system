import pdfplumber

pdf_path = r"data/25-26 Group Accident Policy 100013386.pdf"

try:
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        
        for page_num in range(min(2, len(pdf.pages))):
            text = pdf.pages[page_num].extract_text()
            print(f"\n{'='*70}")
            print(f"PAGE {page_num + 1} TEXT:")
            print(f"{'='*70}")
            print(f"Length: {len(text) if text else 0} characters")
            print(text[:1500] if text else "NO TEXT EXTRACTED")
            
            # Also check for tables
            tables = pdf.pages[page_num].extract_tables()
            if tables:
                print(f"\nFound {len(tables)} table(s) on this page")
                
except Exception as e:
    print(f"Error: {e}")