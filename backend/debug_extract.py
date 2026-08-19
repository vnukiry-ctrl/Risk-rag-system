from pypdf import PdfReader

# Test the first PDF
pdf_path = r"data/24-25 Garage Automobile Policy ENDT - Extension to July 1, 2025.pdf"

try:
    reader = PdfReader(pdf_path)
    print(f"PDF loaded successfully!")
    print(f"Total pages: {len(reader.pages)}")
    
    print("\n" + "=" * 70)
    print("FIRST PAGE TEXT:")
    print("=" * 70)
    text = reader.pages[0].extract_text()
    print(text[:1500])
    
    print("\n" + "=" * 70)
    print("SEARCHING FOR KEY TERMS:")
    print("=" * 70)
    
    search_terms = ["POLICY", "INSURER", "INSURANCE", "BROKER", "AGENT", "NO."]
    for term in search_terms:
        if term.upper() in text.upper():
            idx = text.upper().find(term.upper())
            print(f"\n✓ Found '{term}':")
            print(f"  {text[max(0, idx-50):idx+150]}")
        else:
            print(f"\n✗ NOT FOUND: {term}")
            
except Exception as e:
    print(f"Error: {e}")