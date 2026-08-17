from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_documents(data_folder="./data"):
    """Load all PDFs, DOCX, and TXT from folder"""
    documents = []
    
    for file in os.listdir(data_folder):
        filepath = os.path.join(data_folder, file)
        
        if file.endswith('.pdf'):
            print(f"Loading PDF: {file}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        
        elif file.endswith('.docx'):
            print(f"Loading DOCX: {file}")
            loader = Docx2txtLoader(filepath)
            documents.extend(loader.load())
        
        elif file.endswith('.txt'):
            print(f"Loading TXT: {file}")
            loader = TextLoader(filepath)
            documents.extend(loader.load())
    
    return documents

def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(chunks)} chunks")
    if chunks:
        print(f"First chunk: {chunks[0].page_content[:200]}...")