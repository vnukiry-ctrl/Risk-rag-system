"""
Milestone 4 performance benchmark: indexing/embedding throughput.

WHAT THIS MEASURES: time to chunk + embed + write the current backend/data/
documents into Qdrant, via vector_store.index_documents() directly -- the
half of the pipeline query latency (perf_smoke_test.py, tests/test_quality.py)
doesn't touch. Tracking this separately matters because it scales with
document COUNT and SIZE, not with query volume: it's the number that
degrades quietly as more documents get added, long before anyone notices
query latency getting worse.

MEANING of what's reported:
- seconds per document / per chunk: the actual embedding+upsert cost.
  IDEAL: near-linear scaling with chunk count (batch_size=50 in
  vector_store.py already gets ~15x over one-request-per-chunk -- see the
  DECISION comment there). NOT GOOD: a large jump in seconds/chunk after a
  change to batch_size, chunk size, or embedding model -- that's exactly the
  kind of regression this benchmark exists to catch before it's noticed
  weeks later as "indexing got slow."
- chunk count: NOT a quality signal by itself (more chunks isn't "better"),
  just useful context for interpreting the timing.

IMPORTANT -- READ BEFORE RUNNING:
1. This calls index_documents(), which does client.recreate_collection() --
   it REBUILDS the Qdrant collection from backend/data/ PDFs. Existing query
   results depend on this collection, so re-run /extract afterward if you
   need documents_db's in-memory metadata repopulated (see README's known
   gap: documents_db doesn't persist across restarts, but the Qdrant index
   the benchmark writes here does).
2. Qdrant's on-disk local mode locks its storage folder to ONE process at a
   time (confirmed directly while building this suite -- see
   tests/conftest.py for the exact error). STOP the running backend
   (uvicorn) before running this script, or it will fail with a
   portalocker "AlreadyLocked" error, not a graceful retry.
3. This intentionally bypasses insurance_loader's LLM metadata-extraction
   step (extract_metadata_with_llm) -- that step costs a real Groq API call
   per document and measures something else entirely (extraction quality/
   latency, not embedding/indexing throughput). Only text extraction +
   normalization + chunking + embedding are timed here.

Usage: stop the backend, then run this script directly.
"""
import glob
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from insurance_loader import extract_text_from_pdf, normalize_text, DEFAULT_DATA_FOLDER
from vector_store import index_documents


def build_parent_chunks_without_llm_extraction():
    """Same shape as insurance_loader.load_insurance_documents()['parent_chunks'],
    minus the LLM metadata call -- metadata fields are left None since this
    benchmark isn't measuring extraction quality."""
    parent_chunks = []
    pdf_paths = sorted(glob.glob(os.path.join(DEFAULT_DATA_FOLDER, "*.pdf")))
    for path in pdf_paths:
        filename = os.path.basename(path)
        text = normalize_text(extract_text_from_pdf(path))
        if not text.strip():
            print(f"  skipping {filename}: no text extracted")
            continue
        parent_chunks.append({
            "content": text,
            "metadata": {},
            "source_file": filename,
        })
    return parent_chunks


def run():
    print("Building parent chunks (PDF extraction + normalization, no LLM calls)...")
    start = time.time()
    parent_chunks = build_parent_chunks_without_llm_extraction()
    extract_elapsed = time.time() - start
    print(f"  {len(parent_chunks)} documents extracted in {extract_elapsed:.2f}s")

    print("\nIndexing (chunking + embedding + Qdrant upsert)...")
    start = time.time()
    chunks_indexed = index_documents(parent_chunks)
    index_elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"Documents processed:      {len(parent_chunks)}")
    print(f"Child chunks indexed:     {chunks_indexed}")
    print(f"Text extraction time:     {extract_elapsed:.2f}s ({extract_elapsed / max(len(parent_chunks), 1):.2f}s/doc)")
    print(f"Embedding + indexing time: {index_elapsed:.2f}s ({index_elapsed / max(chunks_indexed, 1):.3f}s/chunk)")
    print(f"Total:                    {extract_elapsed + index_elapsed:.2f}s")
    print("=" * 60)
    print(
        "\nReminder: the Qdrant collection was just rebuilt from backend/data/ with "
        "empty metadata (no policy_number/insurer/etc.) -- run /extract before using "
        "/query again if you need real structured metadata back."
    )


if __name__ == "__main__":
    run()
