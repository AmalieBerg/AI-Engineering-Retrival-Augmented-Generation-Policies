"""Document loaders for the policy corpus.

Routes each file to the appropriate LangChain loader based on extension and
attaches consistent metadata (doc_id, doc_title, source_path) to every Document.

The doc_id is the canonical citation key used downstream by the RAG prompt
and the citation-accuracy evaluation. We extract it from the document body
when present (every Northwind policy has a 'Document ID: POL-XX-NNN' header)
and fall back to the filename otherwise.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

# Regex for "Document ID: POL-XX-NNN" — matches in markdown bold, HTML, PDF text, txt
_DOC_ID_PATTERN = re.compile(
    r"Document\s*ID[:\s\*]*\s*(POL-[A-Z]{2,4}-\d{3})",
    re.IGNORECASE,
)

# Regex for the first level-1 heading (markdown) or first non-empty line as title fallback
_MD_TITLE_PATTERN = re.compile(r"^#\s+(.+?)$", re.MULTILINE)


def _extract_doc_id(text: str, fallback: str) -> str:
    """Extract POL-XX-NNN from document text, or fall back to filename stem."""
    match = _DOC_ID_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return fallback


def _extract_title(text: str, fallback: str) -> str:
    """Try to extract a clean document title."""
    # Try markdown H1
    md_match = _MD_TITLE_PATTERN.search(text)
    if md_match:
        return md_match.group(1).strip()
    # Fall back to first non-empty line (works for txt and PDF-extracted text)
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("Document ID"):
            # Strip markdown emphasis if present
            return re.sub(r"[*_#=]+", "", line).strip()
    return fallback


def _load_one(path: Path) -> List[Document]:
    """Load one file using the right LangChain loader for its extension."""
    ext = path.suffix.lower()
    if ext == ".md":
        loader = UnstructuredMarkdownLoader(str(path))
    elif ext == ".html" or ext == ".htm":
        loader = BSHTMLLoader(str(path), open_encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file extension: {ext} ({path})")
    return loader.load()


def load_corpus(corpus_dir: Path) -> List[Document]:
    """Load every supported file under corpus_dir (recursively) into Documents.

    Each Document is decorated with metadata:
      - doc_id:       Canonical POL-XX-NNN identifier (used in citations)
      - doc_title:    Human-readable title
      - source_path:  Relative path from the corpus root
      - file_format:  md | html | pdf | txt
      - page:         (PDFs only) 1-indexed page number from PyPDFLoader

    Returns documents in stable filename-sorted order for reproducible chunking.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    supported_exts = {".md", ".html", ".htm", ".pdf", ".txt"}
    files = sorted(
        p for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_exts
    )

    all_docs: List[Document] = []
    for path in files:
        rel_path = str(path.relative_to(corpus_dir))
        ext = path.suffix.lower().lstrip(".")
        if ext == "htm":
            ext = "html"

        try:
            raw_docs = _load_one(path)
        except Exception as e:
            print(f"  [WARN] Failed to load {rel_path}: {e}")
            continue

        # For PDFs, raw_docs is one Document per page. For everything else, it's a single
        # Document. Either way, we need the FULL document text to extract doc_id/title.
        full_text = "\n".join(d.page_content for d in raw_docs)
        doc_id = _extract_doc_id(full_text, fallback=path.stem.upper())
        doc_title = _extract_title(full_text, fallback=path.stem)

        for d in raw_docs:
            # Preserve any loader-specific metadata (e.g., PyPDFLoader adds 'page')
            d.metadata.update({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "source_path": rel_path,
                "file_format": ext,
            })
            all_docs.append(d)

    return all_docs
