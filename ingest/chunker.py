"""Chunking strategy for the policy corpus.

We use a hybrid approach:
  1. For markdown content, split on heading boundaries (H1/H2/H3) first to
     preserve semantic sections. Each section is then further chunked with a
     character-level recursive splitter if it exceeds chunk_size.
  2. For all other formats (or markdown sections that are already small enough),
     fall back to RecursiveCharacterTextSplitter.

The recursive splitter uses an overlap of chunk_overlap characters between
adjacent chunks. Per the Quantic course note: "larger chunk sizes with more
overlap reduce the potential for related information to be split across chunks."

Each output chunk inherits the parent document's metadata and additionally gets:
  - chunk_index:    0-indexed position within the document
  - section_path:   for markdown, the heading breadcrumb (e.g., "3. PTO Accrual")
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# Heading levels we split markdown on, with the metadata key for each
_MD_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def _build_char_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try paragraph breaks first, then sentence breaks, then words
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def _section_path(meta: dict) -> str:
    """Build a 'h1 > h2 > h3' style breadcrumb from header metadata."""
    parts = [meta.get(k) for k in ("h1", "h2", "h3")]
    return " > ".join(p for p in parts if p)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Chunk a list of loaded documents into retrieval-ready chunks.

    Markdown documents are split on heading boundaries first, then any oversized
    sections are further reduced via recursive character splitting. Non-markdown
    documents are split directly with the recursive splitter.
    """
    md_header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS,
        # Keep headers in the body of the chunk -- helps the retriever
        # understand context and helps the LLM cite the section.
        strip_headers=False,
    )
    char_splitter = _build_char_splitter(chunk_size, chunk_overlap)

    chunks: List[Document] = []

    # Group by source so we can re-index chunks per-document
    by_source: dict[str, List[Document]] = {}
    for d in documents:
        by_source.setdefault(d.metadata.get("source_path", "unknown"), []).append(d)

    for source_path, source_docs in by_source.items():
        # Recombine multi-page documents (e.g., PDFs) into one text body so we
        # don't artificially split on page boundaries.
        full_text = "\n\n".join(d.page_content for d in source_docs)
        base_meta = {**source_docs[0].metadata}
        # Strip the page number from base meta (we'll add it back per chunk where useful)
        base_meta.pop("page", None)

        fmt = base_meta.get("file_format", "")

        # Markdown: split on headings first, then size-bound each section
        if fmt == "md":
            heading_chunks = md_header_splitter.split_text(full_text)
            doc_chunks: List[Document] = []
            for hc in heading_chunks:
                hc_meta = {**base_meta, **hc.metadata}
                hc_meta["section_path"] = _section_path(hc_meta)
                if len(hc.page_content) <= chunk_size:
                    doc_chunks.append(Document(page_content=hc.page_content, metadata=hc_meta))
                else:
                    # Section too big — sub-split
                    sub_texts = char_splitter.split_text(hc.page_content)
                    for sub in sub_texts:
                        doc_chunks.append(Document(page_content=sub, metadata=dict(hc_meta)))
            split_chunks = doc_chunks
        else:
            # HTML / PDF / TXT: recursive character split on the full text
            sub_texts = char_splitter.split_text(full_text)
            split_chunks = [
                Document(page_content=t, metadata=dict(base_meta)) for t in sub_texts
            ]

        # Stamp chunk_index per document for stable references
        for i, c in enumerate(split_chunks):
            c.metadata["chunk_index"] = i
            chunks.append(c)

    return chunks
