"""Standalone verification of loader logic against the corpus.

This script doesn't import langchain (so it runs without the full requirements
installed). It replicates the doc_id / title extraction and file routing logic
from ingest/loaders.py to verify it works on the actual corpus files.

Useful in restricted environments and as a quick sanity check.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader
from bs4 import BeautifulSoup

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

DOC_ID_PATTERN = re.compile(
    r"Document\s*ID[:\s\*]*\s*(POL-[A-Z]{2,4}-\d{3})",
    re.IGNORECASE,
)
MD_TITLE = re.compile(r"^#\s+(.+?)$", re.MULTILINE)


def extract_doc_id(text: str, fallback: str) -> str:
    m = DOC_ID_PATTERN.search(text)
    return m.group(1).upper() if m else fallback


def extract_title(text: str, fallback: str) -> str:
    m = MD_TITLE.search(text)
    if m:
        return m.group(1).strip()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("Document ID"):
            return re.sub(r"[*_#=]+", "", line).strip()
    return fallback


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext in (".html", ".htm"):
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        # BSHTMLLoader uses the document's text content
        return soup.get_text("\n", strip=True)
    return path.read_text(encoding="utf-8")


def main() -> int:
    if not CORPUS_DIR.is_dir():
        print(f"ERROR: corpus not found at {CORPUS_DIR}", file=sys.stderr)
        return 1

    supported = {".md", ".html", ".htm", ".pdf", ".txt"}
    files = sorted(p for p in CORPUS_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in supported)

    print(f"Found {len(files)} files in corpus\n")

    fmt_counts: dict[str, int] = {}
    doc_id_to_files: dict[str, list[str]] = {}
    issues: list[str] = []

    for path in files:
        rel = path.relative_to(CORPUS_DIR)
        ext = path.suffix.lower().lstrip(".")
        if ext == "htm":
            ext = "html"
        fmt_counts[ext] = fmt_counts.get(ext, 0) + 1

        try:
            text = read_file(path)
        except Exception as e:
            issues.append(f"  [ERROR] {rel}: failed to read - {e}")
            continue

        if not text.strip():
            issues.append(f"  [ERROR] {rel}: extracted text is empty")
            continue

        doc_id = extract_doc_id(text, fallback=path.stem.upper())
        title = extract_title(text, fallback=path.stem)
        doc_id_to_files.setdefault(doc_id, []).append(str(rel))

        # Verify doc_id was actually extracted (not the fallback)
        if not doc_id.startswith("POL-"):
            issues.append(f"  [WARN] {rel}: no POL-XX-NNN found, fell back to '{doc_id}'")

        print(f"  {ext:5s}  {doc_id}  {title[:50]:50s}  ({len(text):>6,} chars)")

    print(f"\nFormat distribution: {fmt_counts}")
    print(f"Unique doc IDs: {len(doc_id_to_files)}")

    # Detect duplicates
    duplicates = {k: v for k, v in doc_id_to_files.items() if len(v) > 1}
    if duplicates:
        print("\n[WARN] Duplicate doc_id assignments:")
        for did, paths in duplicates.items():
            print(f"  {did}: {paths}")

    if issues:
        print("\n=== ISSUES ===")
        for issue in issues:
            print(issue)
        return 1

    print("\nAll files parsed cleanly with valid doc_ids.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
