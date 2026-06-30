# Corpus Manifest — Northwind Technologies Policies

Synthetic policy corpus for the AI Engineering Project RAG system. All documents are fictional and authored for this project; they describe a made-up company called "Northwind Technologies."

## Document Index

| # | Doc ID | Title | Format | Topic Area | Owner |
|---|--------|-------|--------|------------|-------|
| 1 | POL-HR-001 | PTO and Vacation Policy | md | PTO, sick, bereavement, jury duty | People Ops |
| 2 | POL-HR-002 | Remote Work Policy | md | Remote/hybrid, equipment, stipends | People Ops |
| 3 | POL-HR-003 | Parental Leave Policy | md | Parental leave, adoption, return-to-work | People Ops |
| 4 | POL-HR-004 | 2025 Holiday Schedule | html | Holidays, winter closure, floating holidays | People Ops |
| 5 | POL-HR-005 | Performance Review Process | txt | Reviews, ratings, promotions, PIPs | People Ops |
| 6 | POL-HR-006 | Anti-Harassment Policy | pdf | Harassment, reporting, investigations | People Ops |
| 7 | POL-FIN-001 | Expense Reimbursement Policy | md | Expenses, travel limits, professional development | Finance |
| 8 | POL-FIN-002 | Procurement Policy | md | Vendor management, approval thresholds | Finance |
| 9 | POL-FIN-003 | Business Travel Policy | txt | Travel booking, class of service, risk | Finance |
| 10 | POL-IT-001 | IT Acceptable Use Policy | html | Acceptable use, AI tools, monitoring | CIO |
| 11 | POL-IT-003 | BYOD Policy | md | Personal device use, MDM, privacy | CIO |
| 12 | POL-SEC-001 | Information Security Policy | md | Auth, devices, network, data handling | CISO |
| 13 | POL-SEC-002 | Security Incident Response Policy | md | Incident severity, response, notifications | CISO |
| 14 | POL-LEG-001 | Code of Conduct | pdf | Ethics, conflicts, anti-bribery | Legal + People Ops |
| 15 | POL-LEG-003 | Data Privacy Policy | pdf | GDPR/CCPA, data subject rights, retention | Chief Privacy Officer |

**Total:** 15 documents, ~37 pages (8 md, 2 html, 2 txt, 3 pdf) — within the 5–20 file / 30–120 page project requirement.

## Format Distribution

The corpus deliberately mixes all four required formats to demonstrate the ingestion pipeline handles each:

- **Markdown (8):** policies with structured headings — exercises content-aware chunking
- **HTML (2):** intranet-style pages with tables and metadata — exercises HTML parsing
- **Plain text (2):** simple text-only documents — exercises basic loader fallback
- **PDF (3):** formal documents generated via reportlab — exercises PDF text extraction

## Cross-References Between Documents

The corpus contains intentional cross-references between documents (e.g., the BYOD Policy references the Information Security Policy; the Travel Policy references the Expense Policy). This:

- Mirrors real-world policy corpora
- Provides natural multi-source evaluation questions
- Tests whether the retriever can pull related context across multiple files

## Testable Facts (sample, for evaluation set design)

The corpus contains many specific, verifiable facts that anchor evaluation questions. Examples:

- PTO accrual: 15/20/25 days at 0-2/3-5/6+ years tenure (POL-HR-001 §3)
- Remote work core hours: 10am–3pm local (POL-HR-002 §6)
- Home office stipend: $750 fully remote / $400 hybrid (POL-HR-002 §4)
- Birthing parent leave: 20 weeks at 100% (POL-HR-003 §3.1)
- Procurement bid threshold: 3 bids for purchases over $25,000 (POL-FIN-002 §4)
- SEV-1 incident acknowledgment: within 15 minutes (POL-SEC-002 §4)
- GDPR breach notification: within 72 hours (POL-LEG-003 §9)
- Password length: minimum 14 characters (POL-SEC-001 §3.1)
- Per diem meals: $20 breakfast / $30 lunch / $60 dinner (POL-FIN-001 §4.3)
- Performance review cycle: biannual, July and January (POL-HR-005 §2)

## Conventions

Every document includes a header block at the top with:

- **Document ID** (e.g., POL-HR-001)
- **Version** (e.g., 2.4)
- **Effective Date**
- **Owner**

These are used as canonical citations by the RAG application.

## Licensing

This corpus is original content authored for the AI Engineering Project. It is included in the repo for evaluation purposes and may be freely modified, extended, or replaced.
