# Security Incident Response Policy

**Document ID:** POL-SEC-002
**Version:** 4.1
**Effective Date:** March 1, 2025
**Owner:** Chief Information Security Officer

## 1. Purpose

This policy defines how Northwind Technologies identifies, responds to, and recovers from security incidents. The objective is to minimize impact, preserve evidence, and meet regulatory obligations.

## 2. Definitions

- **Security Event:** An observed occurrence that may indicate a security concern (e.g., a failed login attempt).
- **Security Incident:** A confirmed event that has resulted, or could result, in unauthorized access, disclosure, modification, or destruction of Northwind data or systems.
- **Breach:** A security incident involving confirmed unauthorized access to or disclosure of personal data.

## 3. Incident Severity Levels

| Severity | Definition | Examples |
|----------|------------|----------|
| **SEV-1 (Critical)** | Major impact to systems or data; immediate response required | Customer data breach, ransomware, production outage with data risk |
| **SEV-2 (High)** | Significant risk but contained; rapid response required | Successful phishing with credential compromise, malware on a single endpoint |
| **SEV-3 (Medium)** | Limited risk or contained; standard response | Unauthorized access attempt blocked by controls, minor policy violation |
| **SEV-4 (Low)** | Minor concern; routine investigation | Single failed login from unusual location, suspicious but benign email |

## 4. Response Times

The Security Operations Center (SOC) is staffed 24/7 and will acknowledge incidents within the following timelines:

- **SEV-1:** Acknowledged within **15 minutes**; response team assembled within 30 minutes
- **SEV-2:** Acknowledged within **1 hour**; response team assembled within 4 hours
- **SEV-3:** Acknowledged within **4 hours**
- **SEV-4:** Acknowledged within **1 business day**

## 5. Reporting an Incident

Employees who suspect a security incident must report it within **1 hour** of discovery through any of the following:

- Email: security-incident@northwind.example
- Slack: #security-incidents
- PagerDuty: page the on-call security engineer (for SEV-1 or SEV-2)
- Phone: 1-800-NW-SECURE (24/7 hotline)

Do not attempt to remediate the incident yourself. Preserving evidence is critical.

## 6. Incident Response Process

Northwind follows a six-phase incident response process aligned with NIST SP 800-61:

1. **Preparation:** Maintain readiness through training, tooling, and runbooks
2. **Identification:** Detect and triage the incident
3. **Containment:** Isolate affected systems to prevent spread
4. **Eradication:** Remove the threat (e.g., malware, attacker access)
5. **Recovery:** Restore systems to normal operation with monitoring
6. **Lessons Learned:** Conduct a post-incident review within **10 business days** of resolution

## 7. Incident Response Team (IRT)

The IRT is led by the **Incident Commander (IC)**, designated by the CISO. The IRT may include:

- Security Engineer (lead investigator)
- Platform Engineer (system access and recovery)
- Legal Counsel
- Communications Lead
- Privacy Officer (for incidents involving personal data)
- External counsel and forensics firm (engaged as needed)

For SEV-1 incidents, the CEO, CFO, and General Counsel are notified within 1 hour of confirmation.

## 8. Communications

External communications during an incident are managed exclusively by the Communications Lead in coordination with Legal. Employees must not publicly discuss the incident, including on social media, until cleared by Communications.

Internal communications are managed through the dedicated incident Slack channel created for each SEV-1 or SEV-2 incident.

## 9. Regulatory Notifications

For incidents involving personal data:

- **GDPR (EU residents):** Notification to the supervisory authority within **72 hours** of awareness
- **CCPA (California residents):** Notification to affected individuals "in the most expedient time possible"
- **HIPAA (if applicable):** Notification within **60 days** of discovery
- **SEC (material incidents for public reporting):** Form 8-K filing within **4 business days** of determining materiality

The Privacy Officer manages regulatory notifications in coordination with Legal.

## 10. Documentation

The IRT must maintain a contemporaneous record of all actions taken during an incident, including:

- Timeline of events
- Systems and data affected
- Actions taken (with timestamps and operator identity)
- Decisions made and rationale
- Communications sent

Incident records are retained for a minimum of **7 years**.

## 11. Tabletop Exercises

The Security team conducts at least **two tabletop exercises** per year simulating realistic incidents. Participation is mandatory for the IRT and recommended for senior leadership.

## 12. Post-Incident Review

Every SEV-1 and SEV-2 incident concludes with a blameless post-incident review (PIR). The PIR identifies root causes, contributing factors, and concrete action items with owners and deadlines. Action items must be tracked to closure.
