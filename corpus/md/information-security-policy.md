# Information Security Policy

**Document ID:** POL-SEC-001
**Version:** 5.2
**Effective Date:** February 1, 2025
**Owner:** Chief Information Security Officer

## 1. Purpose

This policy establishes the security requirements that all employees, contractors, and third parties must follow when accessing Northwind Technologies information systems and data.

## 2. Information Classification

Northwind classifies information into four tiers:

- **Public:** Information that may be shared freely (marketing material, public announcements)
- **Internal:** Information intended for employees only (org charts, internal memos)
- **Confidential:** Sensitive business information (financials, strategy documents, customer lists)
- **Restricted:** Highly sensitive information (PII, payment data, source code, security keys, M&A documents)

Each piece of information must be tagged with its classification level. When in doubt, treat as Confidential.

## 3. Authentication and Access

### 3.1 Passwords

All Northwind accounts require passwords that are:

- Minimum **14 characters** in length
- A mix of uppercase, lowercase, numbers, and symbols
- Not reused from any prior account
- Changed every **180 days** for systems handling Restricted data; passwords for other systems do not expire on a schedule but must be changed immediately upon any suspected compromise

The use of the company password manager (1Password Business) is required for all employees.

### 3.2 Multi-Factor Authentication

MFA is mandatory for:

- All accounts with access to Confidential or Restricted information
- Administrative access to any production system
- VPN connections
- Email accounts

Approved second factors are: hardware security keys (YubiKey), Okta Verify push notifications, and TOTP authenticator apps. SMS-based MFA is **not** permitted for systems handling Restricted data.

### 3.3 Single Sign-On (SSO)

All eligible applications must use Okta SSO. Local credentials are prohibited for systems integrated with Okta.

## 4. Device Security

### 4.1 Company Devices

All company-issued laptops must:

- Run Northwind's standard MDM agent (Jamf for macOS, Intune for Windows)
- Have full-disk encryption enabled (FileVault or BitLocker)
- Auto-lock after **5 minutes** of inactivity
- Receive security updates within **14 days** of release

### 4.2 Personal Devices

Personal devices may only access company data through approved methods described in the BYOD Policy (POL-IT-003). Source code may not be accessed from personal devices under any circumstances.

## 5. Network Security

### 5.1 VPN

The company VPN (Cisco AnyConnect) must be used when:

- Accessing any internal-only application from outside a Northwind office
- Working from public Wi-Fi networks
- Handling Confidential or Restricted data

### 5.2 Public Wi-Fi

Public Wi-Fi (cafes, airports, hotels) is permitted only with VPN enabled. Connecting to open networks without VPN is prohibited.

## 6. Email and Phishing

Employees must:

- Report suspicious emails to security@northwind.example or via the "Report Phish" button in Outlook
- Not click links or open attachments from unverified senders
- Not forward Confidential or Restricted information to personal email accounts
- Complete the **quarterly phishing training** assigned by the Security team

Failure to complete required security training within 30 days of assignment may result in temporary account suspension.

## 7. Incident Reporting

Suspected security incidents must be reported within **1 hour** of discovery to:

- security-incident@northwind.example (email)
- #security-incidents (Slack)
- The on-call security engineer via PagerDuty for critical incidents

See the Incident Response Policy (POL-SEC-002) for detailed response procedures.

## 8. Data Handling

- Restricted data must be encrypted at rest and in transit using AES-256 or equivalent
- USB drives are prohibited; use approved cloud transfer methods
- Hard copies of Confidential or Restricted material must be stored in locked cabinets and shredded when no longer needed
- Customer data may not be downloaded to local machines without an approved data access ticket

## 9. Third-Party and Vendor Security

Any vendor that will process or store Northwind data must:

- Complete a Security Review (managed by the Security GRC team)
- Sign a Data Processing Agreement (DPA) if handling personal data
- Maintain at minimum SOC 2 Type II certification or equivalent

## 10. Enforcement

Violations of this policy may result in disciplinary action up to and including termination, and may be reported to law enforcement where applicable. Inadvertent violations should be self-reported to security@northwind.example.
