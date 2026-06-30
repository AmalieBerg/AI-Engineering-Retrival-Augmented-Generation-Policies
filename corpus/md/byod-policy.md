# Bring Your Own Device (BYOD) Policy

**Document ID:** POL-IT-003
**Version:** 2.1
**Effective Date:** June 1, 2025
**Owner:** Chief Information Officer

## 1. Purpose

This policy defines the conditions under which employees may use personal devices to access Northwind Technologies systems and data. The objective is to enable flexibility while protecting Northwind's information assets.

## 2. Scope

This policy applies to all personal devices (smartphones, tablets, personal laptops, and wearables) used by employees, contractors, or interns to access Northwind email, calendars, instant messaging, or any other company resource.

## 3. Permitted Use Cases

Personal devices may be used to access:

- Northwind email and calendar (via the company MDM-managed mail client)
- Slack (via the official mobile app with SSO)
- Zoom (for meetings only — not for sharing or storing confidential content)
- The company intranet for general announcements

## 4. Prohibited Activities

The following are **never** permitted on personal devices:

- Accessing or storing source code
- Accessing customer personally identifiable information (PII) or payment data
- Downloading Confidential or Restricted documents (see POL-SEC-001 for classifications)
- Accessing production systems or administrative consoles
- Installing unauthorized productivity or screen-recording tools that capture company data

## 5. Enrollment Requirements

Before using a personal device for work, the employee must:

1. Enroll the device in the Northwind Mobile Device Management (MDM) system (Jamf or Intune)
2. Accept the BYOD User Agreement electronically
3. Enable device-level encryption
4. Set a passcode of at least **6 digits** (alphanumeric preferred)
5. Enable automatic OS updates
6. Allow Northwind to remotely wipe the work container in the event of loss, theft, or termination

The MDM agent will create a separate, encrypted work container on the device. Personal data outside this container will not be accessed or modified by Northwind.

## 6. Supported Operating Systems

Northwind supports BYOD enrollment for the following:

- **iOS:** Version 16 or later
- **iPadOS:** Version 16 or later
- **Android:** Version 12 or later (with Google Play Protect enabled)
- **macOS personal laptops:** Not supported for BYOD; use company-issued equipment
- **Windows personal laptops:** Not supported for BYOD; use company-issued equipment

## 7. Lost or Stolen Devices

Lost or stolen devices must be reported within **2 hours** of discovery to:

- security@northwind.example
- The IT Help Desk: 1-800-NW-HELP

Upon report, Northwind will issue a remote wipe of the work container. The employee's personal data may be affected if the device cannot be selectively wiped.

## 8. Termination of Employment

Upon termination of employment, the work container on any enrolled personal device will be remotely wiped. The employee is responsible for backing up personal data before their last day of work.

## 9. Privacy

Northwind's MDM cannot access:

- Personal photos, contacts, messages, or call history
- Personal app data outside the work container
- Location data outside of an active "Find My Device" event for a lost device

Northwind **can** monitor:

- Apps installed within the work container
- Compliance with security policies (encryption, passcode, OS version)
- Data accessed through Northwind-managed apps

## 10. Stipend

Employees who use personal devices for work-related purposes are eligible for a monthly BYOD stipend of **$50** to offset device and data costs. This stipend is in addition to any cell phone stipend provided under POL-FIN-001.

## 11. Withdrawal

Employees may withdraw from the BYOD program at any time by contacting IT to deenroll their device. Once deenrolled, the device may no longer be used to access Northwind systems.
