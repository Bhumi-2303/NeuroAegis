# NeuroAegis Privacy Policy

**Last Updated:** August 2026  
**Version:** 2.0

---

## 1. Purpose & Scope

This Privacy Policy explains how NeuroAegis ("the Platform") collects, processes, stores, and protects personal and health-related data. It applies to all users of the NeuroAegis web application, API, and associated services.

NeuroAegis is an **experimental research tool** — it is **not** an FDA/CE-cleared medical device. This policy describes our data practices in the context of a research platform.

---

## 2. Data We Collect

| Data Category | Examples | Basis |
|---|---|---|
| **Patient Demographics** | Name, age, gender, weight, height | Clinician-provided via form input |
| **Medical History** | Free-text clinical notes, vital signs | Clinician-provided via form input |
| **EEG Data** | Uploaded `.csv`, `.edf`, `.txt` files | Clinician-uploaded for analysis |
| **Prediction Results** | Seizure probability, SHAP explanations, model metadata | System-generated during analysis |
| **User Accounts** | Username, hashed password, role | Created by system administrators |

**We do NOT collect:**
- Insurance or billing information
- Social Security Numbers or government IDs
- Location data or browser fingerprints
- Data from minors without authorized representative consent

---

## 3. How We Use Data

- **EEG Analysis:** Uploaded EEG files are processed through our ML pipeline to generate seizure predictions and explainability outputs.
- **Patient Records:** Demographic and medical history data are stored alongside prediction results to enable report generation and audit trails.
- **Authentication:** User credentials are used solely for access control. Passwords are hashed using bcrypt and are never stored in plaintext.

---

## 4. Data Storage & Retention

- **Patient data and prediction results** are stored in a PostgreSQL database.
- **Raw EEG files** are processed in memory and are NOT permanently stored after analysis completes. Temporary files created during `.edf` processing are deleted immediately after use.
- **Data retention:** Patient records are retained until explicitly deleted by an administrator. Users may request deletion at any time (see Section 7).

---

## 5. Data Security

| Measure | Implementation |
|---|---|
| **Encryption in transit** | TLS 1.2+ via Nginx reverse proxy |
| **Authentication** | JWT-based with bcrypt password hashing |
| **Authorization** | Role-based access control (clinician, researcher, admin) |
| **Password storage** | bcrypt hash — plaintext passwords are never stored |
| **Security headers** | X-Content-Type-Options, X-Frame-Options, HSTS, Referrer-Policy |
| **CORS** | Restricted to configured origins |
| **Input validation** | Filename sanitization, upload size limits, schema validation |

---

## 6. Informed Consent

Before any patient data is submitted for analysis, the clinician must:

1. Confirm they have legal authority to process the patient's data.
2. Acknowledge that NeuroAegis is a research tool, not a clinical device.
3. Explicitly check an informed consent checkbox in the application.

Consent is recorded with a timestamp in the patient record.

---

## 7. Data Subject Rights (GDPR Article 15–22)

Data subjects (or their authorized representatives) have the right to:

- **Access** their stored data by contacting the system administrator.
- **Rectification** of inaccurate data.
- **Erasure** ("Right to be Forgotten") — administrators can permanently delete all patient data and associated prediction records via the `DELETE /api/v1/data/patient/{id}` endpoint.
- **Restriction of processing** — by contacting the data controller.

To exercise any of these rights, contact the system administrator.

---

## 8. Third-Party Data Sharing

NeuroAegis does **not** share, sell, or transfer patient data to any third party. All processing occurs on infrastructure controlled by the deploying organization.

---

## 9. Medical Disclaimer

> **NeuroAegis is NOT a medical device.** It is an experimental research tool designed for research and demonstration purposes only. It has not been validated in clinical trials and is not approved by the FDA, EMA, or any regulatory body. It must not be used as the sole basis for diagnosis, treatment, or medication decisions. Always consult a qualified healthcare professional for medical advice and interpretation of EEG data.

---

## 10. Changes to This Policy

We may update this Privacy Policy to reflect changes in our practices or legal requirements. The "Last Updated" date at the top of this document will be revised accordingly.

---

## 11. Contact

For privacy inquiries, data deletion requests, or to report a security concern, contact the system administrator of your NeuroAegis deployment.
