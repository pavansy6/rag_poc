# Cloud Security Standard
## Organization: [Company Name]
**Document Owner:** Cloud Architecture Team / CISO
**Version:** 1.0

### 1. Overview
This standard defines the baseline security requirements for all cloud infrastructure (IaaS, PaaS, SaaS) utilized by the organization.

### 2. Baseline Configuration Requirements
- **Public Exposure:** No databases or storage buckets may be publicly accessible over the internet without explicit CISO approval.
- **Encryption:** All storage volumes, object storage, and databases must be encrypted using customer-managed keys (CMK) via the cloud provider's Key Management Service (KMS).

### 3. Identity in the Cloud
- Avoid long-lived access keys. Instead, use IAM roles and temporary credentials.
- All administrative console access must require MFA.

### 4. Logging and Monitoring
- Cloud provider auditing logs must be enabled in all regions and forwarded to the central SIEM.
- Alerts must be configured for unauthorized configuration changes, anomalous billing spikes, and root account logins.
