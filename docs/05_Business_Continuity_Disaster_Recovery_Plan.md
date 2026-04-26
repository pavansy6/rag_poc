# Business Continuity and Disaster Recovery (BCDR) Plan
## Organization: [Company Name]
**Document Owner:** CISO / VP of Operations
**Version:** 1.1

### 1. Purpose
To ensure the organization can maintain operations or quickly resume critical functions following a disruptive event.

### 2. Scope
This BCDR plan covers all mission-critical infrastructure, including our primary cloud hosting environment and customer portals.

### 3. Recovery Objectives
- **Recovery Time Objective (RTO):** Critical platforms must be restored within 4 hours.
- **Recovery Point Objective (RPO):** Maximum allowable data loss for critical systems is 1 hour.

### 4. Disaster Scenarios and Failover
- **Primary Data Center Outage:** Automatic DNS failover routing traffic to the secondary geographic availability zone.
- **Ransomware Event:** Immediate isolation of the environment and restoration from offline, immutable backups.

### 5. Testing
The BCDR plan will be tested bi-annually. Testing will include full technical failover simulations and tabletop communication exercises.
