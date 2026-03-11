### API Response Contract (v1.0.2)
All endpoints return a standardized JSON envelope:
- `success`: Boolean status
- `data`:
    - `verdict`: "Real" or "Fake"
    - `risk_details`: Human-readable red flags and safety protocols
    - `scoring`: Weighted confidence metrics
    - `lineage`: Model version tracking