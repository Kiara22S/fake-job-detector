class RiskEngine:
    """
    The 'Voice' of the system. 
    Translates ML features into a high-fidelity security report.
    """
    def __init__(self):
        # The 'Threat Registry' - Centralized Feature -> Sentence mapping
        self.THREAT_MAP = {
            "has_payment_request": "Financial Solicitation: Request for upfront fees or deposits.",
            "gmail_domain": "Identity Anomaly: Use of non-corporate/free email domain.",
            "new_domain": "Technical Risk: Links to a newly registered or untrusted domain.",
            "contains_urgent_words": "Psychological Trigger: Use of predatory urgency language.",
            "location_missing": "Operational Anomaly: Missing verifiable physical headquarters."
        }

    def generate_report(self, processed_row):
        """
        Orchestrates the reasons and safety advice into a standard JSON contract.
        """
        # Efficient list comprehension for O(1) lookups
        reasons = [
            desc for flag, desc in self.THREAT_MAP.items() 
            if processed_row.get(flag) == 1
        ]
        
        category = processed_row.get("risk_category", "Low")
        
        return {
            "risk_assessment": {
                "level": category,
                "score": float(processed_row.get("risk_score", 0)),
                "flags_detected": len(reasons)
            },
            "investigation_findings": reasons if reasons else ["Standard professional patterns detected."],
            "mitigation_strategy": self._get_protocol(category)
        }

    def _get_protocol(self, level):
        # Safety guardrails based on risk level
        protocols = {
            "High": "CRITICAL: Cease communication. Do not share documents or funds.",
            "Medium": "VERIFY: Cross-reference company credentials on LinkedIn.",
            "Low": "MONITOR: Proceed but maintain standard security awareness."
        }
        return protocols.get(level, "Follow standard safety guidelines.")