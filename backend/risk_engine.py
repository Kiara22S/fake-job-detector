import datetime

class RiskEngine:
    """
    The 'Voice' of the system. 
    Translates ML features into a high-fidelity security report.
    """
    def __init__(self):
        # 1. Explanation Mapping: Feature -> Sentence
        self.THREAT_MAP = {
            "has_payment_request": "Financial Solicitation: Request for upfront fees or deposits.",
            "gmail_domain": "Identity Anomaly: Use of non-corporate/free email domain.",
            "new_domain": "Technical Risk: Links to a newly registered or untrusted domain.",
            "contains_urgent_words": "Psychological Trigger: Use of predatory urgency language.",
            "location_missing": "Operational Anomaly: Missing verifiable physical headquarters."
        }

    def generate_report(self, processed_row, confidence_score):
        """
        Orchestrates reasons and safety advice into a FAANG-level JSON contract.
        """
        # Logic: Extract active sentences based on binary features (1 = flag detected)
        reasons = [
            desc for flag, desc in self.THREAT_MAP.items() 
            if processed_row.get(flag) == 1
        ]
        
        # Determine risk level based on your 1:8 weighted model's confidence
        # FAANG Rule: If confidence > 80%, force category to 'High'
        category = "Low"
        if confidence_score > 0.80 or len(reasons) >= 3:
            category = "High"
        elif confidence_score > 0.40:
            category = "Medium"
        
        return {
            "analysis_metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "confidence": f"{round(confidence_score * 100, 2)}%"
            },
            "risk_assessment": {
                "level": category,
                "flags_detected": len(reasons)
            },
            "investigation_findings": reasons if reasons else ["Standard professional patterns detected."],
            "mitigation_strategy": self._get_protocol(category)
        }

    def _get_protocol(self, level):
        protocols = {
            "High": "CRITICAL: High-risk signatures detected. Cease communication immediately.",
            "Medium": "VERIFY: Potential risk. Cross-reference company credentials on LinkedIn.",
            "Low": "MONITOR: Proceed but maintain standard security awareness."
        }
        return protocols.get(level, "Follow standard safety guidelines.")