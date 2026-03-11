class RiskEngine:
    """
    The 'Voice' of the application. 
    Translates raw data into human-understandable security insights.
    """
    def __init__(self):
        # Professional-grade mapping of data flags to human warnings
        self.flag_map = {
            "has_payment_request": "The listing mentions upfront fees or deposits.",
            "gmail_domain": "The contact uses a free/personal email address.",
            "new_domain": "The post contains links to unverified domains.",
            "contains_urgent_words": "Language used creates predatory urgency.",
            "location_missing": "No physical corporate office location provided."
        }

    def get_comprehensive_report(self, processed_row):
        """
        Orchestrates the reasons and safety advice into a standardized report.
        """
        reasons = [msg for flag, msg in self.flag_map.items() if processed_row.get(flag) == 1]
        category = processed_row.get("risk_category", "Low")
        
        return {
            "risk_level": category,
            "risk_score": float(processed_row.get("risk_score", 0)),
            "red_flags": reasons if reasons else ["No high-risk patterns detected."],
            "safety_protocol": self._generate_advice(category)
        }

    def _generate_advice(self, category):
        protocols = {
            "High": "CRITICAL: High fraud correlation. Do not share documents or funds.",
            "Medium": "VERIFY: Cross-reference company details on LinkedIn/Glassdoor.",
            "Low": "SAFE: Standard patterns detected. Maintain general awareness."
        }
        return protocols.get(category, "Stay alert and follow standard job-search safety.")