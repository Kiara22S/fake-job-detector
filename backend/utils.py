import whois
import ssl
import socket
from datetime import datetime
from urllib.parse import urlparse

# Task 1: Domain Extraction
def extract_domain(url: str):
    return urlparse(url).netloc

# Task 2: WHOIS Lookup
def get_whois_info(domain: str):
    try:
        w = whois.whois(domain)
        # FAANG-level check: Is the domain less than 6 months old?
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        return {"creation_date": creation_date, "registrar": w.registrar}
    except:
        return None

# Task 3: SSL Validation
def validate_ssl(hostname: str):
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                return True # Valid SSL
    except:
        return False #missing or invalid SSL