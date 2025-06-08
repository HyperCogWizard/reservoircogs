#!/usr/bin/env python
"""
Network configuration validation test.
This script validates that network configuration allows access to required services.
"""

import urllib.request
import urllib.error
import sys
import time

def test_url(url, timeout=10):
    """Test if a URL is accessible"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.getcode() == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False

def main():
    """Run network configuration validation tests"""
    print("=== Network Configuration Validation ===")
    
    # Test URLs
    test_urls = [
        "https://pypi.org",
        "https://files.pythonhosted.org", 
        "https://github.com",
        "https://objects.githubusercontent.com"
    ]
    
    all_passed = True
    
    for url in test_urls:
        print(f"Testing {url}...", end=" ")
        if test_url(url):
            print("✓ OK")
        else:
            print("✗ FAILED")
            all_passed = False
    
    if all_passed:
        print("\n✓ All network tests passed!")
        return 0
    else:
        print("\n✗ Some network tests failed!")
        print("Check firewall configuration and allowlist.")
        return 1

if __name__ == "__main__":
    sys.exit(main())