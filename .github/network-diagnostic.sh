#!/bin/bash

# Network Connectivity Diagnostic Script
# Use this script to test connectivity to required services

echo "=== Network Connectivity Diagnostic ==="
echo "Testing connectivity to required services..."
echo

# Test GitHub connectivity
echo "Testing GitHub services..."
curl -I --connect-timeout 10 https://github.com && echo "✓ github.com - OK" || echo "✗ github.com - FAILED"
curl -I --connect-timeout 10 https://api.github.com && echo "✓ api.github.com - OK" || echo "✗ api.github.com - FAILED"
curl -I --connect-timeout 10 https://objects.githubusercontent.com && echo "✓ objects.githubusercontent.com - OK" || echo "✗ objects.githubusercontent.com - FAILED"
echo

# Test PyPI connectivity
echo "Testing PyPI services..."
curl -I --connect-timeout 10 https://pypi.org && echo "✓ pypi.org - OK" || echo "✗ pypi.org - FAILED"
curl -I --connect-timeout 10 https://files.pythonhosted.org && echo "✓ files.pythonhosted.org - OK" || echo "✗ files.pythonhosted.org - FAILED"
curl -I --connect-timeout 10 https://bootstrap.pypa.io/get-pip.py && echo "✓ bootstrap.pypa.io - OK" || echo "✗ bootstrap.pypa.io - FAILED"
echo

# Test other services
echo "Testing other services..."
curl -I --connect-timeout 10 https://codecov.io && echo "✓ codecov.io - OK" || echo "✗ codecov.io - FAILED"
echo

# Test DNS resolution
echo "Testing DNS resolution..."
nslookup pypi.org >/dev/null 2>&1 && echo "✓ DNS resolution for pypi.org - OK" || echo "✗ DNS resolution for pypi.org - FAILED"
nslookup github.com >/dev/null 2>&1 && echo "✓ DNS resolution for github.com - OK" || echo "✗ DNS resolution for github.com - FAILED"
echo

# Test pip connectivity
echo "Testing pip connectivity..."
python -m pip --version && echo "✓ pip is available" || echo "✗ pip is not available"
python -c "import ssl; print('✓ SSL context available')" 2>/dev/null || echo "✗ SSL context not available"
echo

echo "=== Diagnostic Complete ==="
echo "If any tests show FAILED, those services may be blocked by firewall rules."
echo "Refer to .github/firewall-allowlist.md for the complete list of required hosts."