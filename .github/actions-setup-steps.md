# GitHub Actions Setup Steps Configuration

This document provides configuration for GitHub Actions setup steps to ensure proper network access before firewall restrictions are applied.

## Pre-Firewall Network Setup

The following steps should be included at the beginning of GitHub Actions workflows to establish network connectivity before any firewall restrictions:

### Step 1: Network Connectivity Test

```yaml
- name: Configure network access
  run: |
    # Ensure network connectivity is established before any package installations
    # This step helps establish network access before firewall restrictions
    echo "Testing network connectivity..."
    curl -I --connect-timeout 10 https://pypi.org || echo "PyPI connectivity test failed"
    curl -I --connect-timeout 10 https://github.com || echo "GitHub connectivity test failed"
    curl -I --connect-timeout 10 https://codecov.io || echo "Codecov connectivity test failed"
    echo "Network configuration completed"
```

### Step 2: Configure Package Managers

```yaml
- name: Configure pip for reliability
  run: |
    # Configure pip with longer timeouts and retries for better reliability
    python -m pip config set global.timeout 60
    python -m pip config set global.retries 3
    python -m pip config set global.trusted-host "pypi.org files.pythonhosted.org"
```

### Step 3: Set Environment Variables

```yaml
- name: Set network environment variables
  run: |
    # Set environment variables for better network handling
    echo "PIP_TIMEOUT=60" >> $GITHUB_ENV
    echo "PIP_RETRIES=3" >> $GITHUB_ENV
    echo "PYTHONHTTPSVERIFY=0" >> $GITHUB_ENV
```

## Complete Workflow Example

```yaml
name: "Testing with Network Configuration"

on:
  push:
    branches: [master, main]
  pull_request:
    types: [opened, synchronize]

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    steps:
      # Step 1: Network setup BEFORE checkout
      - name: Configure network access
        run: |
          echo "Testing network connectivity..."
          curl -I --connect-timeout 10 https://pypi.org || echo "PyPI connectivity test failed"
          curl -I --connect-timeout 10 https://github.com || echo "GitHub connectivity test failed"
          echo "Network configuration completed"
      
      # Step 2: Checkout code
      - uses: actions/checkout@v4
      
      # Step 3: Setup Python with network configuration
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      # Step 4: Configure package managers
      - name: Configure pip for reliability
        run: |
          python -m pip config set global.timeout 60
          python -m pip config set global.retries 3
          python -m pip config set global.trusted-host "pypi.org files.pythonhosted.org"
      
      # Step 5: Install dependencies with timeouts
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -r requirements.txt
        timeout-minutes: 10
```

## Network Troubleshooting Steps

If network issues persist, add these diagnostic steps:

```yaml
- name: Network diagnostics
  run: |
    echo "=== Network Diagnostics ==="
    echo "Testing DNS resolution..."
    nslookup pypi.org
    nslookup github.com
    
    echo "Testing HTTP connectivity..."
    curl -v --connect-timeout 30 https://pypi.org/simple/ || true
    
    echo "Testing package installation..."
    python -m pip install --dry-run --no-deps requests || true
```

## Best Practices

1. **Early Network Setup**: Always configure network access before any package installations
2. **Timeouts**: Set appropriate timeouts for all network operations
3. **Retries**: Configure retry logic for package managers
4. **Diagnostics**: Include network diagnostic steps for troubleshooting
5. **Trusted Hosts**: Configure trusted hosts for package repositories

## Firewall Configuration Reference

For the complete list of hosts that need firewall access, see [firewall-allowlist.md](./firewall-allowlist.md).