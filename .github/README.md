# Network and Firewall Configuration

This directory contains configuration files and documentation for resolving GitHub Copilot firewall permission issues.

## Files

- **[firewall-allowlist.md](./firewall-allowlist.md)** - Complete list of URLs and hosts that need firewall access
- **[actions-setup-steps.md](./actions-setup-steps.md)** - GitHub Actions configuration for network setup

## Quick Setup

### For System Administrators

Add these domains to your firewall allow list for HTTPS (port 443):

```
github.com
api.github.com
pypi.org
files.pythonhosted.org
codecov.io
objects.githubusercontent.com
raw.githubusercontent.com
```

### For Developers

The GitHub Actions workflow has been updated to:

1. Test network connectivity before package installations
2. Configure pip with longer timeouts and retries
3. Set appropriate timeouts for all network operations

## Troubleshooting

If you see errors like:
- `ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.`
- `Failed to connect to github.com`

Check that your firewall allows access to the domains listed in [firewall-allowlist.md](./firewall-allowlist.md).

## Testing Network Access

Run this command to test connectivity:

```bash
curl -I https://pypi.org && echo "PyPI OK" || echo "PyPI BLOCKED"
curl -I https://github.com && echo "GitHub OK" || echo "GitHub BLOCKED"
```

## GitHub Actions Setup

The workflow now includes these network-aware setup steps:

1. **Pre-flight network test** - Tests connectivity before any installations
2. **Pip configuration** - Sets timeouts and retries for better reliability  
3. **Step timeouts** - Prevents hanging on network issues
4. **Job timeout** - Overall job timeout for resource management

For detailed configuration, see [actions-setup-steps.md](./actions-setup-steps.md).