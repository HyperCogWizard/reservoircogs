# Firewall Allow List Configuration

This document lists the URLs and hosts that need to be added to firewall allow lists for GitHub Copilot and CI/CD processes to function properly.

## Required Hosts for GitHub Actions

### GitHub Services
- `github.com` - Main GitHub platform
- `api.github.com` - GitHub API access
- `uploads.github.com` - File uploads to GitHub
- `objects.githubusercontent.com` - Git objects and raw content
- `raw.githubusercontent.com` - Raw file content from repositories
- `codeload.github.com` - Repository downloads
- `actions-results.githubusercontent.com` - GitHub Actions results
- `pipelines.actions.githubusercontent.com` - GitHub Actions pipelines
- `*.actions.githubusercontent.com` - GitHub Actions wildcard domains

### Python Package Index (PyPI)
- `pypi.org` - Python Package Index main site
- `*.pypi.org` - PyPI wildcard domains
- `files.pythonhosted.org` - Python package files
- `pythonhosted.org` - Python package hosting
- `*.pythonhosted.org` - PyPI hosting wildcard
- `bootstrap.pypa.io` - pip bootstrap scripts
- `get-pip.py` - pip installation script

### Content Delivery Networks (CDNs)
- `fastly.com` - Fastly CDN (used by PyPI)
- `*.fastly.com` - Fastly CDN wildcard
- `*.fastlylb.net` - Fastly load balancer
- `cloudflare.com` - Cloudflare CDN
- `*.cloudflare.com` - Cloudflare wildcard
- `*.akamai.net` - Akamai CDN
- `*.akamaized.net` - Akamai CDN

### Code Coverage Services
- `codecov.io` - Codecov service
- `uploader.codecov.io` - Codecov uploader service
- `*.codecov.io` - Codecov wildcard

### C++ Package Repositories
- `archive.ubuntu.com` - Ubuntu package archives
- `security.ubuntu.com` - Ubuntu security updates
- `*.ubuntu.com` - Ubuntu wildcard
- `ppa.launchpad.net` - Personal Package Archives
- `launchpad.net` - Launchpad platform
- `*.launchpad.net` - Launchpad wildcard
- `packages.microsoft.com` - Microsoft packages
- `*.packages.microsoft.com` - Microsoft packages wildcard

### OpenCog Dependencies
- `github.com/opencog/*` - OpenCog repositories
- `objects.githubusercontent.com/opencog/*` - OpenCog raw content
- `releases.ubuntu.com` - Ubuntu releases
- `*.releases.ubuntu.com` - Ubuntu releases wildcard

### Additional Package Managers
- `npmjs.org` - NPM registry (if Node.js dependencies exist)
- `registry.npmjs.org` - NPM registry
- `*.npmjs.org` - NPM wildcard
- `yarnpkg.com` - Yarn package manager
- `dl.yarnpkg.com` - Yarn downloads

### DNS Servers
- `8.8.8.8` - Google DNS
- `8.8.4.4` - Google DNS secondary
- `1.1.1.1` - Cloudflare DNS
- `1.0.0.1` - Cloudflare DNS secondary
- `208.67.222.222` - OpenDNS
- `208.67.220.220` - OpenDNS secondary

## Port Requirements

- **HTTPS (443)** - All HTTPS traffic
- **HTTP (80)** - Some package repositories may use HTTP
- **DNS (53)** - DNS resolution (both TCP and UDP)
- **SSH (22)** - Git operations over SSH
- **Git (9418)** - Git protocol (if used)

## IP Ranges (if domain filtering is not available)

If your firewall requires IP addresses instead of domains, you may need to allow these service IP ranges:

### GitHub IP Ranges
- `140.82.112.0/20` - GitHub main services
- `185.199.108.0/22` - GitHub Pages and assets
- `192.30.252.0/22` - GitHub API and web services

### PyPI/Fastly IP Ranges
- Fastly CDN IP ranges vary and should be obtained from Fastly's official documentation

**Note**: IP ranges can change frequently. Domain-based filtering is strongly recommended.

## Configuration Instructions

### For Corporate Firewalls
Add these domains to your firewall's allow list for outbound HTTPS traffic on port 443 and HTTP traffic on port 80.

### For GitHub Actions Network Policies
Configure the network access in the Actions workflow before any package installation steps.

### DNS Resolution
Ensure that DNS resolution works for all listed domains.

## Verification Commands

Test network connectivity to these services:

```bash
# Test PyPI connectivity
curl -I https://pypi.org
curl -I https://files.pythonhosted.org

# Test GitHub connectivity  
curl -I https://api.github.com
curl -I https://github.com

# Test Codecov connectivity
curl -I https://codecov.io

# Test CDN connectivity
curl -I https://bootstrap.pypa.io
```

## Common Timeout Solutions

If you experience persistent timeouts:

1. **Increase timeout values**: Configure longer timeouts for package installations
2. **Use mirrors**: Configure pip to use mirror repositories
3. **Retry logic**: Implement retry mechanisms for failed requests
4. **Caching**: Use package caching to reduce external requests

## Troubleshooting

If you encounter timeout errors like:
- `ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.`
- `Failed to connect to github.com`
- `Connection timed out to files.pythonhosted.org`

Check that the above hosts are allowed through your firewall configuration and that DNS resolution is working properly.