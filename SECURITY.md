# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**DO NOT** open a public GitHub issue for security vulnerabilities.

**Please email:** dfr@esmad.ipp.pt with subject line "SECURITY: <brief description>"

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

**Response time:** Within 48 hours

## Security Best Practices

- Always install from official PyPI: `pip install efficient-plackett-luce`
- Verify package integrity
- Don't load pickle files from untrusted sources (model.save/load)
- Sanitize user input before passing to the model
