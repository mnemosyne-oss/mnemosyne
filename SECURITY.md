# Security Policy

## Supported Versions

Only the latest release receives security updates. Mnemosyne is moving fast
and we can't maintain backports for older versions.

| Version | Supported          |
|---------|--------------------|
| latest  | ✅                 |
| older   | ❌                 |

## Reporting a Vulnerability

Mnemosyne stores agent memory and conversation data. A vulnerability could
expose sensitive information. If you find something, we want to know.

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead:

- **Open a [GitHub Issue](https://github.com/AxDSan/mnemosyne/issues) marked as sensitive** 
  (GitHub lets you flag an issue as confidential when filing it)
- Or **DM @AxDSan** directly on GitHub or Discord

We aim to acknowledge receipt within 48 hours and issue a fix within 7 days
for confirmed vulnerabilities. You'll be credited in the release notes
(unless you prefer to stay anonymous).

## What We Consider a Security Issue

- Remote code execution through memory operations
- Data leakage between user namespaces / group_ids
- Injection attacks through episode content
- Credential exposure in logs or error messages
- Unsafe deserialization in import paths

## What Is Not a Security Issue

- Missing features or performance bottlenecks
- Known limitations documented in the README
- Vulnerabilities in third-party dependencies (report to them directly)

## Disclosure Policy

We follow a 30-day embargo period after a fix is released before publishing
full details. This gives users time to update.
