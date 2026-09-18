# Before publishing changes to this repository

- Review every tracked file with `git diff --cached` and `git grep` for passwords, API keys, tokens, private hostnames/IPs, SSH paths, user names, internal project names, private model aliases, document text, and raw benchmark responses.
- Keep model caches, raw results, logs, `.env`, credentials, and machine-specific Compose overrides out of Git. `.gitignore` helps but is not a substitute for inspection.
- Recheck every number against the corresponding local benchmark JSON; do not mix decoder-only throughput with end-to-end or aggregate throughput.
- Confirm the intended registry **manifest digest**, not just the local Docker image ID. Never cite a mutable `nightly` tag as exact provenance.
- Test the example Compose file on an isolated host or maintenance window before calling it a proven launch recipe. As of this draft, it is a sanitized adaptation of tested flags, not a fresh run of the exact Compose file.
- Keep the license scope clear: MIT for code/configuration examples and CC BY 4.0 for written documentation and benchmark reports. Check third-party material separately.
- Review whether the hardware and benchmark descriptions are comfortable to share publicly; omit optional host specifics if necessary.
- Review the destination repository and visibility before each push. Publishing this cookbook does not publish production configurations or raw data.
