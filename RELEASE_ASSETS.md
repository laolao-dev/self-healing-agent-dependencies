# Release assets — v0.1.0

- `self-healing-agent-dependencies-v0.1.0.zip`: source/documentation archive generated from the tagged Git commit, with root `self-healing-agent-dependencies-v0.1.0/`. Contains only files tracked in this public repository.
- `self-healing-agent-dependencies-v0.1.0.zip.sha256`: SHA256 of the complete archive.

No dependency implementation, database, binary application, model, artwork, private log or internal batch report is attached. GitHub-generated source archives are additional platform-provided downloads; the attached ZIP is the explicitly checksum-verified asset.

`PUBLIC_FILE_MANIFEST.json` lists every payload file with byte count and SHA256, excluding itself and SHA256SUMS to avoid recursive hashes. SHA256SUMS covers payload plus PUBLIC_FILE_MANIFEST.json, excluding only itself. Hashes refer to Git blob / attached ZIP bytes, not shell-dependent checkout line endings.

Download both attachments, verify the archive hash with `sha256sum -c self-healing-agent-dependencies-v0.1.0.zip.sha256` (or PowerShell `Get-FileHash -Algorithm SHA256`), extract it, then run `sha256sum -c SHA256SUMS` from the archive root. The integrity manifest is not a cryptographic publisher signature.
