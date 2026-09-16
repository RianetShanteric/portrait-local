# Security Policy

PortraitLocal is a local-first application for processing personal
photographs. Security reports should avoid exposing photographs, model
weights, credentials or other private data.

## Supported Scope

The current public scope is the `0.4.0` source snapshot and the latest
repository state on `main`. The project is a portfolio release and does not
promise a guaranteed response time or a supported production deployment.

## Reporting a Vulnerability

Please do not disclose a suspected vulnerability in a public issue.

If GitHub private vulnerability reporting is enabled for this repository, use
that channel. Otherwise, contact the repository owner through GitHub before
making the issue public. Include a concise description, impact, affected file
or command, and reproduction steps that use synthetic or public data only.

Do not attach user photographs, downloaded model weights, API keys, access
tokens, private keys or complete local environment archives.

## Privacy and Local Data

PortraitLocal is designed to process photographs locally. Keep `data/input`,
`data/output`, runtime directories, virtual environments and downloaded model
artifacts out of commits and issue attachments. The repository intentionally
contains no user photographs, model weights or runtime data.
