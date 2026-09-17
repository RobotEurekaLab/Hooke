# Repository conventions

## Generated experiment artifacts

- Write new screenshots, videos, render frames, full experiment records, and raw logs under the ignored `temp/` directory. Normal serializer outputs may use the already ignored `Hooke/logs/` directory.
- Set output and media destinations explicitly when a generation script defaults to a tracked directory.
- Preserve the existing tracked `docs/assets/` media used by documentation and demonstration pages.
- Commit source code, documentation, and concise validation summaries as needed. Keep new generated media and raw experiment data out of Git unless the user explicitly requests their inclusion.
