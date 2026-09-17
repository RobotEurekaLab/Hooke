# Repository conventions

## Generated experiment artifacts

- Write new screenshots, videos, render frames, full experiment records, and raw logs under the ignored `temp/` directory. Normal serializer outputs may use the already ignored `Hooke/logs/` directory.
- Set output and media destinations explicitly when a generation script defaults to a tracked directory.
- Keep only curated documentation illustrations and media required by demonstration pages in `docs/assets/`. Archive intermediate captures under `temp/`; update documentation links when removing tracked images. Historical validation reports retain their original artifact paths and hashes.
- New files under `docs/assets/` are ignored by default. Include selected media with `git add -f` only when the user explicitly requests their inclusion.
- Commit source code, documentation, and concise validation summaries as needed. Keep new generated media and raw experiment data out of Git unless the user explicitly requests their inclusion.
