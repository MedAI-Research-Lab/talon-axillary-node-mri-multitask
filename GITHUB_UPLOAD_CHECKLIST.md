# GitHub upload checklist

Public-release verification:
- [ ] Replace the placeholder author/repository fields in `CITATION.cff`.
- [ ] Decide and approve the final software license; replace the restrictive pre-publication `LICENSE` if appropriate.
- [ ] Institutional, ethics, consent-waiver, data-protection, and applicable journal requirements for the two deidentified clinical-image composites were reviewed and satisfied before public release.
- [ ] The public repository contains only the approved deidentified composite figures and excludes raw MRI, raster masks, clinical metadata workbooks, identity keys, per-pixel prediction arrays, and model checkpoints.
- [ ] Keep raw MRI, masks, metadata workbooks, identity keys, prediction arrays, and checkpoints outside GitHub.
- [ ] Run `python scripts/audit_public_release.py` and confirm `passed: true`.
- [ ] Run `python -m unittest discover -s tests -v` and confirm all tests pass.
- [ ] Regenerate `SHA256SUMS.txt` with `python scripts/build_release_manifest.py` after any change.
- [ ] Add the final manuscript citation/DOI after acceptance.
