# GitHub upload checklist

Before making the repository public:

- [ ] Replace the placeholder author/repository fields in `CITATION.cff`.
- [ ] Decide and approve the final software license; replace the restrictive pre-publication `LICENSE` if appropriate.
- [ ] Confirm institutional, ethics, consent/waiver, and journal permission for the two deidentified clinical-image composites listed in `docs/DATA_GOVERNANCE.md`.
- [ ] If clinical-image publication permission is uncertain, remove Figure 3 and Supplementary Figure S9 before upload.
- [ ] Keep raw MRI, masks, metadata workbooks, identity keys, prediction arrays, and checkpoints outside GitHub.
- [ ] Run `python scripts/audit_public_release.py` and confirm `passed: true`.
- [ ] Run `python -m unittest discover -s tests -v` and confirm all tests pass.
- [ ] Regenerate `SHA256SUMS.txt` with `python scripts/build_release_manifest.py` after any change.
- [ ] Add the final manuscript citation/DOI after acceptance.
