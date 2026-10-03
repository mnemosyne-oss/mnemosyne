# Maintainers

## Project Lead

**Abdias J. Moya (AJ)** — [@AxDSan](https://github.com/AxDSan)

- Founder and primary architect of Mnemosyne
- Final decision-making authority on core architecture: tiered memory model, recall pipeline, storage layer
- PyPI publishing, repository administration, and organization ownership
- Strategic direction and commercial entity formation

## Co-Maintainer

**Denis Hache** — [@dplush](https://github.com/dplush)

- Write access: merge PRs, push branches
- PR approval authority on designated code areas (see CODEOWNERS)
- Drives features, integrations, documentation, testing, and tooling
- Co-authorship credit on releases and publications

## Decision Framework

| Area | Authority |
|------|-----------|
| Core architecture (beam, banks, profiles, recall pipeline) | AJ (final say) |
| Features, integrations, tooling | Denis can drive independently |
| Breaking changes to public APIs | Requires consensus (AJ + Denis) |
| PyPI publishing / release | AJ only |
| Repo settings / org admin | AJ only |
| Docs, tests, CI | Either maintainer |

## Review and Merge

Either maintainer may review any pull request, including one opened by an outside
contributor. There is no routing restriction on who reviews what.

An approval means the code is correct. It does not by itself decide that the change
ships. Merge authority follows the Decision Framework above, so a pull request that
changes a supported public API, the storage schema, or a configuration contract
carries the `needs-decision` label and escalates to that table rather than being
settled in review. Review continues while it is labelled; only the merge waits.

## Governance

- All contributors must sign the [CLA](CLA.md) before contributions are accepted
- The project remains MIT licensed; the core engine is free forever
- A Delaware C-Corporation is planned as the long-term legal entity for Mnemosyne
- This document may be amended by mutual written agreement of both maintainers

---

*Last updated: 2026-08-24*
