# Mnemosyne Maintainers

## Roles

### Lead Maintainer (@AxDSan)
- Final say on all technical and project decisions
- Controls repository settings, access, and releases
- Responsible for PyPI publishing and GitHub releases
- Sets project vision and long-term roadmap

### Co-Maintainer
- Merge access on approved areas (defined in CODEOWNERS)
- Can approve pull requests but cannot merge releases or change repository settings
- Input on architectural decisions and project direction
- Responsible for specific subsystems as defined in CODEOWNERS
- Must follow consensus process for major changes

### Contributor
- Fork and PR model
- No direct write access to main repository
- Can contribute code, documentation, issues, and discussions

## Adding a Co-Maintainer

1. **Nomination**: Lead maintainer identifies candidate or community nominates via issue/discussion
2. **Trial Period (30 days)**: Grant write access to evaluate contributions
3. **Evaluation**: Lead maintainer assesses:
   - Code quality and consistency
   - Communication and collaboration
   - Alignment with project values and principles
   - Ability to ship and maintain features
4. **Decision**:
   - If working: promote to co-maintainer with announced responsibilities
   - If not: revert access with feedback, maintain contributor relationship
5. **Announcement**: Update MAINTAINERS.md and notify community

## Removing Access

Lead maintainer may revoke co-maintainer access if:
- Inactivity for 90+ days without notice
- Consistent violation of project principles or code of conduct
- Inability to reach consensus on critical technical decisions
- Security or trust concerns

## Decision Making

- **Day-to-day**: Co-maintainers can merge PRs in their CODEOWNERS areas after approval
- **Architectural changes**: Require discussion and consensus between maintainers
- **Breaking changes**: Require explicit approval from lead maintainer
- **Releases**: Lead maintainer controls version bumping and publishing
- **Tie-breaker**: Lead maintainer has final say

## Responsibilities

All maintainers must:
- Respond to issues/PRs within reasonable time (typically 2-3 business days)
- Maintain code quality and test coverage
- Follow contributing guidelines
- Help triage issues and guide newcomers
- Represent the project positively in public spaces

## Emergency Process

For critical security issues or production-blocking bugs:
- Any maintainer can push fix directly to main after minimum review
- Post-mortem issue must be opened within 24 hours
- Lead maintainer must be notified ASAP

---
*Last updated: 2026-06-27*