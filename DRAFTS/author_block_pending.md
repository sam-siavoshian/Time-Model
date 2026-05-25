# PENDING author block (apply to PAPER.md preamble + CITATION.cff after Omar approves via email_omar_review.md)

## PAPER.md preamble replacement

Replace current preamble:

```markdown
# Chronometric Injection: Time-Conditional Behavior in a Frozen LLM via Per-Layer FiLM of Real Elapsed Seconds

**Author:** Saam Siavoshian (independent)
**Original draft:** 2026-05-12 (as IPCN spec)
**Round-1 empirical:** 2026-05-22 (single-seed v11)
**Round-2 reviewer-rigor pass:** 2026-05-23
**Cross-seed release:** 2026-05-24 (v15 n=3, checkpoint release v15.0)
```

with:

```markdown
# Chronometric Injection: Time-Conditional Behavior in a Frozen LLM via Per-Layer FiLM of Real Elapsed Seconds

**Saam Siavoshian¹** · **Omar Ramadan²**

¹Independent
²Johns Hopkins University, Data Science MS

**Correspondence:** samsiavoshian2009@gmail.com

**Original draft:** 2026-05-12 (as IPCN spec)
**Round-1 empirical:** 2026-05-22 (single-seed v11)
**Round-2 reviewer-rigor pass:** 2026-05-23
**Cross-seed release:** 2026-05-24 (v15 n=3, checkpoint release v15.0)

## Author Contributions

We follow the CRediT taxonomy ([https://credit.niso.org](https://credit.niso.org)).

- **S.S.:** Conceptualization (chronometric injection architecture, post-pivot), Methodology, Software, Formal analysis, Investigation, Validation, Writing — original draft, Visualization, Project administration.
- **O.R.:** Conceptualization (original IPCN memory-routing architecture, documented in Appendix D §D.22, deprecated after nine null results), Resources (GPU compute infrastructure: NVIDIA DGX Spark prototype), Writing — review & editing, Supervision (paper review via JHU advisor [advisor name]).

## Acknowledgments

S.S. thanks O.R. for the original IPCN ideation, GPU access on the Spark prototype, and a careful manuscript review. We thank [JHU advisor name] for the faculty-level review pass before public release. We thank the Qwen team for releasing Qwen 2.5 3B-Instruct as the frozen base model. Compute for the v15 cross-seed release was provided by NVIDIA's DGX Spark prototype program through O.R.'s lab at Johns Hopkins University.
```

## CITATION.cff replacement

Replace authors block:

```yaml
authors:
  - family-names: "Siavoshian"
    given-names: "Saam"
    # TODO: add ORCID once registered: https://orcid.org/####-####-####-####
```

with:

```yaml
authors:
  - family-names: "Siavoshian"
    given-names: "Saam"
    affiliation: "Independent"
    email: "samsiavoshian2009@gmail.com"
    # TODO: add ORCID once registered: https://orcid.org/####-####-####-####
  - family-names: "Ramadan"
    given-names: "Omar"
    affiliation: "Johns Hopkins University, Data Science MS"
    # TODO: add Omar's email + ORCID
```

And in `preferred-citation.authors` mirror the same two-author list.

## paper.bib new entry for self-citation

Add to paper.bib:

```bibtex
@article{siavoshian2026chronometric,
  title         = {Chronometric Injection: Time-Conditional Behavior in a Frozen LLM via Per-Layer FiLM of Real Elapsed Seconds},
  author        = {Siavoshian, Saam and Ramadan, Omar},
  year          = {2026},
  eprint        = {TBD},  % fill in once arXiv ID assigned
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://github.com/sam-siavoshian/Time-Model},
  note          = {Preprint. Three cross-seed checkpoints released as GitHub release v15.0.}
}
```

---

## When to apply

After Omar replies to the review-request email (`email_omar_review.md`) and:
1. Approves the author-contribution statement as-is, or sends an accepted redline.
2. Confirms his JHU advisor will do (or has done) a review pass.
3. Says go for arXiv.

Then: apply all three replacements above, regenerate paper.bib if needed, commit with message `feat(authorship): add Omar Ramadan as second author with CRediT statement after his sign-off + JHU advisor review`.

If Omar declines co-authorship: do nothing to author block; just add Omar to Acknowledgments per the email's "Notes for Saam" section.
