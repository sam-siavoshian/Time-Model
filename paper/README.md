# IEEE LaTeX paper source

This directory contains the IEEE-conference-format LaTeX source for the
Chronometric Injection paper.

## Files

- `main.tex` — the manuscript (IEEEtran conference class).
- `refs.bib` — bibliography (copy of repo-root `paper.bib`).
- `figures/` — embedded figures (copies of repo-root `figures/`).

## Authors

1. **Sam Siavoshian** — first author and corresponding author. Independent.
2. **Omar Ramadan** — co-author. Department of Data Science, Johns Hopkins
   University.
3. **JHU faculty advisor co-author** — to be added prior to publication.

## Build

The recommended path is Overleaf, which ships `IEEEtran.cls` by default.

1. Create a new Overleaf project, upload everything in this directory.
2. Set the compiler to `pdfLaTeX`.
3. Set the main document to `main.tex`.
4. Compile.

Local build (requires MacTeX or TeX Live with IEEEtran installed):

```bash
cd paper
latexmk -pdf main.tex
```

If `latexmk` is unavailable:

```bash
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

## What is intentionally not included

- The pre-pivot IPCN / memory-routing thread (Appendix D in the source
  manuscript) is omitted. The paper focuses on Chronometric Injection as
  the load-bearing architectural contribution.
- Retracted claims (OOD behavioral transfer, single-scalar-dial framing)
  are noted in `Section~\ref{sec:critical-controls}` and
  `Section~\ref{sec:limitations}`, not removed silently.
