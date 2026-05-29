# IEEE LaTeX paper source

This directory contains the IEEE-conference-format LaTeX source for the
Chronometric Injection paper.

## Files

- `main.tex` — the manuscript (IEEEtran conference class).
- `refs.bib` — bibliography (copy of repo-root `paper.bib`).
- `figures/` — embedded figures (copies of repo-root `figures/`).

## Authors

The author block is defined in `main.tex`. Do not add placeholder authors in
this README; keep it aligned with the manuscript source.

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

## Source-of-truth note

`main.tex` is the paper source. The pre-pivot IPCN / memory-routing thread is
not part of this LaTeX manuscript. Current limitations and negative results are
described directly in the discussion, including the negative TPS result and the
limits of prompt-vs-residual conditioning.
