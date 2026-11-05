# Compiling the paper

## Overleaf (recommended, free)
1. Go to [overleaf.com](https://www.overleaf.com) → New Project → Upload Project.
2. Upload `reefscan_ieee.tex` (the bibliography is embedded via `thebibliography`, so no `.bib` run is needed).
3. Compiler: pdfLaTeX → Recompile. Done.

## Local TeX (MiKTeX/TeX Live)
```bash
cd research_paper
pdflatex reefscan_ieee.tex
pdflatex reefscan_ieee.tex     # run twice for refs
```

## Filling in the numbers
1. Train on the real datasets (see `docs/TRAINING.md`).
2. Run `python evaluate.py --task fish` and `--task coral`; run the enhancement benchmark.
3. Open `results/{fish,coral}/model_leaderboard.csv` and `results/enhancement/enhancement_benchmark.csv`.
4. Replace every red **[TO-FILL]** cell in `reefscan_ieee.tex` (marked with `\tofill{}`) and recompile.
