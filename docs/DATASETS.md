# Dataset Guide — ReefScan AI

Three public, free datasets drive the system. `python download_data.py` fetches all of them.

## 1. Fish4Knowledge (F4K) — fish species classification
- **Source:** official University of Edinburgh ground-truth mirror —
  https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/ (live, quota-free)
- **Package:** single `fishRecognition_GT.tar` (487 MB, MD5 published alongside) + `class_id.csv` (0.6 MB)
- **Verified content:** `fish_image/fish_01..fish_23/*.png` — 27,394 fish crops whose
  **directory IS the species label** (fish_NN == cluster N; verified count-for-count against the
  official `class_id.csv`: 27,370 rows, 23 clusters); matching `mask_image/mask_NN/` masks included
- **Curation:** 13 of 23 clusters have ≥ 100 images (26,903 crops kept); seeded **video-group**
  split 80/10/10 — crops of one video segment share a split (no scene-level leakage)
- **Curated layout:** `datasets/f4k/curated/{train,validation,test}/<species_NN>/*.png`

## 2. DeepFish (Scientific Reports 2020) — fish presence & segmentation
- **Site:** https://alzayats.github.io/DeepFish/ · **Mirror:** HF `Alzayats/DeepFish`
- **Scale:** 39,766 Full-HD frames from 20 tropical habitats; binary presence labels for all,
  point labels for 3,200, **pixel masks for 620** (the part we use for detection boxes)
- **License:** CC BY 4.0
- **Download:** automatic (`datasets/deepfish/DeepFish/...`). Used by `python train.py --task seg`.

## 3. NOAA PIFSC ESD Coral Bleaching (HuggingFace) — coral health
- **Repo:** `NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset`
- **Scale:** 10,419 × 224px patches (train 7,292 / validation 1,562 / test 1,565)
- **Labels:** binary — `CORAL` (healthy hard coral) vs `CORAL_BL` (bleached)
- **Materialised layout:**
```
datasets/coral/imagefolders/{train,validation,test}/{CORAL,CORAL_BL}/*.jpg
```
- **Severity innovation:** because the public labels are binary, ReefScan adds the
  **Whiteness-Guided Bleaching Severity (WGBS)** layer to produce the 5-class severity
  required by the project spec — see `docs/RESEARCH_METHODOLOGY.md` §3.

## 4. Synthetic dataset (built-in, zero download)
`run_smoke_test.py` generates `datasets/synthetic/{fish,coral,seg,enhance}` — coloured,
textured mini-datasets that make every pipeline stage verifiable in minutes on CPU.

## 5. Expected disk usage
| Dataset | Size | Needed for |
|---|---|---|
| F4K raw | 487 MB tar + 0.6 MB class_id.csv | species classification |
| F4K curated | ~0.9 GB | (species crops kept after prepare) |
| DeepFish | ~7.1 GB | segmentation / boxes |
| NOAA coral | ~770 MB | coral analysis |
| Synthetic | ~20 MB | smoke test / demo |

## 6. Ethics & licensing
All three datasets are publicly released for research. FishNet/DeepFish are CC BY-family;
the NOAA dataset is a U.S. federal scientific product. Cite the original papers
(see `research_paper/references.bib`) in any derived work.
