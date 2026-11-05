# Dataset Guide — ReefScan AI

Three public, free datasets drive the system. `python download_data.py` fetches all of them.

## 1. FishNet (ICCV 2023) — species classification
- **Site:** https://fishnet-2023.github.io/ · **Paper:** Khan et al., ICCV 2023
- **Scale:** 94,532 images, 17,357 aquatic species, taxonomy-organised (order/family/genus/species)
- **Download:** official Google Drive archive (~15 GB) — handled automatically by `src/data/downloaders/fishnet.py` (gdown)
- **Manual fallback:** open https://drive.google.com/file/d/1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3/view in a browser, place the zip at `datasets/fishnet/raw/FishNet.zip`, re-run `python download_data.py --dataset fishnet` (extraction resumes).

### Curated subset (why and how)
Training 17k species on free compute is infeasible, so `prepare_fishnet.py` builds a
**taxonomy-level subset** (default: `family`, top-25 classes with ≥60 train images):

```
datasets/fishnet/curated/train/<CLASS>/*.jpg
datasets/fishnet/curated/validation/<CLASS>/*.jpg
datasets/fishnet/curated/test/<CLASS>/*.jpg
```
Tune via config: `fish.prepare.{level, top_k, min_images_per_class}`.

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
| FishNet raw | ~15 GB | species classification |
| FishNet curated | ~1–2 GB | (subset kept after prepare) |
| DeepFish | ~7.1 GB | segmentation / boxes |
| NOAA coral | ~770 MB | coral analysis |
| Synthetic | ~20 MB | smoke test / demo |

## 6. Ethics & licensing
All three datasets are publicly released for research. FishNet/DeepFish are CC BY-family;
the NOAA dataset is a U.S. federal scientific product. Cite the original papers
(see `research_paper/references.bib`) in any derived work.
