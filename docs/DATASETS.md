# Dataset Guide — ReefScan AI

Three public, free datasets drive the system. `python download_data.py` fetches all of them.

## 1. fishnet.ai v0.1.2 (The Nature Conservancy) — fish species from bbox crops
- **Site:** https://www.fishnet.ai/download · **Paper:** arXiv:2106.09178
- **Download:** official S3 mirror (quota-free, resumable): images 5.7 GB + labels 2.1 MB
  (`https://fishnet-data.s3-us-west-1.amazonaws.com/foid_{images,labels}_v012.zip`) —
  fetched by `src/data/downloaders/fishnet.py` via `curl -f -C -`.
- **VERIFIED labels schema** (`foid_labels_bbox_v012.csv`):
  `img_id, bbox_id, x_min, x_max, y_min, y_max, label_l1, label_l2` —
  159,119 boxes over 34,987 frames; 28 `label_l1` classes (Albacore 33.9K,
  Yellowfin tuna 6.7K, …, `Human` 104K), 13 `label_l2` codes.
  No split column in v0.1.2 → `prepare_fishnet.py` derives a seeded
  **image-level** 80/10/10 split (all crops of one frame share a split — no leakage).

### Curated crop subset (why and how)
Classification trains on **cropped fish regions** (matching the real-time pipeline):
`Human`/`Unknown`/`No fish` rows are excluded; classes with <100 boxes dropped;
top-12 kept (config: `fish.prepare.{level, top_k, min_images_per_class,
exclude_classes, split_fractions}`). Output:

```
datasets/fishnet/curated/<split>/<CLASS>/<img_id>_bbox<id>.jpg
```

A `_COMPLETE` marker guards against partial runs: interrupted preparations resume
per-crop, never silently train on a truncated set.

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
| fishnet.ai raw | ~5.7 GB images + 2.1 MB labels | species classification |
| fishnet.ai curated crops | ~1–2 GB | (crops kept after prepare) |
| DeepFish | ~7.1 GB | segmentation / boxes |
| NOAA coral | ~770 MB | coral analysis |
| Synthetic | ~20 MB | smoke test / demo |

## 6. Ethics & licensing
All three datasets are publicly released for research. FishNet/DeepFish are CC BY-family;
the NOAA dataset is a U.S. federal scientific product. Cite the original papers
(see `research_paper/references.bib`) in any derived work.
