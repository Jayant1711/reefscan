# Training Guide — ReefScan AI

## 1. What gets trained
| Task | Model(s) | Dataset | Head |
|---|---|---|---|
| `fish` | resnet34, efficientnet_b0, mobilenet_v3, convnext_tiny | FishNet curated subset | ImageNet-pretrained backbone → dropout → Linear |
| `coral` | same 4 backbones | NOAA coral patches (binary) | + optional ordinal penalty (severity-aware) |
| `seg` | SmallUNet (~1.9 M params) | DeepFish segmentation masks | BCE + soft Dice |

## 2. Commands
```bash
# single model
python train.py --task fish --model resnet34

# all four backbones (writes the leaderboard automatically)
python train.py --task fish --model all

# coral (auto-downloads the NOAA dataset if missing)
python train.py --task coral --model all

# segmentation (auto-downloads DeepFish if missing)
python train.py --task seg

# quick CPU sanity run
python train.py --task fish --model resnet34 --quick
```

## 3. Training recipe (why it works on small data)
- **Transfer learning:** ImageNet init; discriminative LRs (backbone `lr`, fresh head `10×lr`).
- **Class balancing:** `WeightedRandomSampler` (equal exposure/epoch) + inverse-√ class weights in the loss. The two compound deliberately — the sampler equalises exposure while the mild inverse-√ tilt counteracts the majority bias of weighted-F1 model selection. The ablation is a flag, not a code edit: `--set fish.loss_class_weight_mode=none` (loss weights off) or `=inverse`.
- **Validation distribution:** val/test loaders use the *natural* class distribution (the sampler is train-only), so early stopping and every leaderboard number reflect the real-world distribution.
- **Underwater-aware augmentation:** hue ±0.05 (depth colour drift), saturation/brightness/contrast ±0.3, rotation ±15°, random erasing 0.25; mixup (α=0.2, p=0.5) for fish.
- **Optimisation:** AdamW, weight-decay 1e-4, cosine schedule with 1 warm-up epoch, label smoothing 0.1, grad-clip 5.0.
- **Early stopping** on validation weighted-F1 (patience 5); best checkpoint auto-restored.
- **AMP mixed precision** engages automatically on CUDA; CPU runs silently fall back to fp32.

## 4. Key hyperparameters (configs/default.yaml)
```yaml
fish: {image_size: 224, batch_size: 32, epochs: 15, lr: 1e-4, label_smoothing: 0.1}
coral:{image_size: 224, batch_size: 32, epochs: 10, lr: 3e-4, ordinal_loss: true}
seg:  {image_size: 256, batch_size: 8,  epochs: 10, lr: 3e-4}
```

## 5. Outputs per model
```
models/<task>/<model>/best.pt          # weights + class names + metrics
models/<task>/<model>/history.csv      # per-epoch train/val curves
models/<task>/best_model.json          # leaderboard winner
results/<task>/<model>/metrics.json    # acc, precision/recall/F1, top-5, throughput
results/<task>/<model>/confusion_matrix.png
results/<task>/<model>/per_class_report.csv
results/<task>/model_leaderboard.csv   # ranked comparison
```

## 6. Expected runtimes
| Setup | fish (1 model) | fish (4 models) | coral | seg |
|---|---|---|---|---|
| Colab T4 | ~25 min | ~2–4 h | ~40 min | ~20 min |
| Laptop CPU | hours | not advised | hours | ~1 h |

Tip: start with `mobilenet_v3` (smallest) to validate the full loop, then run the sweep overnight on Colab.

## 7. Evaluation & XAI after training
```bash
python evaluate.py --task fish                       # metrics + leaderboard refresh
python evaluate.py --enhancement --data datasets/fishnet/curated/train
python evaluate.py --xai --image <sample.jpg>
```
