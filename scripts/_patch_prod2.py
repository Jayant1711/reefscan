"""Production-readiness patch 2: robustness, reproducibility, ONNX, security hardening."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path, replacements, label):
    fp = ROOT / path
    src = fp.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new); changed = True
        elif new in src:
            continue
        else:
            print(f"  !! {label}: absent -> {old[:64]!r}")
    if changed:
        fp.write_text(src, encoding="utf-8")
    print(f"{label}: {'OK' if changed else 'no changes'}")


# ---- 1) torch.load -> weights_only=True at every site (verified safe: all
#         checkpoints are plain dict/str/int/float/list/tensor payloads) ----
patch("src/classification/engine.py", [
    ('ckpt = torch.load(best_path, map_location=self.device, weights_only=False)',
     'ckpt = torch.load(best_path, map_location=self.device, weights_only=True)'),
], "engine weights_only")
patch("src/realtime/pipeline.py", [
    ("torch.load(ckpt_path, map_location=device, weights_only=False)",
     "torch.load(ckpt_path, map_location=device, weights_only=True)"),
], "pipeline weights_only (2 sites)")
patch("evaluate.py", [
    ('torch.load(models_dir / name / "best.pt", map_location="cpu", weights_only=False)',
     'torch.load(models_dir / name / "best.pt", map_location="cpu", weights_only=True)'),
], "root evaluate weights_only")
patch("src/classification/evaluate.py", [
    ('ckpt = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)',
     'ckpt = torch.load(model_dir / "best.pt", map_location=device, weights_only=True)'),
], "classification evaluate weights_only")
patch("src/segmentation/evaluate_seg.py", [
    ('bundle = torch.load(ckpt_path, map_location=device, weights_only=False)',
     'bundle = torch.load(ckpt_path, map_location=device, weights_only=True)'),
], "evaluate_seg weights_only")
patch("scripts/export_onnx.py", [
    ('ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)',
     'ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)'),
], "export_onnx weights_only")

# run_smoke_test may use either spelling
p = ROOT / "run_smoke_test.py"
src = p.read_text(encoding="utf-8")
if "weights_only=False" in src:
    src = src.replace("weights_only=False", "weights_only=True")
    p.write_text(src, encoding="utf-8")
    print("run_smoke_test weights_only: OK")

# ---- 2) train.py: per-backbone isolation + --continue ----
patch("train.py", [
    ('''    parser.add_argument("--quick", action="store_true", help="Tiny fast run (CPU smoke test)")
    return parser.parse_args()''',
     '''    parser.add_argument("--quick", action="store_true", help="Tiny fast run (CPU smoke test)")
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="Skip runs whose runs/<task>/<model>_<strategy>/manifest.json already exists")
    return parser.parse_args()'''),
    ('''    out_dir = ensure_dir(Path(cfg.paths.models_dir) / task / model_name)
    trainer = Trainer(''',
     '''    manifest_path = (Path(cfg.paths.results_dir).parent / "runs" / task /
                     f"{model_name}_{(strategy_override or section.get('strategy', 'finetune'))}" /
                     "manifest.json")
    if getattr(args_like(), "continue_run", False) and manifest_path.exists():
        logger.info("Skipping %s/%s — manifest exists at %s (--continue).",
                    task, model_name, manifest_path)
        return None

    out_dir = ensure_dir(Path(cfg.paths.models_dir) / task / model_name)
    trainer = Trainer('''),
    ('''    models = SUPPORTED_MODELS if args.model == "all" else [args.model]
    for model_name in models:
        logger.info("=== Training %s / %s (%s) ===",
                    args.task, model_name, cfg[args.task].get("strategy", "finetune"))
        train_classifier(args.task, model_name, cfg, device)

    logger.info("Training complete. Leaderboard: results/%s/model_leaderboard.csv", args.task)
    return 0''',
     '''    models = SUPPORTED_MODELS if args.model == "all" else [args.model]
    failures: list[str] = []
    for model_name in models:
        logger.info("=== Training %s / %s (%s) ===",
                    args.task, model_name, cfg[args.task].get("strategy", "finetune"))
        try:
            train_classifier(args.task, model_name, cfg, device, strategy_override=args.strategy)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            failures.append(f"{model_name}: CUDA OOM ({exc}) — retry with "
                            f"--set {args.task}.batch_size=16")
            logger.error("OOM on %s — continuing with remaining backbones.", model_name)
        except Exception as exc:  # noqa: BLE001 — one backbone must not kill the screen
            failures.append(f"{model_name}: {type(exc).__name__}: {exc}")
            logger.exception("Backbone %s failed — continuing.", model_name)

    if failures and len(failures) == len(models):
        raise RuntimeError("All backbones failed:\\n" + "\\n".join(failures))
    for failure in failures:
        logger.warning("FAILED backbone: %s", failure)

    logger.info("Training complete. Leaderboard: results/%s/model_leaderboard.csv", args.task)
    return 0'''),
], "train.py robustness")

# _write_manifest needs args access for --continue; use a tiny module-level holder
patch("train.py", [
    ('''logger = logging.getLogger("train")''',
     '''logger = logging.getLogger("train")
_CLI_ARGS: argparse.Namespace | None = None


def args_like() -> argparse.Namespace:
    """Module-level access to parsed CLI args (for --continue inside helpers)."""
    return _CLI_ARGS if _CLI_ARGS is not None else argparse.Namespace(continue_run=False)'''),
    ('''def main() -> int:
    args = parse_args()''',
     '''def main() -> int:
    global _CLI_ARGS
    args = parse_args()
    _CLI_ARGS = args'''),
], "train.py args holder")

# ---- 3) manifest additions: seed, versions, full config, dataset summary ----
patch("src/utils/provenance.py", [
    ('''def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")''',
     '''def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def env_versions() -> dict:
    """Runtime versions a reproducer needs (python/torch/cuda)."""
    import sys

    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda or "cpu-build"
    except Exception:  # pragma: no cover
        torch_version, cuda_version = "unknown", "unknown"
    return {"python": sys.version.split()[0], "torch": torch_version, "cuda": cuda_version}'''),
    ('''    config_snapshot: dict,
    params: dict,
    train_seconds: float,
    epochs_ran: int,
    final_metrics: dict,
    artifacts: dict[str, str] | None = None,
    project_root: str | Path | None = None,
) -> dict:
    """Assemble the canonical per-run manifest written to ``runs/<task>/...``."""
    return {
        "schema": "reefscan-run-manifest-v1",
        "recorded_at": utc_now_iso(),
        "git_sha": git_sha(project_root),
        "platform": detect_platform(),
        "gpu": gpu_name(),
        "task": task,
        "model": model_name,
        "strategy": strategy,
        "dataset": dataset,
        "config": config_snapshot,
        "params": params,''',
     '''    config_snapshot: dict,
    params: dict,
    train_seconds: float,
    epochs_ran: int,
    final_metrics: dict,
    artifacts: dict[str, str] | None = None,
    project_root: str | Path | None = None,
    seed: int | None = None,
    dataset_summary: dict | None = None,
) -> dict:
    """Assemble the canonical per-run manifest written to ``runs/<task>/...``."""
    return {
        "schema": "reefscan-run-manifest-v2",
        "recorded_at": utc_now_iso(),
        "git_sha": git_sha(project_root),
        "platform": detect_platform(),
        "gpu": gpu_name(),
        "env": env_versions(),
        "seed": seed,
        "task": task,
        "model": model_name,
        "strategy": strategy,
        "dataset": dataset,
        "dataset_summary": dataset_summary or {},
        "config": config_snapshot,
        "params": params,'''),
], "manifest v2 additions")

# train.py: pass seed/versions/full config/dataset summary
patch("train.py", [
    ('''    manifest = build_manifest(
        task=task,
        model_name=model_name,
        strategy=strategy,
        dataset=str(section.get("data_dir", "")),
        config_snapshot={
            "image_size": section.get("image_size"),
            "batch_size": section.get("batch_size"),
            "epochs": section.get("epochs"),
            "lr": section.get("lr"),
            "optimizer": section.get("optimizer"),
            "scheduler": section.get("scheduler"),
            "early_stopping": section.get("early_stopping"),
            "metric_for_best": section.get("metric_for_best"),
        },''',
     '''    manifest = build_manifest(
        task=task,
        model_name=model_name,
        strategy=strategy,
        dataset=str(section.get("data_dir", "")),
        config_snapshot={k: v for k, v in section.items() if k != "augmentation"},'''),
    ('''        artifacts={
            "checkpoint": str(Path(cfg.paths.models_dir) / task / model_name / "best.pt"),
            "metrics": str(Path(cfg.paths.results_dir) / task / model_name / "metrics.json"),
            "leaderboard": str(Path(cfg.paths.results_dir) / task / "model_leaderboard.csv"),
        },
        project_root=Path(__file__).resolve().parent,
    )''',
     '''        artifacts={
            "checkpoint": str(Path(cfg.paths.models_dir) / task / model_name / "best.pt"),
            "metrics": str(Path(cfg.paths.results_dir) / task / model_name / "metrics.json"),
            "leaderboard": str(Path(cfg.paths.results_dir) / task / "model_leaderboard.csv"),
        },
        project_root=Path(__file__).resolve().parent,
        seed=cfg.project.get("seed", 42),
        dataset_summary=dataset_summary,
    )'''),
    ('''def _write_manifest(cfg, task: str, model_name: str, strategy: str, train_seconds: float,
                    epochs_ran: int, final_metrics: dict, params: dict) -> Path:''',
     '''def _write_manifest(cfg, task: str, model_name: str, strategy: str, train_seconds: float,
                    epochs_ran: int, final_metrics: dict, params: dict,
                    dataset_summary: dict | None = None) -> Path:'''),
    ('''    _write_manifest(
        cfg, task, model_name, train_cfg.strategy, train_seconds, epochs_ran,
        final_metrics={k: v for k, v in metrics.items()
                       if isinstance(v, (int, float)) and k != "n"},
        params=trainer.params,
    )''',
     '''    _write_manifest(
        cfg, task, model_name, train_cfg.strategy, train_seconds, epochs_ran,
        final_metrics={k: v for k, v in metrics.items()
                       if isinstance(v, (int, float)) and k != "n"},
        params=trainer.params,
        dataset_summary={
            "num_classes": bundle.num_classes,
            "class_counts": dict(bundle.class_counts),
            "n_train": len(bundle.train.dataset),
            "n_val": len(bundle.val.dataset) if bundle.val else 0,
            "n_test": len(bundle.test.dataset) if bundle.test else 0,
        },
    )'''),
], "train.py manifest v2 wiring")

# ---- 4) ckpt_version in classification checkpoints ----
patch("src/classification/engine.py", [
    ('''                {
                    "model": self.model.state_dict(),
                    "model_name": self.model_name,''',
     '''                {
                    "ckpt_version": 1,
                    "model": self.model.state_dict(),
                    "model_name": self.model_name,'''),
], "ckpt_version")

# ---- 5) ONNX extras + friendly ImportError ----
p = ROOT / "requirements.txt"
src = p.read_text(encoding="utf-8")
if "onnxruntime" not in src:
    src = src.rstrip() + """

# --- optional: ONNX export/deployment (python scripts/export_onnx.py) ---
onnx>=1.15
onnxscript
onnxruntime>=1.16
"""
    p.write_text(src, encoding="utf-8")
    print("requirements onnx extras: OK")

patch("scripts/export_onnx.py", [
    ('''    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)''',
     '''    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)'''),
], "export_onnx noop anchor")

# friendly import guard at top of export
p = ROOT / "scripts/export_onnx.py"
src = p.read_text(encoding="utf-8")
if "No module named 'onnx'" not in src:
    src = src.replace(
        "import torch",
        "import torch\n\n"
        "try:\n"
        "    import onnx  # noqa: F401\n"
        "except ImportError as _exc:\n"
        "    raise SystemExit(\n"
        "        'ONNX extras missing — install with: pip install onnx onnxscript onnxruntime'\n"
        "    ) from _exc",
        1,
    )
    p.write_text(src, encoding="utf-8")
    print("export_onnx friendly import: OK")

# ---- 6) survey thresholds from a shared constant + config pinning test ----
patch("src/coral_analysis/severity.py", [
    ('''logger = logging.getLogger(__name__)''',
     '''logger = logging.getLogger(__name__)

# Single source for the default WGBS class cut-offs (mirrors configs/default.yaml
# coral.severity.thresholds; a pinning test keeps the two in sync).
DEFAULT_THRESHOLDS = (15, 35, 60, 85)'''),
    ("def index_to_class(index: float, thresholds: tuple[float, ...] = (15, 35, 60, 85)) -> str:",
     "def index_to_class(index: float, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> str:"),
    ("def index_to_class_strict(index: float, thresholds: tuple[float, ...] = (15, 35, 60, 85)) -> str:",
     "def index_to_class_strict(index: float, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> str:"),
], "severity DEFAULT_THRESHOLDS")

patch("src/analytics/survey.py", [
    ("thresholds = tuple(self.cfg.coral.severity.thresholds) if self.cfg else (15, 35, 60, 85)",
     "thresholds = tuple(self.cfg.coral.severity.thresholds) if self.cfg else DEFAULT_THRESHOLDS"),
], "survey uses DEFAULT_THRESHOLDS")

if "from ..coral_analysis.severity import index_to_class_strict" in (ROOT / "src/analytics/survey.py").read_text(encoding="utf-8") \
        and "DEFAULT_THRESHOLDS" not in (ROOT / "src/analytics/survey.py").read_text(encoding="utf-8").split("def _condition_distribution")[0]:
    p = ROOT / "src/analytics/survey.py"
    src = p.read_text(encoding="utf-8")
    src = src.replace(
        "from ..coral_analysis.severity import index_to_class_strict",
        "from ..coral_analysis.severity import DEFAULT_THRESHOLDS, index_to_class_strict",
    )
    p.write_text(src, encoding="utf-8")
    print("survey import extended")

# ---- 7) deepfish TypeError fallback: validate members on old pythons ----
patch("src/data/downloaders/deepfish.py", [
    ('''        try:
            tf.extractall(data_dir, filter="data")  # py3.12+: no deprecation, safe members
        except TypeError:  # py<3.11.4 has no filter kwarg
            tf.extractall(data_dir)''',
     '''        try:
            tf.extractall(data_dir, filter="data")  # py3.12+: no deprecation, safe members
        except TypeError:
            # py<3.11.4: no filter kwarg — validate members manually (no absolute
            # paths, no '..' parts, no links) before the vulnerable extractall.
            unsafe = [
                m for m in tf.getmembers()
                if m.issym() or m.islnk()
                or m.name.startswith(("/", "\\\\"))
                or ".." in m.name.replace("\\\\", "/").split("/")
            ]
            if unsafe:
                raise RuntimeError(f"Unsafe tar members: {[u.name for u in unsafe[:5]]}")
            tf.extractall(data_dir)'''),
], "deepfish fallback hardening")

# ---- 8) docs: test count 77→(78 with export test), T4 wording, DEPLOYMENT claim ----
patch("PLATFORMS.md", [
    ("!python -m pytest tests/ -q    # verify the full suite (69 tests) on the platform",
     "!python -m pytest tests/ -q    # verify the full suite (78 tests) on the platform"),
    ("# 4. E1 — coral feature-extraction screen (all 4 backbones, ~35 min on T4×2)",
     "# 4. E1 — coral feature-extraction screen (all 4 backbones, ~35 min on a T4)"),
], "PLATFORMS counts+wording")
patch("scripts/build_platform_notebooks.py", [
    ("Duration ≈ 1.5–2.5 h on two T4s.",
     "Duration ≈ 1.5–2.5 h on a T4."),
], "notebook intro wording")

p = ROOT / "docs/DEPLOYMENT.md"
src = p.read_text(encoding="utf-8")
if "1.5–2.5×" in src:
    src = src.replace("1.5–2.5×", "often ~2× (benchmark-dependent)")
    p.write_text(src, encoding="utf-8")
    print("DEPLOYMENT claim softened")

print("PROD PATCH 2 COMPLETE")
