"""Production-readiness patch 1: dashboard regressions, pipeline tolerance, README, security."""
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


# ---- 1) pipeline.py: restore _load_or_none (was lost in later edits) ----
patch("src/realtime/pipeline.py", [
    ('''        self.fish_model, self.fish_classes, self.fish_src, self.fish_img_size = load_best_classifier(
            "fish", cfg.paths.models_dir, self.device, cfg.paths.demo_dir
        )
        self.coral_model, self.coral_classes, self.coral_src, self.coral_img_size = load_best_classifier(
            "coral", cfg.paths.models_dir, self.device, cfg.paths.demo_dir
        )
        self.segmenter = load_segmenter(cfg.paths.models_dir, self.device)''',
     '''        def _load_or_none(task: str):
            try:
                return load_best_classifier(task, cfg.paths.models_dir, self.device, cfg.paths.demo_dir)
            except FileNotFoundError as exc:
                logger.warning("%s classifier unavailable: %s", task, exc)
                return None, [], "missing", 224

        self.fish_model, self.fish_classes, self.fish_src, self.fish_img_size = _load_or_none("fish")
        self.coral_model, self.coral_classes, self.coral_src, self.coral_img_size = _load_or_none("coral")
        self.segmenter = load_segmenter(cfg.paths.models_dir, self.device)'''),
    # CAM construction tolerates missing fish model
    ('''        self._cam: GradCAM | None = GradCAM(self.fish_model, find_target_layer(self.fish_model)) if enable_cam else None''',
     '''        self._cam: GradCAM | None = (
            GradCAM(self.fish_model, find_target_layer(self.fish_model))
            if enable_cam and self.fish_model is not None else None
        )'''),
    # shared-singleton hygiene: reset tracker per track=True analysis
    ('''        enhanced = self.enhancer.enhance(image_bgr) if self.cfg.enhancement.enabled else image_bgr
        timings["enhance"] = time.perf_counter() - t0''',
     '''        if track and self.cfg.realtime.get("tracker_enabled", True):
            # Dashboard singleton is process-wide: reset per analysis so
            # concurrent sessions cannot bleed track IDs into each other.
            self.tracker = CentroidTracker(
                max_disappeared=self.cfg.realtime.tracker_max_disappeared,
                min_distance=self.cfg.realtime.tracker_min_distance,
            )
        enhanced = self.enhancer.enhance(image_bgr) if self.cfg.enhancement.enabled else image_bgr
        timings["enhance"] = time.perf_counter() - t0'''),
], "pipeline _load_or_none restored + tracker reset")

# ---- 2) species.py: correct gate placement (analysis assigned BEFORE gate) ----
p = ROOT / "src/dashboard/pages/species.py"
src = p.read_text(encoding="utf-8")
old = '''    with st.spinner("Analysing ..."):
        pipeline = get_pipeline(enable_cam=False)
    if pipeline.fish_model is None:
        st.error("No trained fish model found — run `python run_smoke_test.py` (demo) or "
                     "`python train.py --task fish` first.", icon="🚨")
        return
        analysis = pipeline.analyze_image(image)
        heatmap = None
        if show_heatmap and analysis.detections:
            from ...explainability.cam import GradCAM
            from ...explainability.targets import find_target_layer
            from ...explainability.visualize import overlay_cam_on_image

            with GradCAM(pipeline.fish_model, find_target_layer(pipeline.fish_model)) as cam:
                tensor = pipeline._preprocess(analysis.enhanced, pipeline.fish_img_size)
                heatmap = cam.generate(tensor)
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))'''
new = '''    with st.spinner("Analysing ..."):
        pipeline = get_pipeline(enable_cam=False)
    if pipeline.fish_model is None:
        st.error("No trained fish model found — run `python run_smoke_test.py` (demo) or "
                 "`python train.py --task fish` first.", icon="🚨")
        return
    analysis = pipeline.analyze_image(image)
    heatmap = None
    if show_heatmap and analysis.detections:
        from ...explainability.cam import GradCAM
        from ...explainability.targets import find_target_layer

        with GradCAM(pipeline.fish_model, find_target_layer(pipeline.fish_model)) as cam:
            tensor = pipeline._preprocess(analysis.enhanced, pipeline.fish_img_size)
            heatmap = cam.generate(tensor)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))'''
assert old in src, "species gate block"
src = src.replace(old, new)
p.write_text(src, encoding="utf-8")
print("species.py gate fixed")

# ---- 3) coral.py: same ----
patch("src/dashboard/pages/coral.py", [
    ('''    with st.spinner("Assessing coral condition ..."):
        pipeline = get_pipeline(enable_cam=False)
    if pipeline.coral_model is None:
        st.error("No trained coral model found — run `python run_smoke_test.py` (demo) or "
                     "`python train.py --task coral` first.", icon="🚨")
        return
        analysis = pipeline.analyze_image(image)''',
     '''    with st.spinner("Assessing coral condition ..."):
        pipeline = get_pipeline(enable_cam=False)
    if pipeline.coral_model is None:
        st.error("No trained coral model found — run `python run_smoke_test.py` (demo) or "
                 "`python train.py --task coral` first.", icon="🚨")
        return
    analysis = pipeline.analyze_image(image)'''),
], "coral.py gate fixed")

# ---- 4) realtime.py: gate back inside the button block, sanitize uploaded name ----
p = ROOT / "src/dashboard/pages/realtime.py"
src = p.read_text(encoding="utf-8")
old = '''    if st.button("🚀 Run analysis", type="primary", use_container_width=True):
        tmp_dir = Path(tempfile.mkdtemp(prefix="reefscan_"))
        src = tmp_dir / uploaded.name
        src.write_bytes(uploaded.getvalue())
        out_path = tmp_dir / "annotated.mp4"

        pipeline = get_pipeline(enable_cam=False)
    if pipeline.fish_model is None:
        st.error("No trained fish model found — run `python run_smoke_test.py` (demo) or "
                     "`python train.py --task fish` first.", icon="🚨")
        return
        pipeline.cfg.realtime.frame_skip = frame_skip'''
new = '''    if st.button("🚀 Run analysis", type="primary", use_container_width=True):
        tmp_dir = Path(tempfile.mkdtemp(prefix="reefscan_"))
        src = tmp_dir / Path(uploaded.name).name  # sanitize client-controlled name
        src.write_bytes(uploaded.getvalue())
        out_path = tmp_dir / "annotated.mp4"

        pipeline = get_pipeline(enable_cam=False)
        if pipeline.fish_model is None:
            st.error("No trained fish model found — run `python run_smoke_test.py` (demo) or "
                     "`python train.py --task fish` first.", icon="🚨")
            return
        pipeline.cfg.realtime.frame_skip = frame_skip'''
assert old in src, "realtime gate block"
src = src.replace(old, new)
p.write_text(src, encoding="utf-8")
print("realtime.py gate fixed + filename sanitized")

# ---- 5) README: resolve committed merge-conflict markers (keep rich side) ----
p = ROOT / "README.md"
src = p.read_text(encoding="utf-8")
if "<<<<<<< HEAD" in src:
    lines = src.split("\n")
    out, mode = [], "head"   # keep the HEAD (rich) side, drop repo-side + markers
    for line in lines:
        if line.startswith("<<<<<<<"):
            mode = "head"
            continue
        if line.startswith("======="):
            mode = "other"
            continue
        if line.startswith(">>>>>>>"):
            mode = "done"
            continue
        if mode in ("head", "done", "start"):
            out.append(line)
        # mode == "other" -> skip the remote side
    src = "\n".join(out)
    p.write_text(src, encoding="utf-8")
    print("README conflict markers resolved")
else:
    print("README already clean")

print("PROD PATCH 1 COMPLETE")
