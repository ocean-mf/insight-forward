"""
run_inference.py — CLI entrypoint for the RAFT + SAM 2 UVOS pipeline.

Usage examples:

  # Basic — small model, save demo to demo.mp4
  python run_inference.py --input talk.mp4

  # Choose model, side-by-side layout, save binary masks
  python run_inference.py \\
      --input talk.mp4 \\
      --output results/demo.mp4 \\
      --model small \\
      --side-by-side \\
      --save-masks results/masks/

  # Research mode: use base+ model, compile SAM 2, auto prompt frame
  python run_inference.py \\
      --input talk.mp4 --model base_plus \\
      --prompt-frame -1 --compile --bidirectional

  # Disable AMP (debugging)
  python run_inference.py --input talk.mp4 --no-amp
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_inference",
        description="UVOS Pipeline: RAFT optical flow + SAM 2 video segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── I/O ──────────────────────────────────────────────────────────────
    io = parser.add_argument_group("I/O")
    io.add_argument("--input", "-i", required=True, metavar="VIDEO",
                    help="Path to input video file (MP4, AVI, MOV, …)")
    io.add_argument("--output", "-o", default="demo.mp4", metavar="MP4",
                    help="Output demo MP4 path  [default: demo.mp4]")
    io.add_argument("--save-masks", metavar="DIR",
                    help="Directory to save per-frame binary masks as PNG files")

    # ── Model ────────────────────────────────────────────────────────────
    model_g = parser.add_argument_group("Model")
    model_g.add_argument(
        "--model", default="small",
        choices=["tiny", "small", "base_plus", "large",
                 "facebook/sam2.1-hiera-tiny",
                 "facebook/sam2.1-hiera-small",
                 "facebook/sam2.1-hiera-base-plus",
                 "facebook/sam2.1-hiera-large"],
        help="SAM 2 model variant  [default: small]",
    )
    model_g.add_argument("--checkpoint-dir", metavar="DIR",
                         help="Directory to cache SAM 2 checkpoints  "
                              "[default: ~/.cache/uvos/checkpoints]")
    model_g.add_argument("--raft-iters", type=int, default=20, metavar="N",
                         help="RAFT refinement iterations  [default: 20]")

    # ── Inference ────────────────────────────────────────────────────────
    inf = parser.add_argument_group("Inference")
    inf.add_argument("--prompt-frame", type=int, default=0, metavar="IDX",
                     help="Frame index to use as SAM 2 prompt "
                          "(-1 = auto-select by highest saliency)  [default: 0]")
    inf.add_argument("--bidirectional", action="store_true", default=True,
                     help="Propagate backward from prompt frame (default: on)")
    inf.add_argument("--no-bidirectional", dest="bidirectional", action="store_false",
                     help="Disable backward propagation")
    inf.add_argument("--top-k-motion", type=float, default=0.10, metavar="FRAC",
                     help="Top-k fraction of motion magnitude for centroid  [default: 0.10]")
    inf.add_argument("--device", default="cuda", metavar="DEVICE",
                     help="PyTorch device  [default: cuda]")
    inf.add_argument("--no-amp", action="store_true",
                     help="Disable Automatic Mixed Precision (float16)")
    inf.add_argument("--compile", action="store_true",
                     help="Enable torch.compile for SAM 2 encoder (slow warm-up)")
    inf.add_argument("--keep-raft", action="store_true",
                     help="Keep RAFT on GPU while running SAM 2 (needs ~12 GB VRAM)")

    # ── Visualiser ───────────────────────────────────────────────────────
    viz = parser.add_argument_group("Visualiser")
    viz.add_argument("--side-by-side", action="store_true",
                     help="Output original + annotated side-by-side")
    viz.add_argument("--mask-color", default="0,230,120", metavar="B,G,R",
                     help="Mask overlay colour as B,G,R  [default: 0,230,120]")
    viz.add_argument("--mask-alpha", type=float, default=0.45, metavar="ALPHA",
                     help="Mask overlay transparency 0–1  [default: 0.45]")
    viz.add_argument("--output-fps", type=float, default=None, metavar="FPS",
                     help="Override output FPS  [default: match source]")
    viz.add_argument("--codec", default="mp4v", metavar="FOURCC",
                     help="VideoWriter fourcc codec  [default: mp4v]")
    viz.add_argument("--mode-label", default="RAFT + SAM 2 (UVOS)", metavar="TEXT",
                     help="Processing mode label shown in HUD")
    viz.add_argument("--no-watermark", action="store_true",
                     help="Remove watermark text")

    # ── Misc ─────────────────────────────────────────────────────────────
    misc = parser.add_argument_group("Misc")
    misc.add_argument("--verbose", "-v", action="store_true",
                      help="Enable debug-level logging")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers unless verbose
    if not verbose:
        for name in ("sam2", "hydra", "omegaconf", "torch", "PIL"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _parse_bgr_color(s: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected B,G,R colour string, got: {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def save_binary_masks(
    results: list,
    output_dir: str,
    meta,
) -> None:
    """Save per-frame binary masks as 8-bit PNG files (0 = BG, 255 = FG)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_digits = max(5, len(str(meta.num_frames - 1)))
    saved = 0
    for r in results:
        name = str(r.frame_idx).zfill(n_digits) + ".png"
        mask_uint8 = (r.mask.astype(np.uint8)) * 255
        cv2.imwrite(str(out / name), mask_uint8)
        saved += 1

    print(f"Saved {saved} binary masks → {output_dir}")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    log = logging.getLogger("run_inference")

    # ── Validate input ────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        return 1

    # ── Build EngineConfig ─────────────────────────────────────────────
    from uvos_engine import EngineConfig, UVOSEngine

    engine_cfg = EngineConfig(
        device=args.device,
        sam2_model=args.model,
        checkpoint_dir=args.checkpoint_dir,
        raft_iters=args.raft_iters,
        use_amp=not args.no_amp,
        compile_sam2=args.compile,
        top_k_motion=args.top_k_motion,
        prompt_frame_idx=args.prompt_frame,
        bidirectional_propagation=args.bidirectional,
        offload_raft_after_flow=not args.keep_raft,
    )

    # ── Build VisualizerConfig ─────────────────────────────────────────
    from visualizer import VisualizerConfig, VideoVisualizer

    viz_cfg = VisualizerConfig(
        mask_color_bgr=_parse_bgr_color(args.mask_color),
        mask_alpha=args.mask_alpha,
        side_by_side=args.side_by_side,
        output_fps=args.output_fps,
        codec=args.codec,
        processing_mode=args.mode_label,
        watermark_text="" if args.no_watermark else "Motion-as-Attention | UVOS",
    )

    # ── Run pipeline ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    log.info("Starting UVOS pipeline on: %s", input_path.name)

    engine = UVOSEngine(engine_cfg)
    results, meta = engine.process_video(str(input_path))

    elapsed_seg = time.perf_counter() - t0
    log.info(
        "Segmentation done in %.1f s  (%.2f fps effective)",
        elapsed_seg,
        meta.num_frames / elapsed_seg,
    )

    # ── Render demo video ──────────────────────────────────────────────
    pipeline_fps = meta.num_frames / elapsed_seg
    log.info("Rendering demo video…")
    viz = VideoVisualizer(viz_cfg)
    viz.create_demo_video(str(input_path), results, args.output, meta,
                          pipeline_fps=pipeline_fps)

    # ── Optional: save masks ───────────────────────────────────────────
    if args.save_masks:
        save_binary_masks(results, args.save_masks, meta)

    # ── Summary ───────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t0
    avg_saliency = float(np.mean([r.saliency_score for r in results]))
    mask_coverage = float(np.mean([r.mask.mean() for r in results])) * 100

    print("\n" + "─" * 52)
    print("  UVOS Pipeline Summary")
    print("─" * 52)
    print(f"  Input          {input_path.name}")
    print(f"  Frames         {meta.num_frames}  @ {meta.fps:.2f} fps")
    print(f"  Resolution     {meta.width} × {meta.height}")
    print(f"  SAM 2 model    {engine_cfg.sam2_model}")
    print(f"  Prompt frame   {engine_cfg.prompt_frame_idx}")
    print(f"  Avg saliency   {avg_saliency:.3f}")
    print(f"  Avg mask cover {mask_coverage:.1f}%")
    print(f"  Total time     {total_elapsed:.1f} s")
    print(f"  Output         {args.output}")
    print("─" * 52 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
