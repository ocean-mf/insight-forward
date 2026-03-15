"""
visualizer.py — Demo video renderer for the UVOS pipeline.

Produces a broadcast-quality MP4 with:
  • Semi-transparent mask overlay (object highlight)
  • Glowing motion-centroid marker (the "Unsupervised Attention" indicator)
  • HUD text: current FPS, motion saliency score, processing mode
  • Optional side-by-side layout (original | annotated)
  • Handles variable aspect ratios and resolutions (Zoom recordings, 4:3, 16:9, etc.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from uvos_engine import FrameResult, VideoMeta


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VisualizerConfig:
    # Mask overlay colour (BGR) and opacity
    mask_color_bgr: tuple[int, int, int] = (0, 230, 120)   # vivid green
    mask_alpha: float = 0.45                                 # 0 = invisible, 1 = solid

    # Centroid marker
    centroid_color_bgr: tuple[int, int, int] = (0, 200, 255)  # amber/cyan
    centroid_radius: int = 10       # inner dot radius
    centroid_glow_rings: int = 3    # number of expanding glow rings
    glow_ring_step: int = 8         # px gap between rings
    crosshair_length: int = 24      # half-length of crosshair arms
    crosshair_thickness: int = 2

    # HUD text
    font: int = cv2.FONT_HERSHEY_DUPLEX
    font_scale: float = 0.55
    font_thickness: int = 1
    hud_color_bgr: tuple[int, int, int] = (220, 220, 220)
    hud_bg_color_bgr: tuple[int, int, int] = (20, 20, 20)
    hud_bg_alpha: float = 0.65
    hud_padding: int = 8
    hud_line_spacing: int = 4

    # Branding watermark (set to "" to disable)
    watermark_text: str = "Motion-as-Attention | UVOS"
    watermark_alpha: float = 0.55

    # Layout
    side_by_side: bool = False
    side_by_side_gap: int = 4          # px gap between panels

    # Video encoding
    output_fps: Optional[float] = None  # None = use source FPS
    codec: str = "mp4v"                 # 'mp4v' works on all platforms; 'avc1' for H.264
    jpeg_quality: int = 95             # internal frame quality (not output video)

    # Processing mode label shown in HUD
    processing_mode: str = "RAFT + SAM 2 (UVOS)"


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

class VideoVisualizer:
    """
    Renders a UVOS result sequence into an annotated demo MP4.

    Usage::

        viz = VideoVisualizer(VisualizerConfig(side_by_side=True))
        viz.create_demo_video("input.mp4", results, "demo.mp4", meta)
    """

    def __init__(self, config: Optional[VisualizerConfig] = None) -> None:
        self.cfg = config or VisualizerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_demo_video(
        self,
        source_video_path: str,
        results: list[FrameResult],
        output_path: str,
        meta: VideoMeta,
        pipeline_fps: float = 0.0,
    ) -> None:
        """
        Read source frames, apply overlays, and write the demo MP4.

        Args:
            source_video_path: Original video (for reading raw pixels).
            results:           Per-frame results from UVOSEngine.
            output_path:       Destination MP4 path.
            meta:              VideoMeta (fps, resolution, …).
            pipeline_fps:      Actual inference throughput to display in HUD.
                               If 0.0, computed from ``len(results) / elapsed`` is used.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        out_fps = self.cfg.output_fps or meta.fps
        out_w, out_h = self._compute_output_dims(meta)

        fourcc = cv2.VideoWriter_fourcc(*self.cfg.codec)
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter failed to open: {output_path}")

        cap = cv2.VideoCapture(source_video_path)
        result_map = {r.frame_idx: r for r in results}

        display_fps = pipeline_fps if pipeline_fps > 0 else out_fps

        # Saliency normaliser: running max for display range [0, 1]
        max_saliency = max((r.saliency_score for r in results), default=1.0) or 1.0

        with tqdm(total=meta.num_frames, desc="Rendering demo", unit="frame") as pbar:
            frame_idx = 0
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                result = result_map.get(frame_idx)

                annotated_bgr = self.render_frame(bgr, result, display_fps, max_saliency)

                if self.cfg.side_by_side:
                    out_frame = self._make_side_by_side(bgr, annotated_bgr, meta)
                else:
                    out_frame = annotated_bgr

                # Resize to target if needed (handles variable-resolution Zoom clips)
                if out_frame.shape[1] != out_w or out_frame.shape[0] != out_h:
                    out_frame = cv2.resize(out_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                writer.write(out_frame)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        writer.release()
        print(f"\nDemo saved → {output_path}  ({frame_idx} frames, {out_fps:.1f} fps)")

    def render_frame(
        self,
        bgr: np.ndarray,
        result: Optional[FrameResult],
        display_fps: float = 0.0,
        max_saliency: float = 1.0,
    ) -> np.ndarray:
        """
        Apply all visual overlays to a single BGR frame.

        This method is also usable standalone (e.g., in a live demo loop).

        Args:
            bgr:          Raw source frame (BGR uint8).
            result:       Corresponding FrameResult (or None for no overlay).
            display_fps:  FPS value to show in HUD.
            max_saliency: Normalisation constant for saliency bar.

        Returns:
            Annotated BGR frame.
        """
        canvas = bgr.copy()

        if result is not None:
            canvas = self._draw_mask(canvas, result.mask)
            canvas = self._draw_centroid(canvas, result.centroid_xy)
            canvas = self._draw_hud(canvas, result, display_fps, max_saliency)

        canvas = self._draw_watermark(canvas)
        return canvas

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_mask(self, canvas: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Blend a semi-transparent coloured mask over the canvas."""
        if not mask.any():
            return canvas

        color = np.array(self.cfg.mask_color_bgr, dtype=np.uint8)
        overlay = canvas.copy()
        overlay[mask] = color

        # Soft boundary: erode mask and draw a 1-px brighter edge
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundary = (mask.astype(np.uint8) - eroded).astype(bool)

        blended = cv2.addWeighted(canvas, 1 - self.cfg.mask_alpha, overlay, self.cfg.mask_alpha, 0)

        # Bright boundary ring
        boundary_color = tuple(min(255, int(c * 1.5)) for c in self.cfg.mask_color_bgr)
        blended[boundary] = boundary_color

        return blended

    def _draw_centroid(
        self,
        canvas: np.ndarray,
        centroid_xy: tuple[float, float],
    ) -> np.ndarray:
        """Draw a glowing dot + crosshair at the motion centroid."""
        cx, cy = int(round(centroid_xy[0])), int(round(centroid_xy[1]))
        h, w = canvas.shape[:2]
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)

        color = self.cfg.centroid_color_bgr
        r = self.cfg.centroid_radius
        rings = self.cfg.centroid_glow_rings
        step = self.cfg.glow_ring_step

        glow_layer = canvas.copy()

        # Glow rings (decreasing opacity outward)
        for i in range(rings, 0, -1):
            ring_r = r + i * step
            ring_alpha = 0.20 - i * 0.04
            cv2.circle(glow_layer, (cx, cy), ring_r, color, thickness=2,
                       lineType=cv2.LINE_AA)
            canvas = cv2.addWeighted(canvas, 1 - ring_alpha, glow_layer, ring_alpha, 0)
            glow_layer = canvas.copy()

        # Solid inner dot
        cv2.circle(canvas, (cx, cy), r, color, thickness=-1, lineType=cv2.LINE_AA)
        # White centre highlight
        cv2.circle(canvas, (cx, cy), max(2, r // 3), (255, 255, 255), thickness=-1,
                   lineType=cv2.LINE_AA)

        # Crosshair arms
        arm = self.cfg.crosshair_length
        thick = self.cfg.crosshair_thickness
        gap = r + 4
        cv2.line(canvas, (cx - arm, cy), (cx - gap, cy), color, thick, cv2.LINE_AA)
        cv2.line(canvas, (cx + gap, cy), (cx + arm, cy), color, thick, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - arm), (cx, cy - gap), color, thick, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy + gap), (cx, cy + arm), color, thick, cv2.LINE_AA)

        return canvas

    def _draw_hud(
        self,
        canvas: np.ndarray,
        result: FrameResult,
        display_fps: float,
        max_saliency: float,
    ) -> np.ndarray:
        """Draw the HUD panel (top-left corner)."""
        cfg = self.cfg
        saliency_norm = min(1.0, result.saliency_score / max(max_saliency, 1e-6))

        hud_lines = [
            f"FPS    {display_fps:>6.1f}",
            f"Sal    {saliency_norm * 100:>5.1f}%",
            f"Mode   {cfg.processing_mode}",
            f"Frame  {result.frame_idx:>5d}",
        ]

        font = cfg.font
        fscale = cfg.font_scale
        fthick = cfg.font_thickness
        pad = cfg.hud_padding
        line_gap = cfg.hud_line_spacing

        # Measure text block
        line_sizes = [cv2.getTextSize(ln, font, fscale, fthick)[0] for ln in hud_lines]
        max_w = max(s[0] for s in line_sizes)
        line_h = max(s[1] for s in line_sizes)
        total_h = len(hud_lines) * (line_h + line_gap) + 2 * pad
        total_w = max_w + 2 * pad

        # Semi-transparent background
        bg = canvas.copy()
        cv2.rectangle(bg, (0, 0), (total_w, total_h), cfg.hud_bg_color_bgr, -1)
        canvas = cv2.addWeighted(canvas, 1 - cfg.hud_bg_alpha, bg, cfg.hud_bg_alpha, 0)

        # Accent bar on the left edge
        cv2.rectangle(canvas, (0, 0), (3, total_h), cfg.centroid_color_bgr, -1)

        # Text lines
        y = pad + line_h
        for ln in hud_lines:
            cv2.putText(canvas, ln, (pad + 6, y), font, fscale,
                        cfg.hud_color_bgr, fthick, cv2.LINE_AA)
            y += line_h + line_gap

        return canvas

    def _draw_watermark(self, canvas: np.ndarray) -> np.ndarray:
        """Draw a small watermark in the bottom-right corner."""
        if not self.cfg.watermark_text:
            return canvas

        font = self.cfg.font
        fscale = self.cfg.font_scale * 0.75
        fthick = 1
        text = self.cfg.watermark_text

        (tw, th), _ = cv2.getTextSize(text, font, fscale, fthick)
        h, w = canvas.shape[:2]
        margin = 8
        x = w - tw - margin
        y = h - margin

        # Shadow
        cv2.putText(canvas, text, (x + 1, y + 1), font, fscale, (0, 0, 0), fthick, cv2.LINE_AA)
        # Text
        alpha = self.cfg.watermark_alpha
        layer = canvas.copy()
        cv2.putText(layer, text, (x, y), font, fscale,
                    self.cfg.hud_color_bgr, fthick, cv2.LINE_AA)
        canvas = cv2.addWeighted(canvas, 1 - alpha, layer, alpha, 0)
        return canvas

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _make_side_by_side(
        self,
        original_bgr: np.ndarray,
        annotated_bgr: np.ndarray,
        meta: VideoMeta,
    ) -> np.ndarray:
        """Concatenate original and annotated frames horizontally."""
        h, w = meta.height, meta.width
        gap = self.cfg.side_by_side_gap

        # Ensure both panels have the same height
        left = cv2.resize(original_bgr, (w, h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(annotated_bgr, (w, h), interpolation=cv2.INTER_AREA)

        separator = np.zeros((h, gap, 3), dtype=np.uint8)

        # Label banners
        left = self._draw_panel_label(left, "Original")
        right = self._draw_panel_label(right, "UVOS")

        return np.concatenate([left, separator, right], axis=1)

    @staticmethod
    def _draw_panel_label(panel: np.ndarray, label: str) -> np.ndarray:
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        x, y = 8, th + 8
        cv2.rectangle(panel, (0, 0), (tw + 16, th + 16), (20, 20, 20), -1)
        cv2.putText(panel, label, (x, y + 4), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return panel

    def _compute_output_dims(self, meta: VideoMeta) -> tuple[int, int]:
        """Return (width, height) of the output video."""
        if self.cfg.side_by_side:
            return (meta.width * 2 + self.cfg.side_by_side_gap, meta.height)
        return (meta.width, meta.height)
