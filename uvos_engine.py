"""
uvos_engine.py — RAFT + SAM 2 Unsupervised Video Object Segmentation Engine

Motion-as-Attention pipeline:
  1. RAFT optical flow → Weighted Motion Centroid (unsupervised attention prompt)
  2. SAM 2 video predictor → mask initialisation + propagation

Usage (library):
    from uvos_engine import UVOSEngine, EngineConfig
    engine = UVOSEngine(EngineConfig())
    results, meta = engine.process_video("input.mp4")
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SAM 2 model identifiers → (config_name, checkpoint_url)
# ---------------------------------------------------------------------------
_SAM2_HF_IDS = {
    "tiny":       "facebook/sam2.1-hiera-tiny",
    "small":      "facebook/sam2.1-hiera-small",
    "base_plus":  "facebook/sam2.1-hiera-base-plus",
    "large":      "facebook/sam2.1-hiera-large",
    # accept full HF IDs directly
    "facebook/sam2.1-hiera-tiny":      "facebook/sam2.1-hiera-tiny",
    "facebook/sam2.1-hiera-small":     "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-base-plus": "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2.1-hiera-large":     "facebook/sam2.1-hiera-large",
}

_SAM2_DIRECT_URLS = {
    "facebook/sam2.1-hiera-tiny":
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "facebook/sam2.1-hiera-small":
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "facebook/sam2.1-hiera-base-plus":
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "facebook/sam2.1-hiera-large":
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

_SAM2_CONFIGS = {
    "facebook/sam2.1-hiera-tiny":      "configs/sam2.1/sam2.1_hiera_t",
    "facebook/sam2.1-hiera-small":     "configs/sam2.1/sam2.1_hiera_s",
    "facebook/sam2.1-hiera-base-plus": "configs/sam2.1/sam2.1_hiera_b+",
    "facebook/sam2.1-hiera-large":     "configs/sam2.1/sam2.1_hiera_l",
}


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """All runtime knobs for the UVOS engine."""

    device: str = "cuda"

    # SAM 2 model variant (alias or full HF ID)
    sam2_model: str = "small"

    # Directory to cache SAM 2 checkpoints; ``None`` → system temp
    checkpoint_dir: Optional[str] = None

    # RAFT inference iterations (more = better quality, slower)
    raft_iters: int = 20

    # Automatic Mixed Precision (float16 on T4)
    use_amp: bool = True

    # torch.compile for SAM 2 image encoder (warm-up on first call ~60 s)
    compile_sam2: bool = False

    # Top-k% motion pixels used to weight the centroid (0–1)
    top_k_motion: float = 0.10

    # Prompt-frame selection: -1 = auto (highest saliency), ≥0 = fixed frame index
    prompt_frame_idx: int = 0

    # If True, propagate backward from the prompt frame in addition to forward
    bidirectional_propagation: bool = True

    # Free RAFT GPU memory before running SAM 2 (useful if VRAM < 10 GB)
    offload_raft_after_flow: bool = True


@dataclass
class VideoMeta:
    """Metadata extracted from the source video."""

    fps: float
    width: int
    height: int
    num_frames: int
    fourcc: str


@dataclass
class FrameResult:
    """Per-frame segmentation result."""

    frame_idx: int
    centroid_xy: tuple[float, float]   # (x, y) in pixel coordinates
    saliency_score: float               # mean optical-flow magnitude
    mask: np.ndarray                    # bool ndarray [H, W]
    flow_magnitude: Optional[np.ndarray] = None  # float32 [H, W], None for prompt frame


# ---------------------------------------------------------------------------
# RAFT Motion Analyser
# ---------------------------------------------------------------------------

class RAFTMotionAnalyzer:
    """
    Wraps torchvision's ``raft_large`` to compute per-frame optical flow and
    derive the Weighted Motion Centroid — the unsupervised "attention prompt."
    """

    def __init__(self, config: EngineConfig) -> None:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        self.device = torch.device(config.device)
        self.top_k = config.top_k_motion
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.num_flow_updates = config.raft_iters

        log.info("Loading RAFT-large (pretrained, %d iters)…", config.raft_iters)
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = raft_large(weights=weights, progress=True)
        self.model = self.model.to(self.device).eval()

    # ------------------------------------------------------------------

    @torch.inference_mode()
    def compute_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> np.ndarray:
        """
        Compute optical flow from frame1 → frame2.

        Args:
            frame1, frame2: uint8 RGB numpy arrays [H, W, 3].

        Returns:
            flow: float32 numpy array [2, H, W] — (dx, dy) in pixels.
        """
        t1 = self._to_tensor(frame1).unsqueeze(0)   # [1, C, H, W]
        t2 = self._to_tensor(frame2).unsqueeze(0)
        t1_t, t2_t = self.transforms(t1, t2)
        t1_t = self._pad_to_8(t1_t).to(self.device)
        t2_t = self._pad_to_8(t2_t).to(self.device)

        with torch.autocast(device_type=self.device.type,
                            dtype=torch.float16,
                            enabled=self.use_amp):
            flow_list = self.model(t1_t, t2_t, num_flow_updates=self.num_flow_updates)

        h_orig, w_orig = frame1.shape[:2]
        flow = flow_list[-1].squeeze(0).float().cpu().numpy()  # [2, H_pad, W_pad]
        return flow[:, :h_orig, :w_orig]

    @torch.inference_mode()
    def compute_flow_batch(
        self,
        frames: list[np.ndarray],
        batch_size: int = 4,
    ) -> list[np.ndarray]:
        """
        Compute optical flow for all consecutive pairs in a frame list.

        Returns:
            flows: list of float32 arrays [2, H, W], length = len(frames) - 1.
        """
        N = len(frames)
        if N < 2:
            return []

        flows: list[np.ndarray] = []
        tensors = [self._to_tensor(f) for f in frames]

        for start in tqdm(range(0, N - 1, batch_size), desc="RAFT optical flow", unit="batch"):
            end = min(start + batch_size, N - 1)
            batch_t1 = torch.stack(tensors[start:end])
            batch_t2 = torch.stack(tensors[start + 1: end + 1])

            t1_t, t2_t = self.transforms(batch_t1, batch_t2)
            t1_t = self._pad_to_8(t1_t).to(self.device)
            t2_t = self._pad_to_8(t2_t).to(self.device)

            with torch.autocast(device_type=self.device.type,
                                dtype=torch.float16,
                                enabled=self.use_amp):
                flow_list = self.model(t1_t, t2_t, num_flow_updates=self.num_flow_updates)

            batch_flows = flow_list[-1].float().cpu().numpy()  # [B, 2, H_pad, W_pad]
            for i in range(batch_flows.shape[0]):
                h, w = frames[start + i].shape[:2]
                flow = batch_flows[i, :, :h, :w]  # crop padding
                flows.append(flow)

        return flows

    # ------------------------------------------------------------------

    def compute_motion_centroid(
        self,
        flow: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray]:
        """
        Derive the Weighted Motion Centroid from an optical-flow field.

        Strategy:
            1. Compute flow magnitude map.
            2. Keep only the top-k% highest-magnitude pixels as weights.
            3. Compute the intensity-weighted centroid (x, y).

        Returns:
            (cx, cy, saliency_score, magnitude_map)
            saliency_score = mean magnitude over top-k% region.
        """
        mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2).astype(np.float32)  # [H, W]
        H, W = mag.shape

        flat = mag.ravel()
        threshold_idx = max(0, int((1.0 - self.top_k) * flat.size) - 1)
        threshold = np.partition(flat, threshold_idx)[threshold_idx]

        # Binary mask of top-k pixels, weighted by their magnitude (robust to ties)
        top_k_mask = mag >= threshold
        weights = mag * top_k_mask.astype(np.float32)
        total = weights.sum()

        if total < 1e-6:
            return W / 2.0, H / 2.0, 0.0, mag

        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        cx = float((weights * xs).sum() / total)
        cy = float((weights * ys).sum() / total)
        saliency = float(mag[top_k_mask].mean()) if top_k_mask.any() else 0.0

        return cx, cy, saliency, mag

    # ------------------------------------------------------------------

    def free_gpu_memory(self) -> None:
        """Move model to CPU and release GPU memory."""
        self.model.cpu()
        torch.cuda.empty_cache()
        log.info("RAFT moved to CPU; GPU memory released.")

    @staticmethod
    def _pad_to_8(t: torch.Tensor) -> torch.Tensor:
        """Pad spatial dims to the nearest multiple of 8 (required by RAFT)."""
        _, _, h, w = t.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
        return t

    @staticmethod
    def _to_tensor(frame: np.ndarray) -> torch.Tensor:
        """Convert uint8 RGB [H, W, 3] → uint8 tensor [3, H, W]."""
        return torch.from_numpy(frame).permute(2, 0, 1)  # [3, H, W]


# ---------------------------------------------------------------------------
# SAM 2 Segmentor
# ---------------------------------------------------------------------------

class SAM2Segmentor:
    """
    Wraps the SAM 2 video predictor.

    Loads the model once; call ``segment_video()`` for each video.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.device = torch.device(config.device)
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self._predictor = self._build_predictor(config)

    # ------------------------------------------------------------------

    def segment_video(
        self,
        frames_dir: str,
        prompt_frame_idx: int,
        centroid_xy: tuple[float, float],
        num_frames: int,
        bidirectional: bool = True,
    ) -> dict[int, np.ndarray]:
        """
        Run SAM 2 on a directory of JPEG frames.

        Args:
            frames_dir:        Path to directory with ``00000.jpg``, ``00001.jpg``, …
            prompt_frame_idx:  Frame index where the motion centroid is injected.
            centroid_xy:       (x, y) pixel coordinates of the motion centroid.
            num_frames:        Total number of frames.
            bidirectional:     If True, also propagate backward from the prompt frame.

        Returns:
            masks: dict mapping frame_idx → bool ndarray [H, W].
        """
        cx, cy = centroid_xy
        predictor = self._predictor

        amp_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        )

        with torch.inference_mode(), amp_ctx:
            state = predictor.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
            )

            # Inject motion centroid as the foreground point prompt
            points = np.array([[cx, cy]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)  # 1 = foreground

            predictor.add_new_points_or_box(
                state,
                frame_idx=prompt_frame_idx,
                obj_id=1,
                points=points,
                labels=labels,
            )

            masks: dict[int, np.ndarray] = {}

            # Forward propagation
            for frame_idx, _, video_masks in predictor.propagate_in_video(state):
                binary = (video_masks[0, 0] > 0.0).cpu().numpy().astype(bool)
                masks[frame_idx] = binary

            # Backward propagation (if prompt is not on frame 0)
            if bidirectional and prompt_frame_idx > 0:
                for frame_idx, _, video_masks in predictor.propagate_in_video(
                    state, reverse=True
                ):
                    if frame_idx not in masks:
                        binary = (video_masks[0, 0] > 0.0).cpu().numpy().astype(bool)
                        masks[frame_idx] = binary

            predictor.reset_state(state)

        return masks

    # ------------------------------------------------------------------

    @staticmethod
    def _build_predictor(config: EngineConfig):
        """Build and return the SAM 2 video predictor (downloads checkpoint if needed)."""
        from sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_hf

        hf_id = _SAM2_HF_IDS.get(config.sam2_model, config.sam2_model)
        log.info("Loading SAM 2 predictor: %s", hf_id)

        try:
            predictor = build_sam2_video_predictor_hf(
                hf_id,
                device=config.device,
                vos_optimized=config.compile_sam2,
            )
            log.info("SAM 2 predictor loaded via HuggingFace Hub.")
            return predictor
        except Exception as hf_err:
            log.warning("HF download failed (%s). Trying direct URL fallback…", hf_err)

        # Fallback: direct download from fbaipublicfiles.com
        return SAM2Segmentor._build_from_direct_url(hf_id, config)

    @staticmethod
    def _build_from_direct_url(hf_id: str, config: EngineConfig):
        """Download checkpoint directly and build predictor."""
        import urllib.request
        from sam2.build_sam import build_sam2_video_predictor

        url = _SAM2_DIRECT_URLS[hf_id]
        config_name = _SAM2_CONFIGS[hf_id]

        cache_dir = Path(config.checkpoint_dir or "~/.cache/uvos/checkpoints").expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = cache_dir / Path(url).name

        if not ckpt_path.exists():
            log.info("Downloading checkpoint from %s …", url)

            def _reporthook(count, block_size, total):
                pct = min(100, int(count * block_size * 100 / total))
                print(f"\r  {pct}%", end="", flush=True)

            urllib.request.urlretrieve(url, ckpt_path, _reporthook)
            print()

        predictor = build_sam2_video_predictor(
            config_file=config_name,
            ckpt_path=str(ckpt_path),
            device=config.device,
            vos_optimized=config.compile_sam2,
        )
        log.info("SAM 2 predictor loaded from local cache.")
        return predictor


# ---------------------------------------------------------------------------
# UVOS Engine (orchestrator)
# ---------------------------------------------------------------------------

class UVOSEngine:
    """
    End-to-end UVOS pipeline: RAFT optical flow → motion centroid → SAM 2 segmentation.

    Example::

        engine = UVOSEngine(EngineConfig(sam2_model="small"))
        results, meta = engine.process_video("talk.mp4")
    """

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        log.info("Initialising UVOS engine on device=%s", self.config.device)
        self.raft = RAFTMotionAnalyzer(self.config)
        self.sam2 = SAM2Segmentor(self.config)

    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        prompt_frame_idx: Optional[int] = None,
    ) -> tuple[list[FrameResult], VideoMeta]:
        """
        Run the full UVOS pipeline on a video file.

        Args:
            video_path:        Path to input video (MP4, AVI, MOV, …).
            prompt_frame_idx:  Frame to use as SAM 2 prompt.  ``None`` → use
                               ``config.prompt_frame_idx`` (-1 = auto-select by
                               highest motion saliency).

        Returns:
            (results, meta)
            results: list of FrameResult, one per frame.
            meta:    VideoMeta with fps, resolution, etc.
        """
        if prompt_frame_idx is None:
            prompt_frame_idx = self.config.prompt_frame_idx

        t_total = time.perf_counter()

        # ── 1. Load frames ──────────────────────────────────────────────
        log.info("Loading frames from: %s", video_path)
        frames, meta = self._load_video(video_path)
        log.info("Loaded %d frames  (%dx%d @ %.2f fps)", meta.num_frames, meta.width, meta.height, meta.fps)

        # ── 2. Compute optical flow (all consecutive pairs) ─────────────
        log.info("Computing RAFT optical flow…")
        flows = self.raft.compute_flow_batch(frames)

        # Centroid per frame.  Frame i gets flow[i] (flow from i→i+1),
        # except the last frame which reuses flow[-1] (flow from N-2→N-1).
        centroids: list[tuple[float, float, float, np.ndarray]] = []
        for i, flow in enumerate(flows):
            cx, cy, sal, mag = self.raft.compute_motion_centroid(flow)
            centroids.append((cx, cy, sal, mag))
        # Duplicate last entry for the final frame
        centroids.append(centroids[-1] if centroids else (meta.width / 2, meta.height / 2, 0.0, None))

        if self.config.offload_raft_after_flow:
            self.raft.free_gpu_memory()

        # ── 3. Select prompt frame ──────────────────────────────────────
        if prompt_frame_idx == -1:
            prompt_frame_idx = int(np.argmax([c[2] for c in centroids]))
            log.info("Auto-selected prompt frame: %d (saliency=%.3f)", prompt_frame_idx, centroids[prompt_frame_idx][2])
        else:
            prompt_frame_idx = min(prompt_frame_idx, meta.num_frames - 1)

        cx_prompt, cy_prompt, _, _ = centroids[prompt_frame_idx]
        log.info("Prompt frame %d | centroid=(%.1f, %.1f)", prompt_frame_idx, cx_prompt, cy_prompt)

        # ── 4. Write JPEG frames for SAM 2 ─────────────────────────────
        log.info("Writing frames for SAM 2…")
        tmp_dir = tempfile.mkdtemp(prefix="uvos_frames_")
        try:
            self._save_frames_to_dir(frames, tmp_dir)

            # ── 5. SAM 2 propagation ────────────────────────────────────
            log.info("Running SAM 2 propagation…")
            masks = self.sam2.segment_video(
                frames_dir=tmp_dir,
                prompt_frame_idx=prompt_frame_idx,
                centroid_xy=(cx_prompt, cy_prompt),
                num_frames=meta.num_frames,
                bidirectional=self.config.bidirectional_propagation,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # ── 6. Package results ──────────────────────────────────────────
        results: list[FrameResult] = []
        H, W = meta.height, meta.width
        empty_mask = np.zeros((H, W), dtype=bool)

        for i in range(meta.num_frames):
            cx, cy, sal, mag = centroids[i]
            mask = masks.get(i, empty_mask)

            # Ensure mask matches frame resolution (SAM 2 may up-sample)
            if mask.shape != (H, W):
                mask_uint8 = mask.astype(np.uint8) * 255
                mask_resized = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = mask_resized > 127

            results.append(FrameResult(
                frame_idx=i,
                centroid_xy=(cx, cy),
                saliency_score=sal,
                mask=mask,
                flow_magnitude=mag,
            ))

        elapsed = time.perf_counter() - t_total
        log.info(
            "Pipeline complete in %.1f s  (%.2f fps effective)",
            elapsed,
            meta.num_frames / elapsed,
        )
        return results, meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_video(video_path: str) -> tuple[list[np.ndarray], VideoMeta]:
        """Read all frames with OpenCV.  Returns RGB uint8 frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])

        frames: list[np.ndarray] = []
        with tqdm(total=num_frames_hint or None, desc="Reading video", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pbar.update(1)

        cap.release()

        if not frames:
            raise ValueError(f"No frames could be read from: {video_path}")

        meta = VideoMeta(
            fps=fps,
            width=width,
            height=height,
            num_frames=len(frames),
            fourcc=fourcc,
        )
        return frames, meta

    @staticmethod
    def _save_frames_to_dir(frames: list[np.ndarray], out_dir: str) -> None:
        """Write RGB frames as zero-padded JPEG files: 00000.jpg, 00001.jpg, …"""
        n_digits = len(str(len(frames) - 1))
        n_digits = max(n_digits, 5)  # SAM 2 convention: 5 digits minimum

        for i, frame in enumerate(tqdm(frames, desc="Saving frames", unit="frame", leave=False)):
            name = str(i).zfill(n_digits) + ".jpg"
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
