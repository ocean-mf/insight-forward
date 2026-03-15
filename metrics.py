"""
metrics.py — J-measure (mIoU) and F-measure (contour F1) evaluation.

Implements the standard DAVIS benchmark metrics:
  J  (Jaccard / region similarity) = |pred ∩ gt| / |pred ∪ gt|
  F  (contour-based F-measure)     = 2 · precision · recall / (precision + recall)
  J&F score = (J + F) / 2          ← headline benchmark number

Usage examples:

  # Evaluate a single sequence
  python metrics.py \\
      --pred  results/masks/ \\
      --gt    data/DAVIS/Annotations_unsupervised/480p/bear/ \\
      --name  bear

  # Evaluate all sequences in DAVIS split
  python metrics.py \\
      --pred-root  results/ \\
      --gt-root    data/DAVIS/Annotations_unsupervised/480p/ \\
      --output     metrics_report.csv \\
      --split      val

  # Compute only J (faster, no contour operations)
  python metrics.py --pred results/masks/ --gt data/gt/ --j-only
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_j_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Jaccard index (region similarity / mIoU) between two binary masks.

    Args:
        pred: bool or uint8 ndarray [H, W].
        gt:   bool or uint8 ndarray [H, W], same shape as pred.

    Returns:
        J ∈ [0, 1].  Returns 1.0 if both masks are empty.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = (pred & gt).sum()
    union = (pred | gt).sum()

    if union == 0:
        return 1.0  # both empty → perfect agreement

    return float(intersection) / float(union)


def _mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """
    Extract boundary pixels via morphological erosion.

    boundary = mask − erode(mask)

    The dilation ratio scales with image diagonal, following the DAVIS
    benchmark convention.
    """
    mask_u8 = mask.astype(np.uint8)
    h, w = mask_u8.shape
    img_diag = float(np.sqrt(h ** 2 + w ** 2))
    dilation_px = max(1, int(round(dilation_ratio * img_diag)))

    # Add 1-px border to avoid boundary artefacts
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * dilation_px + 1, 2 * dilation_px + 1))
    eroded = cv2.erode(padded, kernel)
    eroded = eroded[1:-1, 1:-1]

    boundary = mask_u8 - np.clip(eroded, 0, mask_u8)
    return boundary.astype(bool)


def compute_f_measure(
    pred: np.ndarray,
    gt: np.ndarray,
    dilation_ratio: float = 0.02,
) -> float:
    """
    Boundary-based F-measure (contour F1) following the DAVIS benchmark.

    Precision = |boundary(pred) ∩ dilated_boundary(gt)| / |boundary(pred)|
    Recall    = |boundary(gt)   ∩ dilated_boundary(pred)| / |boundary(gt)|
    F         = 2·P·R / (P + R)

    Args:
        pred:           bool or uint8 ndarray [H, W].
        gt:             bool or uint8 ndarray [H, W].
        dilation_ratio: boundary width relative to image diagonal.

    Returns:
        F ∈ [0, 1].  Returns 1.0 if both boundaries are empty.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_bound = _mask_to_boundary(pred, dilation_ratio)
    gt_bound = _mask_to_boundary(gt, dilation_ratio)

    # Dilate boundaries for tolerance matching
    h, w = pred.shape
    img_diag = float(np.sqrt(h ** 2 + w ** 2))
    dilation_px = max(1, int(round(dilation_ratio * img_diag)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * dilation_px + 1, 2 * dilation_px + 1))

    pred_bound_dil = cv2.dilate(pred_bound.astype(np.uint8), kernel).astype(bool)
    gt_bound_dil = cv2.dilate(gt_bound.astype(np.uint8), kernel).astype(bool)

    if pred_bound.sum() == 0 and gt_bound.sum() == 0:
        return 1.0

    precision = float((pred_bound & gt_bound_dil).sum()) / max(float(pred_bound.sum()), 1e-6)
    recall = float((gt_bound & pred_bound_dil).sum()) / max(float(gt_bound.sum()), 1e-6)

    if precision + recall < 1e-9:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Sequence-level evaluation
# ---------------------------------------------------------------------------

@dataclass
class SequenceMetrics:
    """Per-sequence evaluation result."""

    name: str
    num_frames: int

    j_scores: list[float]   # per-frame J values
    f_scores: list[float]   # per-frame F values

    @property
    def mean_j(self) -> float:
        return float(np.mean(self.j_scores)) if self.j_scores else 0.0

    @property
    def mean_f(self) -> float:
        return float(np.mean(self.f_scores)) if self.f_scores else 0.0

    @property
    def jf_score(self) -> float:
        return (self.mean_j + self.mean_f) / 2.0

    @property
    def recall_j(self) -> float:
        """Fraction of frames with J ≥ 0.5 (DAVIS "recall" metric)."""
        if not self.j_scores:
            return 0.0
        return float(sum(j >= 0.5 for j in self.j_scores)) / len(self.j_scores)

    def summary(self) -> str:
        return (
            f"{self.name:30s}  "
            f"J={self.mean_j:.3f}  "
            f"F={self.mean_f:.3f}  "
            f"J&F={self.jf_score:.3f}  "
            f"recall_J={self.recall_j:.2f}  "
            f"({self.num_frames} frames)"
        )


def evaluate_sequence(
    pred_dir: str,
    gt_dir: str,
    sequence_name: str = "",
    j_only: bool = False,
) -> SequenceMetrics:
    """
    Evaluate predicted masks against ground-truth for a single video sequence.

    Args:
        pred_dir:       Directory containing predicted PNG masks (00000.png, …)
        gt_dir:         Directory containing GT PNG masks (same naming convention)
        sequence_name:  Label for reporting.
        j_only:         Skip F-measure (faster).

    Returns:
        SequenceMetrics with per-frame J and F scores.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    pred_files = sorted(pred_dir.glob("*.png"), key=lambda p: int(p.stem))
    gt_files = sorted(gt_dir.glob("*.png"), key=lambda p: int(p.stem))

    if not pred_files:
        raise FileNotFoundError(f"No PNG masks found in pred_dir: {pred_dir}")
    if not gt_files:
        raise FileNotFoundError(f"No PNG masks found in gt_dir: {gt_dir}")

    # Align by frame index (GT may include more frames)
    gt_map = {int(p.stem): p for p in gt_files}
    pred_map = {int(p.stem): p for p in pred_files}

    frame_indices = sorted(pred_map.keys() & gt_map.keys())
    if not frame_indices:
        # Fallback: align by position
        n = min(len(pred_files), len(gt_files))
        frame_indices = list(range(n))
        pred_map = {i: pred_files[i] for i in frame_indices}
        gt_map = {i: gt_files[i] for i in frame_indices}

    j_scores: list[float] = []
    f_scores: list[float] = []

    for idx in tqdm(frame_indices, desc=f"Evaluating {sequence_name or pred_dir.name}", leave=False):
        pred_img = cv2.imread(str(pred_map[idx]), cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(str(gt_map[idx]), cv2.IMREAD_GRAYSCALE)

        if pred_img is None or gt_img is None:
            log.warning("Could not read mask pair at index %d — skipping.", idx)
            continue

        # Resize prediction to match GT resolution if necessary
        if pred_img.shape != gt_img.shape:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        pred_bin = pred_img > 127
        gt_bin = gt_img > 0

        j = compute_j_measure(pred_bin, gt_bin)
        j_scores.append(j)

        if not j_only:
            f = compute_f_measure(pred_bin, gt_bin)
            f_scores.append(f)

    return SequenceMetrics(
        name=sequence_name or str(pred_dir.name),
        num_frames=len(frame_indices),
        j_scores=j_scores,
        f_scores=f_scores if not j_only else [],
    )


def evaluate_dataset(
    pred_root: str,
    gt_root: str,
    sequences: Optional[list[str]] = None,
    j_only: bool = False,
) -> list[SequenceMetrics]:
    """
    Evaluate all sequences under pred_root against gt_root.

    Args:
        pred_root:  Parent directory containing one sub-dir per sequence.
        gt_root:    Parent directory containing one sub-dir per sequence (GT).
        sequences:  If given, evaluate only these sequence names.
        j_only:     Skip F-measure.

    Returns:
        List of SequenceMetrics, one per sequence.
    """
    pred_root = Path(pred_root)
    gt_root = Path(gt_root)

    if sequences is None:
        sequences = sorted(p.name for p in pred_root.iterdir() if p.is_dir())

    all_metrics: list[SequenceMetrics] = []
    for seq in tqdm(sequences, desc="Evaluating sequences"):
        pred_dir = pred_root / seq
        gt_dir = gt_root / seq
        if not pred_dir.exists():
            log.warning("Prediction directory missing: %s — skipping.", pred_dir)
            continue
        if not gt_dir.exists():
            log.warning("GT directory missing: %s — skipping.", gt_dir)
            continue
        try:
            sm = evaluate_sequence(str(pred_dir), str(gt_dir), sequence_name=seq, j_only=j_only)
            all_metrics.append(sm)
        except Exception as exc:
            log.error("Error evaluating %s: %s", seq, exc)

    return all_metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(metrics: list[SequenceMetrics], j_only: bool = False) -> None:
    """Print a formatted table to stdout."""
    print("\n" + "─" * 76)
    print(f"  {'Sequence':<30}  {'J':>6}  {'F':>6}  {'J&F':>6}  {'Recall-J':>9}  Frames")
    print("─" * 76)
    for m in metrics:
        f_str = f"{m.mean_f:.3f}" if not j_only else "  —  "
        jf_str = f"{m.jf_score:.3f}" if not j_only else "  —  "
        print(f"  {m.name:<30}  {m.mean_j:.3f}  {f_str}  {jf_str}"
              f"  {m.recall_j:>8.2f}  {m.num_frames:>6d}")
    print("─" * 76)

    # Global averages
    if metrics:
        j_global = float(np.mean([m.mean_j for m in metrics]))
        recall_global = float(np.mean([m.recall_j for m in metrics]))
        if not j_only:
            f_global = float(np.mean([m.mean_f for m in metrics]))
            jf_global = float(np.mean([m.jf_score for m in metrics]))
            print(f"  {'GLOBAL MEAN':<30}  {j_global:.3f}  {f_global:.3f}  {jf_global:.3f}"
                  f"  {recall_global:>8.2f}")
        else:
            print(f"  {'GLOBAL MEAN':<30}  {j_global:.3f}  {'—':>6}  {'—':>6}"
                  f"  {recall_global:>8.2f}")
    print("─" * 76 + "\n")


def save_csv(metrics: list[SequenceMetrics], output_path: str, j_only: bool = False) -> None:
    """Write per-sequence results to a CSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sequence", "num_frames", "mean_j", "mean_f", "jf_score", "recall_j"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            writer.writerow({
                "sequence":   m.name,
                "num_frames": m.num_frames,
                "mean_j":     f"{m.mean_j:.4f}",
                "mean_f":     f"{m.mean_f:.4f}" if not j_only else "",
                "jf_score":   f"{m.jf_score:.4f}" if not j_only else "",
                "recall_j":   f"{m.recall_j:.4f}",
            })
        if metrics:
            j_global = float(np.mean([m.mean_j for m in metrics]))
            f_global = float(np.mean([m.mean_f for m in metrics])) if not j_only else 0.0
            jf_global = float(np.mean([m.jf_score for m in metrics])) if not j_only else 0.0
            recall_global = float(np.mean([m.recall_j for m in metrics]))
            writer.writerow({
                "sequence":   "GLOBAL_MEAN",
                "num_frames": sum(m.num_frames for m in metrics),
                "mean_j":     f"{j_global:.4f}",
                "mean_f":     f"{f_global:.4f}" if not j_only else "",
                "jf_score":   f"{jf_global:.4f}" if not j_only else "",
                "recall_j":   f"{recall_global:.4f}",
            })
    print(f"Metrics saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="metrics",
        description="Evaluate UVOS segmentation quality (J, F, J&F).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    single = parser.add_argument_group("Single-sequence mode")
    single.add_argument("--pred", metavar="DIR",
                        help="Directory of predicted masks (PNG)")
    single.add_argument("--gt", metavar="DIR",
                        help="Directory of ground-truth masks (PNG)")
    single.add_argument("--name", default="", metavar="NAME",
                        help="Sequence name label  [default: directory name]")

    batch = parser.add_argument_group("Batch-dataset mode")
    batch.add_argument("--pred-root", metavar="DIR",
                       help="Root dir with one sub-dir per sequence (predictions)")
    batch.add_argument("--gt-root", metavar="DIR",
                       help="Root dir with one sub-dir per sequence (GT)")
    batch.add_argument("--sequences", nargs="+", metavar="SEQ",
                       help="Evaluate only these sequences (default: all)")

    parser.add_argument("--output", metavar="CSV",
                        help="Save results to CSV file")
    parser.add_argument("--j-only", action="store_true",
                        help="Compute J only (skip slower F-measure)")
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Single-sequence mode ──────────────────────────────────────────
    if args.pred and args.gt:
        sm = evaluate_sequence(args.pred, args.gt, sequence_name=args.name, j_only=args.j_only)
        print_summary([sm], j_only=args.j_only)
        if args.output:
            save_csv([sm], args.output, j_only=args.j_only)
        return 0

    # ── Batch mode ────────────────────────────────────────────────────
    if args.pred_root and args.gt_root:
        all_metrics = evaluate_dataset(
            args.pred_root,
            args.gt_root,
            sequences=args.sequences,
            j_only=args.j_only,
        )
        print_summary(all_metrics, j_only=args.j_only)
        if args.output:
            save_csv(all_metrics, args.output, j_only=args.j_only)
        return 0

    print("ERROR: provide either --pred + --gt  OR  --pred-root + --gt-root", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
