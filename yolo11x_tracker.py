"""YOLO11x-based object detector and tracker.

This script uses the Ultralytics YOLO11x model in tracking mode to perform
object detection and ID assignment on video streams. It can read from a
camera device, a video file, or an image sequence, and optionally saves the
annotated output. Tracking metrics are maintained to give quick feedback on
how many unique objects have been seen and how many are active in the
current frame.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tracker."""
    parser = argparse.ArgumentParser(
        description="Run YOLO11x detection and tracking on a video source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index, file path, directory, or glob pattern.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11x.pt",
        help="Path or name of the YOLO11x model to load.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run inference on (e.g., 'cuda', '0', 'cpu').",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold used for non-max suppression.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Filter by class IDs; omit for all classes.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated output alongside raw results.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/yolo11x",
        help="Directory where annotated videos will be written if --save is set.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Display the tracker output in an OpenCV window.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Only process every Nth frame from the source to save compute.",
    )
    return parser.parse_args()


class TrackingStats:
    """Minimal tracker statistics for quick feedback on progress."""

    def __init__(self) -> None:
        self.total_frames = 0
        self.seen_ids: set[int] = set()
        self._lifetimes: defaultdict[int, int] = defaultdict(int)

    def update(self, ids: Optional[Iterable[int]]) -> None:
        """Update statistics with the IDs present in the latest frame."""
        self.total_frames += 1
        if ids is None:
            return
        for track_id in ids:
            tid = int(track_id)
            self.seen_ids.add(tid)
            self._lifetimes[tid] += 1

    def avg_lifetime(self) -> float:
        """Return the mean number of frames each tracked object has stayed."""
        if not self._lifetimes:
            return 0.0
        return sum(self._lifetimes.values()) / len(self._lifetimes)


def resolve_source(source: str) -> str | int:
    """Try to cast the source string to an int for camera indices."""
    if source.isdigit():
        return int(source)
    return source


def build_writer(path: Path, frame_shape: tuple[int, int, int], fps: float) -> cv2.VideoWriter:
    """Create a video writer for saving annotated results."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frame_shape[:2]
    return cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), (width, height))


def main(args: argparse.Namespace) -> int:
    source = resolve_source(args.source)
    save_dir = Path(args.save_dir)
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    tracker_kwargs = {
        "conf": args.conf,
        "iou": args.iou,
        "classes": args.classes,
        "device": args.device or None,
        "stream": True,
        "verbose": False,
        "persist": True,
    }

    stats = TrackingStats()
    writer: Optional[cv2.VideoWriter] = None
    window_name = "YOLO11x Tracker"
    frame_index = 0

    try:
        for result in model.track(source=source, **tracker_kwargs):
            frame_index += 1
            if args.frame_stride > 1 and (frame_index - 1) % args.frame_stride:
                continue

            frame = result.orig_img
            if frame is None:
                continue

            plotted = result.plot()
            boxes = result.boxes
            track_ids = boxes.id.int().tolist() if boxes.id is not None else None
            stats.update(track_ids)

            active_count = len(track_ids) if track_ids is not None else 0
            overlay = (
                f"frames: {stats.total_frames}  "
                f"active: {active_count}  "
                f"seen: {len(stats.seen_ids)}  "
                f"avg life: {stats.avg_lifetime():.1f}"
            )
            cv2.putText(
                plotted,
                overlay,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if args.view:
                cv2.imshow(window_name, plotted)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if args.save:
                if writer is None:
                    fps = float(getattr(result, "fps", 0.0))
                    if not fps:
                        fps = 30.0
                    output_path = save_dir / f"yolo11x_{Path(str(args.source)).stem}.mp4"
                    writer = build_writer(output_path, plotted.shape, fps)
                writer.write(plotted)

    except KeyboardInterrupt:
        print("\nStopping tracking (keyboard interrupt).")
    finally:
        if writer is not None:
            writer.release()
        if args.view:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
