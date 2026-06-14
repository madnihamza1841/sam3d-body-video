#!/usr/bin/env python3
"""
sam3d_video.py
==============
End-to-end pipeline:

    video  -->  frames  -->  SAM 3D Body (facebookresearch/sam-3d-body)
                          -->  skeleton-overlay video  +  3D marker CSV

The heavy ML model is isolated behind a thin `Estimator` interface so the
surrounding plumbing (frame extraction, skeleton drawing, CSV export, video
re-assembly) can be exercised end-to-end without GPU/gated weights via
`--self-test` (uses a synthetic estimator producing the *exact* same output
schema as the real model).

Real model output schema (per detected person), see
sam_3d_body/sam_3d_body_estimator.py::process_one_image:
    bbox                (4,)            x1,y1,x2,y2
    pred_keypoints_2d   (70, 2|3)       image-pixel keypoints (+score)
    pred_keypoints_3d   (70, 3|4)       3D keypoints in camera space (meters)
    pred_joint_coords   (J, 3)          MHR skeleton joint positions
    pred_vertices       (V, 3)          mesh vertices
    pred_cam_t          (3,)            camera translation
    focal_length        scalar
    ... (pose / shape / hand params)

The 70 keypoints are named by MHR70 (sam_3d_body/metadata/mhr70.py).

Usage
-----
  # Real model (needs the cloned repo + downloaded gated checkpoints + GPU):
  python sam3d_video.py --video in.mp4 --output out/ \
         --repo /path/to/sam-3d-body --hf-repo-id facebook/sam-3d-body-dinov3

  # Verify the whole pipeline locally with a synthetic estimator:
  python sam3d_video.py --video in.mp4 --output out/ --self-test
  python sam3d_video.py --make-sample-video sample.mp4   # create a test clip
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("OpenCV is required: pip install opencv-python")


# --------------------------------------------------------------------------- #
# MHR70 keypoint metadata (mirrors sam_3d_body/metadata/mhr70.py)
# --------------------------------------------------------------------------- #
MHR70_NAMES = [
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-hip", "right-hip", "left-knee", "right-knee", "left-ankle",
    "right-ankle", "left-big-toe-tip", "left-small-toe-tip", "left-heel",
    "right-big-toe-tip", "right-small-toe-tip", "right-heel",
    "right-thumb-tip", "right-thumb-first-joint", "right-thumb-second-joint",
    "right-thumb-third-joint", "right-index-tip", "right-index-first-joint",
    "right-index-second-joint", "right-index-third-joint", "right-middle-tip",
    "right-middle-first-joint", "right-middle-second-joint",
    "right-middle-third-joint", "right-ring-tip", "right-ring-first-joint",
    "right-ring-second-joint", "right-ring-third-joint", "right-pinky-tip",
    "right-pinky-first-joint", "right-pinky-second-joint",
    "right-pinky-third-joint", "right-wrist", "left-thumb-tip",
    "left-thumb-first-joint", "left-thumb-second-joint",
    "left-thumb-third-joint", "left-index-tip", "left-index-first-joint",
    "left-index-second-joint", "left-index-third-joint", "left-middle-tip",
    "left-middle-first-joint", "left-middle-second-joint",
    "left-middle-third-joint", "left-ring-tip", "left-ring-first-joint",
    "left-ring-second-joint", "left-ring-third-joint", "left-pinky-tip",
    "left-pinky-first-joint", "left-pinky-second-joint",
    "left-pinky-third-joint", "left-wrist", "left-olecranon",
    "right-olecranon", "left-cubital-fossa", "right-cubital-fossa",
    "left-acromion", "right-acromion", "neck",
]
NAME_TO_IDX = {n: i for i, n in enumerate(MHR70_NAMES)}

# Body skeleton as (name, name) pairs -> resolved to index pairs below.
_SKELETON_NAME_LINKS = [
    ("nose", "left-eye"), ("nose", "right-eye"),
    ("left-eye", "left-ear"), ("right-eye", "right-ear"),
    ("nose", "neck"),
    ("neck", "left-shoulder"), ("neck", "right-shoulder"),
    ("left-shoulder", "right-shoulder"),
    ("left-shoulder", "left-elbow"), ("left-elbow", "left-wrist"),
    ("right-shoulder", "right-elbow"), ("right-elbow", "right-wrist"),
    ("left-shoulder", "left-hip"), ("right-shoulder", "right-hip"),
    ("left-hip", "right-hip"),
    ("left-hip", "left-knee"), ("left-knee", "left-ankle"),
    ("right-hip", "right-knee"), ("right-knee", "right-ankle"),
    ("left-ankle", "left-heel"), ("left-ankle", "left-big-toe-tip"),
    ("left-big-toe-tip", "left-small-toe-tip"),
    ("right-ankle", "right-heel"), ("right-ankle", "right-big-toe-tip"),
    ("right-big-toe-tip", "right-small-toe-tip"),
]
SKELETON_LINKS = [
    (NAME_TO_IDX[a], NAME_TO_IDX[b])
    for a, b in _SKELETON_NAME_LINKS
    if a in NAME_TO_IDX and b in NAME_TO_IDX
]

# BGR colors for left / right / center, for nicer skeletons.
_C_LEFT, _C_RIGHT, _C_MID, _C_KPT = (0, 200, 255), (255, 120, 0), (0, 255, 0), (60, 60, 255)


def _link_color(a: int, b: int):
    na, nb = MHR70_NAMES[a], MHR70_NAMES[b]
    if na.startswith("left") or nb.startswith("left"):
        return _C_LEFT
    if na.startswith("right") or nb.startswith("right"):
        return _C_RIGHT
    return _C_MID


# --------------------------------------------------------------------------- #
# Helpers to normalize the model's array shapes
# --------------------------------------------------------------------------- #
def _as_kp2d(arr) -> np.ndarray:
    """Return (N, 3) -> x, y, score."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, a.shape[-1])
    if a.shape[1] == 2:
        a = np.concatenate([a, np.ones((a.shape[0], 1), np.float32)], axis=1)
    return a


def _as_kp3d(arr) -> np.ndarray:
    """Return (N, 3) -> X, Y, Z (drops any trailing score column)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, a.shape[-1])
    return a[:, :3]


# --------------------------------------------------------------------------- #
# Frame extraction / video writing
# --------------------------------------------------------------------------- #
@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    n_frames: int


def extract_frames(video_path: str, frames_dir: str, every: int = 1) -> tuple[List[str], VideoInfo]:
    if not os.path.isfile(video_path):
        sys.exit(f"Video not found: {video_path}")
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    paths: List[str] = []
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            p = os.path.join(frames_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
            saved += 1
        idx += 1
    cap.release()
    info = VideoInfo(fps=fps / every, width=width, height=height, n_frames=saved)
    print(f"[frames] extracted {saved} frames ({width}x{height} @ {fps:.2f}fps) -> {frames_dir}")
    return paths, info


def draw_skeleton(frame_bgr: np.ndarray, persons: list, kpt_thr: float = 0.3) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    for pid, person in enumerate(persons):
        kp = _as_kp2d(person["pred_keypoints_2d"])
        # bbox
        if "bbox" in person and person["bbox"] is not None:
            x1, y1, x2, y2 = [int(v) for v in np.asarray(person["bbox"]).ravel()[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, f"id{pid}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # links
        for (a, b) in SKELETON_LINKS:
            if a >= len(kp) or b >= len(kp):
                continue
            if kp[a, 2] < kpt_thr or kp[b, 2] < kpt_thr:
                continue
            pa = (int(kp[a, 0]), int(kp[a, 1]))
            pb = (int(kp[b, 0]), int(kp[b, 1]))
            cv2.line(img, pa, pb, _link_color(a, b), 2, cv2.LINE_AA)
        # joints
        for i in range(len(kp)):
            if kp[i, 2] < kpt_thr:
                continue
            cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), 3, _C_KPT, -1, cv2.LINE_AA)
    return img


def write_video(frames_bgr_paths: List[str], out_path: str, fps: float) -> None:
    if not frames_bgr_paths:
        sys.exit("No frames to write to video.")
    first = cv2.imread(frames_bgr_paths[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for p in frames_bgr_paths:
        vw.write(cv2.imread(p))
    vw.release()
    print(f"[video]  wrote {len(frames_bgr_paths)} frames -> {out_path}")


# --------------------------------------------------------------------------- #
# CSV export (long / tidy format, ideal for movement analysis)
# --------------------------------------------------------------------------- #
CSV_HEADER = [
    "frame", "time_s", "person_id", "kpt_id", "kpt_name",
    "x_px", "y_px", "score", "X_m", "Y_m", "Z_m",
]


def write_markers_csv(rows: list, out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(CSV_HEADER)
        wr.writerows(rows)
    print(f"[csv]    wrote {len(rows)} marker rows -> {out_path}")


def collect_rows(frame_idx: int, time_s: float, persons: list) -> list:
    out = []
    for pid, person in enumerate(persons):
        kp2 = _as_kp2d(person["pred_keypoints_2d"])
        kp3 = _as_kp3d(person.get("pred_keypoints_3d", np.zeros((len(kp2), 3))))
        n = min(len(kp2), len(kp3), len(MHR70_NAMES))
        for i in range(n):
            out.append([
                frame_idx, round(time_s, 6), pid, i, MHR70_NAMES[i],
                round(float(kp2[i, 0]), 3), round(float(kp2[i, 1]), 3),
                round(float(kp2[i, 2]), 4),
                round(float(kp3[i, 0]), 6), round(float(kp3[i, 1]), 6),
                round(float(kp3[i, 2]), 6),
            ])
    return out


def write_markers_trc(rows: list, out_path: str, fps: float, person_id: int = 0,
                      units: str = "mm", scale: float = 1000.0) -> Optional[str]:
    """Write 3D markers in TRC (OpenSim / Motion Analysis) format.

    TRC is tab-delimited with Frame#, Time, then X/Y/Z per marker. The model's
    3D keypoints are in camera space (X right, Y down, Z forward) in metres;
    here they are scaled to `units` (default mm). All 70 MHR70 markers are
    written in id order so the marker set is constant across frames.
    """
    prows = [r for r in rows if int(r[2]) == person_id]
    if not prows:
        return None
    n_markers = len(MHR70_NAMES)
    # frame -> {kpt_id: (X, Y, Z)} and frame -> time
    grid: dict = {}
    times: dict = {}
    for r in prows:
        fr = int(r[0])
        grid.setdefault(fr, {})[int(r[3])] = (float(r[8]), float(r[9]), float(r[10]))
        times[fr] = float(r[1])
    frames = sorted(grid)
    n_frames = len(frames)

    with open(out_path, "w", newline="") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(out_path)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:g}\t{fps:g}\t{n_frames}\t{n_markers}\t{units}\t"
                f"{fps:g}\t1\t{n_frames}\n")
        # marker-name header: each name followed by two empty cells
        name_cells = "".join(f"{MHR70_NAMES[i]}\t\t\t" for i in range(n_markers))
        f.write(f"Frame#\tTime\t{name_cells}\n")
        # axis sub-header: X1 Y1 Z1 X2 Y2 Z2 ...
        axis_cells = "".join(f"X{i+1}\tY{i+1}\tZ{i+1}\t" for i in range(n_markers))
        f.write(f"\t\t{axis_cells}\n")
        f.write("\n")
        for out_idx, fr in enumerate(frames, start=1):
            cells = [str(out_idx), f"{times[fr]:.6f}"]
            for i in range(n_markers):
                x, y, z = grid[fr].get(i, (0.0, 0.0, 0.0))
                cells += [f"{x*scale:.5f}", f"{y*scale:.5f}", f"{z*scale:.5f}"]
            f.write("\t".join(cells) + "\n")
    print(f"[trc]    wrote {n_frames} frames x {n_markers} markers -> {out_path}")
    return out_path


def write_markers_trc_all(rows: list, out_path: str, fps: float, **kw) -> None:
    """Write one TRC per person (single subject -> out_path; extra subjects get
    a _personN suffix, since TRC is a single-subject format)."""
    pids = sorted({int(r[2]) for r in rows})
    if not pids:
        return
    base, ext = os.path.splitext(out_path)
    for pid in pids:
        path = out_path if pid == pids[0] else f"{base}_person{pid}{ext}"
        write_markers_trc(rows, path, fps, person_id=pid, **kw)


# --------------------------------------------------------------------------- #
# Estimator backends
# --------------------------------------------------------------------------- #
def apply_cpu_patches(device: str) -> None:
    """Best-effort: the upstream estimator hardcodes CUDA. Redirect to cpu/mps.

    Experimental – only meaningful when running the REAL model on a machine
    without CUDA (e.g. Apple Silicon). Must be called before build_real_estimator.
    """
    import torch
    if not torch.cuda.is_available():
        torch.cuda.empty_cache = lambda *a, **k: None  # no-op
    import sam_3d_body.sam_3d_body_estimator as est_mod
    _orig = est_mod.recursive_to

    def _patched(obj, dev):
        return _orig(obj, device if dev == "cuda" else dev)

    est_mod.recursive_to = _patched
    print(f"[patch]  redirected CUDA calls -> {device} (experimental)")


def build_real_estimator(repo: Optional[str], hf_repo_id: str, device: str):
    if repo:
        sys.path.insert(0, os.path.abspath(repo))
    if device != "cuda":
        apply_cpu_patches(device)
    try:
        from notebook.utils import setup_sam_3d_body
    except ImportError as e:
        sys.exit(
            "Could not import the SAM 3D Body package.\n"
            f"  {e}\n"
            "Clone https://github.com/facebookresearch/sam-3d-body, install it per "
            "INSTALL.md, and pass its path via --repo."
        )
    print(f"[model]  loading {hf_repo_id} ...")
    return setup_sam_3d_body(hf_repo_id=hf_repo_id)


class SyntheticEstimator:
    """Drop-in stand-in producing the real output schema, for offline testing.

    Generates one walking-ish person whose 2D keypoints follow a simple
    kinematic puppet so the skeleton video and CSV are visibly correct.
    """

    faces = np.zeros((0, 3), dtype=np.int64)

    def process_one_image(self, img_rgb, **kwargs):
        h, w = img_rgb.shape[:2]
        # mean brightness drives a phase so the puppet "moves" across frames
        t = float(np.mean(img_rgb)) / 255.0
        phase = t * 2 * np.pi
        cx, cy = w * 0.5, h * 0.5
        s = min(w, h) * 0.18  # body scale

        def P(dx, dy):
            return [cx + dx * s, cy + dy * s]

        swing = 0.35 * np.sin(phase)
        arm = 0.5 * np.sin(phase + np.pi)
        named = {
            "nose": P(0, -2.4), "left-eye": P(-0.15, -2.55), "right-eye": P(0.15, -2.55),
            "left-ear": P(-0.35, -2.45), "right-ear": P(0.35, -2.45),
            "neck": P(0, -2.0),
            "left-shoulder": P(-0.6, -1.9), "right-shoulder": P(0.6, -1.9),
            "left-elbow": P(-0.8 + arm, -1.1), "right-elbow": P(0.8 - arm, -1.1),
            "left-wrist": P(-0.9 + 1.4 * arm, -0.3), "right-wrist": P(0.9 - 1.4 * arm, -0.3),
            "left-hip": P(-0.35, 0.0), "right-hip": P(0.35, 0.0),
            "left-knee": P(-0.35 + swing, 1.1), "right-knee": P(0.35 - swing, 1.1),
            "left-ankle": P(-0.35 + 1.5 * swing, 2.2), "right-ankle": P(0.35 - 1.5 * swing, 2.2),
            "left-heel": P(-0.4 + 1.5 * swing, 2.35), "right-heel": P(0.4 - 1.5 * swing, 2.35),
            "left-big-toe-tip": P(-0.25 + 1.5 * swing, 2.45),
            "right-big-toe-tip": P(0.25 - 1.5 * swing, 2.45),
            "left-small-toe-tip": P(-0.45 + 1.5 * swing, 2.45),
            "right-small-toe-tip": P(0.45 - 1.5 * swing, 2.45),
        }
        kp2 = np.zeros((70, 3), np.float32)
        kp3 = np.zeros((70, 3), np.float32)
        for name, (px, py) in named.items():
            i = NAME_TO_IDX[name]
            kp2[i] = [px, py, 1.0]
            # fabricate a plausible 3D point (meters): center & normalize, depth ~2.5m
            kp3[i] = [(px - cx) / s * 0.15, (py - cy) / s * 0.15, 2.5]
        # mark unset (hand/finger) keypoints low-confidence so they aren't drawn
        for i in range(70):
            if kp2[i, 2] == 0:
                kp2[i] = [cx, cy, 0.0]
                kp3[i] = [0, 0, 2.5]
        xs = kp2[kp2[:, 2] > 0, 0]
        ys = kp2[kp2[:, 2] > 0, 1]
        bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
        return [{
            "bbox": bbox,
            "focal_length": float(max(w, h)),
            "pred_keypoints_2d": kp2,
            "pred_keypoints_3d": kp3,
            "pred_joint_coords": kp3.copy(),
            "pred_vertices": np.zeros((0, 3), np.float32),
            "pred_cam_t": np.array([0, 0, 2.5], np.float32),
        }]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(args) -> None:
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    skel_dir = os.path.join(out_dir, "frames_skeleton")
    os.makedirs(skel_dir, exist_ok=True)

    frame_paths, info = extract_frames(args.video, frames_dir, every=args.every)
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
        info.n_frames = len(frame_paths)

    if args.self_test:
        estimator = SyntheticEstimator()
        print("[model]  SELF-TEST mode: using SyntheticEstimator (no real weights)")
    else:
        estimator = build_real_estimator(args.repo, args.hf_repo_id, args.device)

    all_rows: list = []
    skel_paths: List[str] = []
    t0 = time.time()
    for fi, fp in enumerate(frame_paths):
        bgr = cv2.imread(fp)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        persons = estimator.process_one_image(rgb, bbox_thr=args.bbox_thr) or []
        overlay = draw_skeleton(bgr, persons, kpt_thr=args.kpt_thr)
        sp = os.path.join(skel_dir, f"skel_{fi:06d}.jpg")
        cv2.imwrite(sp, overlay)
        skel_paths.append(sp)
        all_rows.extend(collect_rows(fi, fi / info.fps, persons))
        if (fi + 1) % 25 == 0 or fi + 1 == len(frame_paths):
            print(f"[infer]  {fi + 1}/{len(frame_paths)} frames "
                  f"({(fi + 1) / max(1e-9, time.time() - t0):.1f} fps)")

    out_video = os.path.join(out_dir, "skeleton.mp4")
    out_csv = os.path.join(out_dir, "markers.csv")
    out_trc = os.path.join(out_dir, "markers.trc")
    write_video(skel_paths, out_video, info.fps)
    write_markers_csv(all_rows, out_csv)
    write_markers_trc_all(all_rows, out_trc, info.fps)
    print("\n=== DONE ===")
    print(f"  skeleton video : {out_video}")
    print(f"  markers csv    : {out_csv}")
    print(f"  markers trc    : {out_trc}")
    print(f"  frames         : {frames_dir}")


# --------------------------------------------------------------------------- #
# Utility: synthesize a tiny test video so --self-test has something to chew on
# --------------------------------------------------------------------------- #
def make_sample_video(path: str, n: int = 60, w: int = 640, h: int = 480, fps: float = 30.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(n):
        # ramp brightness so SyntheticEstimator's phase advances each frame
        val = int(40 + 170 * (0.5 - 0.5 * np.cos(2 * np.pi * i / n)))
        frame = np.full((h, w, 3), val, np.uint8)
        cv2.putText(frame, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    print(f"[sample] wrote {n}-frame test video -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="input video file")
    ap.add_argument("--output", default="out", help="output directory")
    ap.add_argument("--repo", help="path to cloned facebookresearch/sam-3d-body")
    ap.add_argument("--hf-repo-id", default="facebook/sam-3d-body-dinov3",
                    help="HuggingFace repo id for the checkpoint")
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"],
                    help="compute device (real model hardcodes cuda upstream)")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames (0=all)")
    ap.add_argument("--bbox-thr", type=float, default=0.5, help="detector bbox threshold")
    ap.add_argument("--kpt-thr", type=float, default=0.3, help="keypoint draw threshold")
    ap.add_argument("--self-test", action="store_true",
                    help="use synthetic estimator (verifies plumbing, no weights)")
    ap.add_argument("--make-sample-video", metavar="PATH",
                    help="create a synthetic test video at PATH and exit")
    args = ap.parse_args()

    if args.make_sample_video:
        make_sample_video(args.make_sample_video)
        return
    if not args.video:
        ap.error("--video is required (or use --make-sample-video)")
    run_pipeline(args)


if __name__ == "__main__":
    main()
