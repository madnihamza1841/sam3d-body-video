#!/usr/bin/env python3
"""
make_walk_demo.py
=================
Generates a SYNTHETIC ~15s "person walking" demo and runs it through the
sam3d_video pipeline's drawing/CSV code, producing:

    demo/input.mp4      a rendered humanoid figure doing a walk cycle
    demo/skeleton.mp4   the same video with the MHR70 skeleton overlaid
    demo/markers.csv    3D + 2D marker data (same schema as the real model)

WHY SYNTHETIC: the real facebookresearch/sam-3d-body weights are license-gated
and the upstream code requires a CUDA GPU; neither is available in this
environment (Apple M1). This demo therefore uses a scripted kinematic walk as a
stand-in for the model so the full data flow and output formats can be shown
end-to-end. The input figure and the recovered skeleton come from the SAME
ground-truth walk cycle, so the overlay tracks perfectly.

To run the REAL model on a real video, see README.md.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from sam3d_video import (
    NAME_TO_IDX,
    collect_rows,
    draw_skeleton,
    write_markers_csv,
)

W, H = 540, 720
FPS = 30.0
SECONDS = 15
N = int(FPS * SECONDS)          # 450 frames
STRIDES = 13                    # full leg cycles across the clip
DEPTH = 2.6                     # nominal camera distance (m)


def walk_pose(i: int):
    """Return (kp2 (70,3) px+score, kp3 (70,3) metric) for frame i."""
    phase = 2 * np.pi * STRIDES * (i / N)
    s = min(W, H) * 0.13
    cx = W * 0.5
    bob = 0.07 * s * abs(np.sin(phase))          # vertical bounce, 2x leg freq
    cy = H * 0.40 + bob

    swing = np.sin(phase)                         # right leg phase
    arm = np.sin(phase + np.pi)                   # arms opposite to legs

    def P(dx, dy):
        return [cx + dx * s, cy + dy * s]

    def leg(sign, sw):
        # sign: -1 left / +1 right ; sw: this leg's swing value
        hipx = 0.32 * sign
        lift = max(0.0, sw) * 0.25                # lift foot on forward swing
        knee = P(hipx + 0.45 * sw, 1.15 - 0.6 * lift)
        ankle = P(hipx + 0.95 * sw, 2.25 - lift)
        heel = P(hipx + 0.85 * sw, 2.40 - lift)
        toe = P(hipx + 1.15 * sw, 2.42 - lift)
        return knee, ankle, heel, toe

    def armpose(sign, av):
        shx = 0.55 * sign
        elbow = P(shx + 0.30 * av, -1.05)
        wrist = P(shx + 0.65 * av, -0.30 + 0.10 * abs(av))
        return elbow, wrist

    lk, la, lh, lt = leg(-1, -swing)
    rk, ra, rh, rt = leg(+1, +swing)
    le, lw = armpose(-1, -arm)
    re, rw = armpose(+1, +arm)

    named = {
        "nose": P(0, -2.45), "left-eye": P(-0.13, -2.58), "right-eye": P(0.13, -2.58),
        "left-ear": P(-0.30, -2.46), "right-ear": P(0.30, -2.46),
        "neck": P(0, -2.05),
        "left-shoulder": P(-0.55, -1.95), "right-shoulder": P(0.55, -1.95),
        "left-elbow": le, "right-elbow": re, "left-wrist": lw, "right-wrist": rw,
        "left-hip": P(-0.32, 0.0), "right-hip": P(0.32, 0.0),
        "left-knee": lk, "right-knee": rk, "left-ankle": la, "right-ankle": ra,
        "left-heel": lh, "right-heel": rh,
        "left-big-toe-tip": lt, "right-big-toe-tip": rt,
        "left-small-toe-tip": [lt[0] - 0.12 * s, lt[1]],
        "right-small-toe-tip": [rt[0] + 0.12 * s, rt[1]],
    }
    kp2 = np.zeros((70, 3), np.float32)
    kp3 = np.zeros((70, 3), np.float32)
    for name, (px, py) in named.items():
        j = NAME_TO_IDX[name]
        kp2[j] = [px, py, 1.0]
        kp3[j] = [(px - cx) / s * 0.16, (py - cy) / s * 0.16, DEPTH + 0.05 * np.sin(phase)]
    # unobserved keypoints (fingers, olecranon, ...) -> score 0 (not drawn)
    for j in range(70):
        if kp2[j, 2] == 0:
            kp2[j] = [cx, cy, 0.0]
            kp3[j] = [0, 0, DEPTH]
    return kp2, kp3


def render_figure(kp2: np.ndarray) -> np.ndarray:
    """Draw a filled humanoid silhouette from the 2D keypoints."""
    img = np.full((H, W, 3), (235, 235, 230), np.uint8)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (210, 210, 205), 6)  # frame border

    def pt(name):
        return np.array(kp2[NAME_TO_IDX[name], :2], np.float32)

    body = (70, 90, 110)      # silhouette color (BGR)
    # torso
    torso = np.array([pt("left-shoulder"), pt("right-shoulder"),
                      pt("right-hip"), pt("left-hip")], np.int32)
    cv2.fillConvexPoly(img, torso, body, cv2.LINE_AA)
    # limbs as thick capsules
    limb_pairs = [
        ("left-shoulder", "left-elbow"), ("left-elbow", "left-wrist"),
        ("right-shoulder", "right-elbow"), ("right-elbow", "right-wrist"),
        ("left-hip", "left-knee"), ("left-knee", "left-ankle"),
        ("right-hip", "right-knee"), ("right-knee", "right-ankle"),
        ("left-ankle", "left-big-toe-tip"), ("right-ankle", "right-big-toe-tip"),
        ("neck", "left-shoulder"), ("neck", "right-shoulder"),
    ]
    for a, b in limb_pairs:
        pa, pb = pt(a).astype(int), pt(b).astype(int)
        cv2.line(img, tuple(pa), tuple(pb), body, 18, cv2.LINE_AA)
        cv2.circle(img, tuple(pa), 9, body, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(pb), 9, body, -1, cv2.LINE_AA)
    # head
    head = ((pt("left-eye") + pt("right-eye")) / 2).astype(int)
    cv2.circle(img, tuple(head), 26, body, -1, cv2.LINE_AA)
    return img


def main():
    out = "demo"
    os.makedirs(out, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vin = cv2.VideoWriter(os.path.join(out, "input.mp4"), fourcc, FPS, (W, H))
    vsk = cv2.VideoWriter(os.path.join(out, "skeleton.mp4"), fourcc, FPS, (W, H))

    rows = []
    for i in range(N):
        kp2, kp3 = walk_pose(i)
        frame = render_figure(kp2)
        persons = [{
            "bbox": np.array([kp2[kp2[:, 2] > 0, 0].min(), kp2[kp2[:, 2] > 0, 1].min(),
                              kp2[kp2[:, 2] > 0, 0].max(), kp2[kp2[:, 2] > 0, 1].max()],
                             np.float32),
            "pred_keypoints_2d": kp2,
            "pred_keypoints_3d": kp3,
        }]
        overlay = draw_skeleton(frame, persons, kpt_thr=0.3)
        vin.write(frame)
        vsk.write(overlay)
        rows.extend(collect_rows(i, i / FPS, persons))

    vin.release()
    vsk.release()
    write_markers_csv(rows, os.path.join(out, "markers.csv"))
    print(f"[demo]   {N} frames @ {FPS}fps ({SECONDS}s) -> {out}/input.mp4, "
          f"{out}/skeleton.mp4, {out}/markers.csv")


if __name__ == "__main__":
    main()
