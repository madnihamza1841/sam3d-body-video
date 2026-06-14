# SAM 3D Body → skeleton video + marker CSV

`sam3d_video.py` takes a video, splits it into frames, runs each frame through
Meta's [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) model, and
produces:

- `out/skeleton.mp4` — the input with the 70-keypoint MHR skeleton drawn on top
- `out/markers.csv` — tidy/long-format 3D + 2D marker data (one row per
  keypoint per person per frame)
- `out/frames/` and `out/frames_skeleton/` — the intermediate frames

## CSV columns

```
frame, time_s, person_id, kpt_id, kpt_name, x_px, y_px, score, X_m, Y_m, Z_m
```

`x_px,y_px` are image pixels (for overlay); `X_m,Y_m,Z_m` are the model's 3D
keypoints in camera space (metric). Keypoint names follow MHR70.

## Quick verification (no GPU / no weights)

This exercises the entire pipeline with a synthetic estimator that emits the
**exact same output schema** as the real model:

```bash
python sam3d_video.py --make-sample-video sample.mp4
python sam3d_video.py --video sample.mp4 --output out --self-test
```

## Included demo (synthetic, 15s walking figure)

The `demo/` folder contains a worked example produced by `make_walk_demo.py`:

| file | what it is |
|------|------------|
| `demo/input.mp4` | a rendered humanoid doing a 15s walk cycle (450 frames @ 30fps) |
| `demo/skeleton.mp4` | the same clip with the MHR70 skeleton overlaid |
| `demo/markers.csv` | 3D + 2D marker data, 31,500 rows (450 frames × 70 keypoints) |

Regenerate with:

```bash
python make_walk_demo.py
```

> **This demo is synthetic.** The real `facebookresearch/sam-3d-body` weights are
> license-gated and the upstream code requires a CUDA GPU — neither is available
> on the Apple-Silicon machine this was built on, and CPU inference over 450
> frames would take hours. The demo uses a scripted kinematic walk as a stand-in
> for the model so the data flow and output formats are demonstrated end-to-end.
> The input figure and recovered skeleton come from the same ground-truth walk
> cycle, so the overlay tracks exactly. For genuine model output, run the real
> model (below) on a GPU machine.

## Running the REAL model

The model is heavy, GPU-oriented, and its weights are gated. Setup:

```bash
# 1. clone + install the model (Python 3.11, CUDA GPU strongly recommended)
git clone https://github.com/facebookresearch/sam-3d-body
cd sam-3d-body
conda create -n sam_3d_body python=3.11 -y && conda activate sam_3d_body
# install torch per https://pytorch.org/get-started/locally/, then:
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops \
    timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog \
    pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb \
    appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree \
    fvcore black pycocotools tensorboard huggingface_hub
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps

# 2. accept the license on HuggingFace, then download the gated checkpoint
huggingface-cli login
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3

# 3. run this pipeline, pointing --repo at the clone
python /path/to/sam3d_video.py \
    --video in.mp4 --output out \
    --repo /path/to/sam-3d-body \
    --hf-repo-id facebook/sam-3d-body-dinov3 \
    --device cuda
```

### Important caveats

- **Gated weights.** `facebook/sam-3d-body-dinov3` requires accepting Meta's
  license on HuggingFace (and may need regional approval). You must do this with
  your own HF account — it can't be automated.
- **CUDA is assumed.** Upstream `process_one_image` hardcodes
  `recursive_to(batch, "cuda")` and `torch.cuda.empty_cache()`. On a non-CUDA
  machine (e.g. Apple Silicon) pass `--device mps` or `--device cpu`; this script
  applies best-effort monkeypatches (`apply_cpu_patches`) to redirect those
  calls. This path is **experimental** — detectron2/pyrender on macOS without
  CUDA is fragile and slow.
- **Detector required for multi-person / full images.** `setup_sam_3d_body`
  wires up a human detector; without it the whole image is treated as one bbox.

## Useful flags

| flag | meaning |
|------|---------|
| `--every N` | keep every Nth frame (downsample) |
| `--max-frames N` | cap number of frames processed |
| `--bbox-thr` | detector confidence threshold |
| `--kpt-thr` | min keypoint score to draw |
| `--self-test` | synthetic estimator, no weights needed |
