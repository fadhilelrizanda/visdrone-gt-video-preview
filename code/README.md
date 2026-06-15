# VisDrone Ground-Truth Video Preview

This package downloads an official VisDrone MOT split, overlays the ground-truth boxes on one sequence, and exports a preview video capped at 60 seconds by default.

## What it does

- downloads `VisDrone2019-MOT-train` or `VisDrone2019-MOT-val`
- extracts the archive locally
- selects one sequence
- draws ground-truth boxes and class labels on each frame
- exports an `.mp4` preview video

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

## Usage

```bash
python3 code/make_visdrone_gt_video.py \
  --split mot_val \
  --data-root ./data \
  --output ./outputs/visdrone_mot_val_preview.mp4 \
  --max-seconds 60
```

To render a specific sequence:

```bash
python3 code/make_visdrone_gt_video.py \
  --split mot_val \
  --sequence uav0000086_00000_v \
  --data-root ./data \
  --output ./outputs/uav0000086_00000_v.mp4
```

## Notes

- The default dataset URLs point to the official VisDrone Google Drive links published in the VisDrone dataset repository.
- The script targets the MOT layout with `sequences/` and `annotations/` directories.
- Output FPS defaults to `20`. You can override it with `--fps`.
