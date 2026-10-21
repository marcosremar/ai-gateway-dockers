"""Whiteboard animation engine — wraps the OpenCV object-by-object hand-drawn
animation engine from storyboard-ai (generate-whiteboard-animated-videos).

Source: https://github.com/Innovate-Inspire/storyboard-ai (GPL-3.0)
We bundle the engine directly so the Docker image is self-contained. Algorithm
itself is intact; we extend it with optional style controls:

  - `background_color` (BGR tuple) — replaces the hardcoded white canvas. Use
    e.g. (250, 253, 254) for off-white paper feel.
  - `accent_palette` (list of BGR tuples) — colors that, after the line-art
    pass completes, are tinted onto the LARGEST connected components in the
    drawing in order. Approximates the "fill cartoon characters with flat
    saturated color" look of channels like RenovadaMente (whose lines are
    drawn first then coloured).
  - `stroke_speed` — multiplier on the legacy `object_skip_rate` /
    `bg_object_skip_rate` (1.0 = default; >1 faster, <1 slower).
  - `PRESETS` registry — named bundles of the above (e.g. "renovadamente").

The base algorithm (`_draw_masked_object`, hand sprite compositor) is GPL-3.0
upstream and unmodified except for parameter wiring.

License: GPL-3.0 (inherited from storyboard-ai LICENSE).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── Style presets ────────────────────────────────────────────────────────────
# Each preset is a dict consumed by `render_whiteboard(preset=...)`. Values
# here override the function's keyword defaults but explicit kwargs win.
#
# Colors are stored as BGR tuples (OpenCV native). RGB equivalents are noted
# in comments for sanity-checking against the visual.md analysis.

PRESETS: dict = {
    # RenovadaMente-Oficial (pt-BR psychology channel) — see
    # project-philosofi/models/renovadamente/analysis/visual.md
    "renovadamente": {
        # Pure white, matches frame 00_open of every analysed video.
        "background_color": (255, 255, 255),  # RGB #ffffff
        # Saturated accent palette in BGR order. Order is applied to
        # LARGEST→smallest connected components after the line-art pass:
        # roughly mimics the "first character gets red shirt, second gets
        # blue, etc." cartoon coloring.
        "accent_palette": [
            (62, 62, 229),    # red       RGB #e53e3e — title accents / fists
            (217, 144, 74),   # blue      RGB #4a90d9 — male shirts / jeans
            (21, 204, 250),   # yellow    RGB #facc15 — sun / lightbulb / brand
            (199, 163, 245),  # pink      RGB #f5a3c7 — female tops
            (96, 174, 39),    # green     RGB #27ae60 — checkmarks / dollars
        ],
        # Whiteboard cartoon style draws faster than encyclopedic builds —
        # makes the hand feel snappy. 1.5× the upstream defaults.
        "stroke_speed": 1.5,
        # Ends with a longer-held final frame so the colored result lingers
        # before the next composition kicks in.
        "end_duration_s": 4,
    },
    # Generic fallback — matches storyboard-ai upstream behaviour exactly.
    "default": {
        "background_color": (255, 255, 255),
        "accent_palette": [],
        "stroke_speed": 1.0,
        "end_duration_s": 3,
    },
}


def _hex_to_bgr(spec: str) -> Tuple[int, int, int]:
    """Accepts '#RRGGBB' or 'RRGGBB' and returns (B, G, R) for OpenCV."""
    s = spec.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"invalid hex color: {spec!r}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def _normalize_color(value) -> Tuple[int, int, int]:
    """Accept '#RRGGBB', 'RRGGBB', [R,G,B], (B,G,R), or None → tuple BGR."""
    if value is None:
        return (255, 255, 255)
    if isinstance(value, str):
        return _hex_to_bgr(value)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        # We can't tell RGB-vs-BGR without convention; assume the API caller
        # gives RGB (more common) and convert. The PRESETS table stores BGR
        # directly so it bypasses this path.
        r, g, b = (int(v) for v in value)
        return (b, g, r)
    raise ValueError(f"cannot interpret colour: {value!r}")


def _normalize_palette(value) -> List[Tuple[int, int, int]]:
    if not value:
        return []
    return [_normalize_color(v) for v in value]


# ── Original storyboard-ai algorithm (unmodified except for `print`s removed) ──

def _euc_dist(arr1, point):
    square_sub = (arr1 - point) ** 2
    return np.sqrt(np.sum(square_sub, axis=1))


def _preprocess_image(img_path, variables):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"cv2 could not read image: {img_path}")
    img_ht, img_wd = img.shape[0], img.shape[1]
    img = cv2.resize(img, (variables.resize_wd, variables.resize_ht))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    _ = clahe.apply(img_gray)

    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )

    variables.img_ht = img_ht
    variables.img_wd = img_wd
    variables.img_gray = img_gray
    variables.img_thresh = img_thresh
    variables.img = img
    return variables


def _get_extreme_coordinates(mask):
    indices = np.where(mask == 255)
    x = indices[1]
    y = indices[0]
    topleft = (int(np.min(x)), int(np.min(y)))
    bottomright = (int(np.max(x)), int(np.max(y)))
    return topleft, bottomright


def _preprocess_hand_image(hand_path, hand_mask_path, variables):
    hand = cv2.imread(hand_path)
    hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)
    if hand is None or hand_mask is None:
        raise ValueError(f"could not read hand sprite ({hand_path}) or mask ({hand_mask_path})")

    top_left, bottom_right = _get_extreme_coordinates(hand_mask)
    hand = hand[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    hand_mask = hand_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    hand_mask_inv = 255 - hand_mask

    hand_mask = hand_mask / 255
    hand_mask_inv = hand_mask_inv / 255

    hand_bg_ind = np.where(hand_mask == 0)
    hand[hand_bg_ind] = [0, 0, 0]

    variables.hand_ht = hand.shape[0]
    variables.hand_wd = hand.shape[1]
    variables.hand = hand
    variables.hand_mask = hand_mask
    variables.hand_mask_inv = hand_mask_inv
    return variables


def _draw_hand_on_img(drawing, hand, x, y, hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd):
    remaining_ht = img_ht - y
    remaining_wd = img_wd - x
    crop_hand_ht = hand_ht if remaining_ht > hand_ht else remaining_ht
    crop_hand_wd = hand_wd if remaining_wd > hand_wd else remaining_wd

    hand_cropped = hand[:crop_hand_ht, :crop_hand_wd]
    hand_mask_inv_cropped = hand_mask_inv[:crop_hand_ht, :crop_hand_wd]

    for c in range(3):
        drawing[y:y + crop_hand_ht, x:x + crop_hand_wd][:, :, c] = (
            drawing[y:y + crop_hand_ht, x:x + crop_hand_wd][:, :, c] * hand_mask_inv_cropped
        )

    drawing[y:y + crop_hand_ht, x:x + crop_hand_wd] = (
        drawing[y:y + crop_hand_ht, x:x + crop_hand_wd] + hand_cropped
    )
    return drawing


def _draw_masked_object(variables, object_mask=None, skip_rate=5, black_pixel_threshold=10):
    img_thresh_copy = variables.img_thresh.copy()
    if object_mask is not None:
        object_mask_black_ind = np.where(object_mask == 0)
        object_ind = np.where(object_mask == 255)
        img_thresh_copy[object_mask_black_ind] = 255

    selected_ind = 0
    n_cuts_vertical = int(math.ceil(variables.resize_ht / variables.split_len))
    n_cuts_horizontal = int(math.ceil(variables.resize_wd / variables.split_len))

    grid_of_cuts = np.array(np.split(img_thresh_copy, n_cuts_horizontal, axis=-1))
    grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))

    cut_having_black = (grid_of_cuts < black_pixel_threshold) * 1
    cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
    cut_black_indices = np.array(np.where(cut_having_black > 0)).T

    counter = 0
    while len(cut_black_indices) > 1:
        selected_ind_val = cut_black_indices[selected_ind].copy()
        range_v_start = selected_ind_val[0] * variables.split_len
        range_v_end = range_v_start + variables.split_len
        range_h_start = selected_ind_val[1] * variables.split_len
        range_h_end = range_h_start + variables.split_len

        temp_drawing = np.zeros((variables.split_len, variables.split_len, 3))
        temp_drawing[:, :, 0] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
        temp_drawing[:, :, 1] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
        temp_drawing[:, :, 2] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]

        variables.drawn_frame[range_v_start:range_v_end, range_h_start:range_h_end] = temp_drawing

        hand_coord_x = range_h_start + int(variables.split_len / 2)
        hand_coord_y = range_v_start + int(variables.split_len / 2)
        drawn_frame_with_hand = _draw_hand_on_img(
            variables.drawn_frame.copy(),
            variables.hand.copy(),
            hand_coord_x,
            hand_coord_y,
            variables.hand_mask_inv.copy(),
            variables.hand_ht,
            variables.hand_wd,
            variables.resize_ht,
            variables.resize_wd,
        )

        cut_black_indices[selected_ind] = cut_black_indices[-1]
        cut_black_indices = cut_black_indices[:-1]
        del selected_ind

        euc_arr = _euc_dist(cut_black_indices, selected_ind_val)
        selected_ind = int(np.argmin(euc_arr))

        counter += 1
        if counter % skip_rate == 0:
            variables.video_object.write(drawn_frame_with_hand)
            variables.frames_written += 1

    if object_mask is not None:
        variables.drawn_frame[:, :, :][object_ind] = variables.img[object_ind]
    else:
        variables.drawn_frame[:, :, :] = variables.img


class _AllVariables:
    def __init__(self, frame_rate, resize_wd, resize_ht, split_len,
                 object_skip_rate, bg_object_skip_rate, end_gray_img_duration_in_sec,
                 background_color=(255, 255, 255), accent_palette=None):
        self.frame_rate = frame_rate
        self.resize_wd = resize_wd
        self.resize_ht = resize_ht
        self.split_len = split_len
        self.object_skip_rate = object_skip_rate
        self.bg_object_skip_rate = bg_object_skip_rate
        self.end_gray_img_duration_in_sec = end_gray_img_duration_in_sec
        self.background_color = background_color
        self.accent_palette = accent_palette or []
        self.frames_written = 0


def _draw_whiteboard_animations(img_path, mask_path, hand_path, hand_mask_path,
                                save_video_path, variables):
    object_mask_exists = mask_path is not None

    variables = _preprocess_image(img_path=img_path, variables=variables)
    variables = _preprocess_hand_image(
        hand_path=hand_path, hand_mask_path=hand_mask_path, variables=variables
    )

    variables.video_object = cv2.VideoWriter(
        save_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        variables.frame_rate,
        (variables.resize_wd, variables.resize_wd),
    )

    bg_bgr = getattr(variables, "background_color", (255, 255, 255))
    variables.drawn_frame = np.zeros(variables.img.shape, np.uint8) + np.array(
        [bg_bgr[0], bg_bgr[1], bg_bgr[2]], np.uint8
    )

    if object_mask_exists:
        with open(mask_path) as file:
            object_masks = json.load(file)

        background_mask = np.zeros((variables.resize_ht, variables.resize_wd), dtype=np.uint8) + 255

        for obj in object_masks["shapes"]:
            object_mask = np.zeros((variables.img_ht, variables.img_wd), dtype=np.uint8)
            object_points = np.array(obj["points"], dtype=np.int32)
            object_points = np.expand_dims(object_points, axis=0)
            cv2.fillPoly(object_mask, object_points, 255)
            object_mask = cv2.resize(object_mask, (variables.resize_wd, variables.resize_ht))
            object_ind = np.where(object_mask == 255)
            background_mask[object_ind] = 0
            _draw_masked_object(
                variables=variables,
                object_mask=object_mask,
                skip_rate=variables.object_skip_rate,
            )

        # background pass with larger split + higher skip rate
        variables.split_len = 20
        _draw_masked_object(
            variables=variables,
            object_mask=background_mask,
            skip_rate=variables.bg_object_skip_rate,
        )
    else:
        _draw_masked_object(variables=variables, skip_rate=variables.object_skip_rate)

    # Optional accent-palette colorization pass. Approximates the
    # "color-by-largest-region" pattern observed in RenovadaMente:
    # after the line art is fully drawn, we tint the largest connected
    # white regions inside the threshold image with the palette colors,
    # in size order. Hand sprite is NOT shown for this pass — it's a
    # post-stroke fill.
    if variables.accent_palette:
        try:
            colored = variables.img.copy()
            # Use threshold image to find enclosed white regions (i.e. the
            # interiors of cartoon characters / objects).
            inv = cv2.bitwise_not(variables.img_thresh)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                inv, connectivity=8
            )
            # Skip label 0 (background outside drawing) and pick top-N by area
            # but ignore very-large regions (likely whole-canvas) and tiny noise.
            total_pixels = variables.resize_ht * variables.resize_wd
            candidates = []
            for lab in range(1, num_labels):
                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area < total_pixels * 0.005:  # <0.5% — noise
                    continue
                if area > total_pixels * 0.40:   # >40% — likely whole frame bg
                    continue
                candidates.append((area, lab))
            candidates.sort(reverse=True)
            for i, (_area, lab) in enumerate(candidates[: len(variables.accent_palette)]):
                color = np.array(variables.accent_palette[i], dtype=np.uint8)
                mask_lab = (labels == lab)
                # Soft fill (0.7 alpha) so outline survives.
                colored[mask_lab] = (
                    colored[mask_lab].astype(np.float32) * 0.3 + color.astype(np.float32) * 0.7
                ).astype(np.uint8)

            # Brief crossfade from line-art (variables.img) → colored result
            xfade_frames = int(variables.frame_rate * 0.6)  # 0.6s
            for k in range(xfade_frames):
                a = (k + 1) / xfade_frames
                blend = (
                    variables.img.astype(np.float32) * (1 - a)
                    + colored.astype(np.float32) * a
                ).astype(np.uint8)
                variables.video_object.write(blend)
                variables.frames_written += 1
            variables.img = colored  # so the hold-end frames show colored
        except Exception:
            # Colourisation is best-effort; if anything fails, fall back to the
            # original line-art result.
            pass

    end_frames = variables.frame_rate * variables.end_gray_img_duration_in_sec
    for _ in range(end_frames):
        variables.video_object.write(variables.img)
        variables.frames_written += 1

    variables.video_object.release()


def render_whiteboard(
    image_path: str,
    mask_path: Optional[str],
    hand_path: str,
    hand_mask_path: str,
    output_path: str,
    frame_rate: int = 25,
    resize: int = 1080,
    split_len: int = 10,
    object_skip_rate: int = 8,
    bg_object_skip_rate: int = 14,
    end_duration_s: int = 3,
    *,
    preset: Optional[str] = None,
    background_color=None,
    accent_palette=None,
    stroke_speed: Optional[float] = None,
) -> dict:
    """Render a whiteboard hand-drawn animation MP4.

    Args:
        image_path: PNG line drawing (will be CLAHE+adaptive-threshold-binarised).
        mask_path:  optional JSON segmentation map (LabelMe `shapes` schema). If
                    None, the whole image is drawn as one pass.
        hand_path:  PNG sprite of a hand holding a marker.
        hand_mask_path: PNG mask isolating the hand from its background.
        output_path: where to write the MP4. Parent dir must exist.
        frame_rate, resize, split_len, object_skip_rate, bg_object_skip_rate,
        end_duration_s: see storyboard-ai/draw-whiteboard-animations.py docstrings.
        preset: optional named style ("renovadamente", "default"). Loads
                background_color, accent_palette, stroke_speed, end_duration_s
                from PRESETS as defaults; explicit kwargs win.
        background_color: '#RRGGBB' hex, [R,G,B] list, or BGR tuple. Replaces
                the default white canvas (`#ffffff`).
        accent_palette: list of color specs (hex/RGB) applied to the largest
                connected components after line-art completes. Mimics the
                "flat-fill cartoon coloring" of channels like RenovadaMente.
        stroke_speed: multiplier on object/bg skip rates (1.0=default, >1=faster).

    Returns:
        {"output_path": ..., "duration_s": ..., "frames_total": ...,
         "render_time_s": ..., "preset": ...}
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Resolve preset → defaults, allow explicit kwargs to override.
    if preset and preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset!r} (have {list(PRESETS)})")
    preset_defaults = PRESETS.get(preset or "default", PRESETS["default"])

    eff_background = background_color if background_color is not None \
        else preset_defaults["background_color"]
    eff_palette = accent_palette if accent_palette is not None \
        else preset_defaults["accent_palette"]
    eff_speed = float(stroke_speed) if stroke_speed is not None \
        else float(preset_defaults["stroke_speed"])
    eff_end_s = end_duration_s if end_duration_s != 3 \
        else int(preset_defaults["end_duration_s"])

    # Apply stroke_speed by lowering the skip_rate (smaller = more frames written
    # = slower). >1 = faster (skip more frames per stroke).
    eff_object_skip = max(2, int(round(object_skip_rate * eff_speed)))
    eff_bg_skip = max(4, int(round(bg_object_skip_rate * eff_speed)))

    bg_bgr = _normalize_color(eff_background) \
        if not (isinstance(eff_background, tuple) and len(eff_background) == 3 and all(isinstance(x, int) for x in eff_background)) \
        else eff_background
    palette_bgr = _normalize_palette(eff_palette) \
        if not (eff_palette and all(isinstance(x, tuple) and len(x) == 3 for x in eff_palette)) \
        else list(eff_palette)

    variables = _AllVariables(
        frame_rate=int(frame_rate),
        resize_wd=int(resize),
        resize_ht=int(resize),
        split_len=int(split_len),
        object_skip_rate=eff_object_skip,
        bg_object_skip_rate=eff_bg_skip,
        end_gray_img_duration_in_sec=int(eff_end_s),
        background_color=bg_bgr,
        accent_palette=palette_bgr,
    )

    t0 = time.time()
    _draw_whiteboard_animations(
        img_path=image_path,
        mask_path=mask_path,
        hand_path=hand_path,
        hand_mask_path=hand_mask_path,
        save_video_path=output_path,
        variables=variables,
    )
    render_time = time.time() - t0

    frames_total = variables.frames_written
    duration_s = round(frames_total / variables.frame_rate, 3) if variables.frame_rate else 0.0

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"engine produced no output at {output_path}")

    return {
        "output_path": output_path,
        "duration_s": duration_s,
        "frames_total": frames_total,
        "render_time_s": round(render_time, 3),
        "preset": preset or "default",
    }
