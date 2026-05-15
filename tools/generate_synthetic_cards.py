"""Generate a synthetic playing-card image dataset for classifier training.

Produces the exact 52-folder ImageFolder layout that train_classifier.py
expects:

    <output>/
      AH/  AH_0000.png  AH_0001.png ...
      2H/  ...
      ...
      KS/  ...

Each card is drawn programmatically with PIL: rank in two corners, suit
symbol(s) in the center, randomized rotation / brightness / blur / noise /
crop so the classifier sees varied inputs. The result won't perfectly
match real casino stream cards — but it gives the pipeline a functional
trained classifier in minutes, which is enough to exercise the full FSM
cycle end-to-end and to validate the integration before you collect real
labeled data.

Usage:
    python tools/generate_synthetic_cards.py --output data/cards --per-class 80
    python train_classifier.py --data data/cards --epochs 8 --batch-size 64
    python run.py --source samples/clip.webm \\
        --roi-config configs/pragmatic_speed_baccarat.yaml \\
        --classifier-weights weights/mobilenetv3_cards_fp16.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS = ("H", "D", "C", "S")
SUIT_GLYPHS = {"H": "♥", "D": "♦", "C": "♣", "S": "♠"}
RED_SUITS = {"H", "D"}

CARD_W, CARD_H = 110, 154


def _try_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_centered(draw: ImageDraw.ImageDraw, xy, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = xy
    draw.text((x - tw / 2, y - th / 2), text, font=font, fill=fill)


def _draw_canonical_card(rank: str, suit: str) -> Image.Image:
    img = Image.new("RGB", (CARD_W, CARD_H), (252, 250, 246))
    draw = ImageDraw.Draw(img)
    color = (200, 30, 40) if suit in RED_SUITS else (20, 20, 20)
    glyph = SUIT_GLYPHS[suit]
    rank_font = _try_font(26)
    small_glyph_font = _try_font(20)
    big_glyph_font = _try_font(56)

    draw.rounded_rectangle(((1, 1), (CARD_W - 2, CARD_H - 2)),
                           radius=8, outline=(150, 150, 150), width=1)

    draw.text((6, 4), rank, font=rank_font, fill=color)
    draw.text((6, 30), glyph, font=small_glyph_font, fill=color)
    draw.text((CARD_W - 26, CARD_H - 50), rank, font=rank_font, fill=color)
    draw.text((CARD_W - 22, CARD_H - 24), glyph, font=small_glyph_font, fill=color)

    _draw_centered(draw, (CARD_W // 2, CARD_H // 2), glyph, big_glyph_font, color)
    return img


def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-12, 12)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(252, 250, 246))

    scale = rng.uniform(0.85, 1.12)
    new_w = max(32, int(img.width * scale))
    new_h = max(32, int(img.height * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    brightness = rng.uniform(0.75, 1.15)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    contrast = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    color = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Color(img).enhance(color)

    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.1)))

    arr = np.array(img, dtype=np.int16)
    noise = np.random.normal(0, rng.uniform(2, 8), arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    pad = 10
    canvas = Image.new(
        "RGB",
        (img.width + pad * 2, img.height + pad * 2),
        (rng.randint(20, 80), rng.randint(20, 60), rng.randint(20, 60)),
    )
    canvas.paste(img, (pad, pad))
    return canvas


def _make_one(rank: str, suit: str, rng: random.Random, size: Tuple[int, int]) -> Image.Image:
    base = _draw_canonical_card(rank, suit)
    aug = _augment(base, rng)
    aug = aug.resize(size, Image.BICUBIC)
    return aug


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--output", type=str, required=True,
                        help="Destination directory (52 subfolders will be created here)")
    parser.add_argument("--per-class", type=int, default=80,
                        help="Number of images per card class (default 80 → 4160 total)")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    total_classes = len(RANKS) * len(SUITS)
    total_images = total_classes * args.per_class
    print(f"writing {total_images} images across {total_classes} classes -> {output}")

    written = 0
    for rank in RANKS:
        for suit in SUITS:
            label = f"{rank}{suit}"
            cls_dir = output / label
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(args.per_class):
                img = _make_one(rank, suit, rng, size=(args.width, args.height))
                img.save(cls_dir / f"{label}_{i:04d}.png")
                written += 1
        print(f"  {rank}* done — {written}/{total_images}")

    print(f"done. dataset ready at {output}")
    print(f"next: python train_classifier.py --data {output} --epochs 8 --batch-size 64")
    return 0


if __name__ == "__main__":
    sys.exit(main())
