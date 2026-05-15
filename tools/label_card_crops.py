"""Interactive labeling helper for card crops captured via --capture-crops.

Reads each PNG in <input>/unlabeled/, shows it in an OpenCV window, and
moves it to <input>/<RankSuit>/ based on two keystrokes (one for rank, one
for suit). The resulting layout is exactly what train_classifier.py expects.

Keybindings:
    rank   a 2 3 4 5 6 7 8 9 0 j q k       (0 = 10)
    suit   h d c s                         (Hearts/Diamonds/Clubs/Spades)
    other  n = skip (leave file for later)
           u = undo last label (moves it back to unlabeled/)
           q = quit
           ENTER = accept the rank you've typed so far (for "10")

Headless mode: pass --batch and a JSON mapping file to label without a GUI
(useful over SSH). See --help for details.

Workflow:
    # 1) collect crops while running the live pipeline
    python run.py --source samples/test.webm \\
        --roi-config configs/pragmatic_speed_baccarat.yaml \\
        --capture-crops out/training/

    # 2) label them (one-key rank + one-key suit)
    python tools/label_card_crops.py --input out/training/

    # 3) retrain on the labeled crops
    python train_classifier.py --data out/training/ --epochs 20
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


RANK_KEYS: Dict[int, str] = {
    ord("a"): "A", ord("A"): "A",
    ord("2"): "2", ord("3"): "3", ord("4"): "4", ord("5"): "5",
    ord("6"): "6", ord("7"): "7", ord("8"): "8", ord("9"): "9",
    ord("0"): "10",
    ord("j"): "J", ord("J"): "J",
    ord("q"): "Q", ord("Q"): "Q",
    ord("k"): "K", ord("K"): "K",
}
SUIT_KEYS: Dict[int, str] = {
    ord("h"): "H", ord("H"): "H",
    ord("d"): "D", ord("D"): "D",
    ord("c"): "C", ord("C"): "C",
    ord("s"): "S", ord("S"): "S",
}
QUIT_KEYS = {ord("q") & 0xFF, 27}  # 27 = ESC; the rank 'q' uses a separate sentinel
ALL_LABELS = [f"{r}{s}" for r in
              ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
              for s in ("H", "D", "C", "S")]


def _read_label_keystroke(window_name: str, image, prompt: str) -> Optional[str]:
    """Open a labelling window over `image` and read rank+suit keys.
    Returns a [Rank][Suit] string, or None to skip, or 'QUIT' to exit."""
    import cv2

    def overlay(text: str):
        canvas = image.copy() if image is not None else None
        if canvas is None:
            canvas = (255 * (cv2 / 255)).astype("uint8")  # noqa: dummy
        h, w = canvas.shape[:2]
        pad = 30
        out = cv2.copyMakeBorder(canvas, pad, pad, pad, pad,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(out, prompt, (8, pad - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, text, (8, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (60, 220, 60), 2, cv2.LINE_AA)
        return out

    rank: Optional[str] = None
    while True:
        cv2.imshow(window_name, overlay(f"rank={rank or '_'}  suit=_"))
        k = cv2.waitKey(0) & 0xFF
        if k == ord("Q"):
            return "QUIT"
        if k == 27:
            return "QUIT"
        if k == ord("n") or k == ord("N"):
            return None
        if rank is None:
            if k in RANK_KEYS:
                rank = RANK_KEYS[k]
                continue
        else:
            if k in SUIT_KEYS:
                return f"{rank}{SUIT_KEYS[k]}"
            if k == 8:  # backspace
                rank = None
                continue


def _label_one(crop_path: Path, output_root: Path) -> Optional[str]:
    import cv2
    img = cv2.imread(str(crop_path))
    if img is None:
        log.warning("could not read %s, skipping", crop_path)
        return None
    h, w = img.shape[:2]
    if max(h, w) < 200:
        scale = 200.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    prompt = f"{crop_path.name}  —  press rank then suit  (n=skip, ESC=quit)"
    label = _read_label_keystroke("label", img, prompt)
    if label == "QUIT":
        return "QUIT"
    if label is None:
        return None
    if label not in ALL_LABELS:
        log.warning("invalid label %s; skipping", label)
        return None
    dest_dir = output_root / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / crop_path.name
    shutil.move(str(crop_path), str(dest))
    log.info("  %s -> %s/", crop_path.name, label)
    return label


def interactive_loop(input_root: Path) -> int:
    import cv2
    unlabeled = input_root / "unlabeled"
    if not unlabeled.exists():
        log.error("no 'unlabeled' directory under %s — pass the same path you "
                  "gave to --capture-crops", input_root)
        return 1
    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("label", 320, 420)
    crops = sorted(unlabeled.glob("*.png"))
    log.info("labeling %d crops in %s", len(crops), unlabeled)
    log.info("rank keys: a 2 3 4 5 6 7 8 9 0(=10) j q k     suit keys: h d c s")
    log.info("n = skip,  ESC or 'Q' = quit")
    labeled = 0
    for crop_path in crops:
        if not crop_path.exists():
            continue
        result = _label_one(crop_path, input_root)
        if result == "QUIT":
            break
        if result is not None:
            labeled += 1
    cv2.destroyAllWindows()
    log.info("done. labeled %d crops. retrain with:", labeled)
    log.info("  python train_classifier.py --data %s --epochs 20", input_root)
    return 0


def batch_loop(input_root: Path, mapping_file: Path) -> int:
    """Headless labeling using a JSON mapping {filename: 'AH', ...}."""
    unlabeled = input_root / "unlabeled"
    if not unlabeled.exists():
        log.error("no 'unlabeled' directory under %s", input_root)
        return 1
    mapping = json.loads(mapping_file.read_text())
    labeled = 0
    for name, label in mapping.items():
        if label not in ALL_LABELS:
            log.warning("skipping invalid label %s for %s", label, name)
            continue
        src = unlabeled / name
        if not src.exists():
            continue
        dest = input_root / label / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        labeled += 1
    log.info("batch-labeled %d crops", labeled)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=str, required=True,
                   help="Same path passed to run.py --capture-crops")
    p.add_argument("--batch", type=str, default=None,
                   help="Optional JSON mapping {filename: 'AH', ...} for "
                        "headless labeling (no GUI)")
    args = p.parse_args()
    root = Path(args.input).expanduser()
    if args.batch:
        return batch_loop(root, Path(args.batch))
    return interactive_loop(root)


if __name__ == "__main__":
    sys.exit(main())
