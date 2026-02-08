from __future__ import annotations

import os
import shutil
from pathlib import Path
import pandas as pd

# ====== CONFIG ======
SRC_ROOT = Path(r"D:\МФТИ\DL\archive (2)")          # папка с подпапками и картинками
CSV_PATH = Path(r"D:\МФТИ\DL\selected_arcface_mainset_1.csv")
OUT_ROOT = Path(r"D:\МФТИ\DL\arcface_mainset_1")     # куда собирать датасет

MODE = "hardlink"   # "hardlink" | "copy"
EXTS = {".jpg", ".jpeg", ".png"}

# ====================

def build_index(src_root: Path) -> dict[str, Path]:
    """image_id -> full path"""
    idx = {}
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            # ключ = имя файла, например 000123.jpg
            idx[p.name] = p
    return idx

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def place_file(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)

    if mode == "hardlink":
        # hardlink работает только на одном диске
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError("MODE must be 'hardlink' or 'copy'")

def main():
    df = pd.read_csv(CSV_PATH)
    if not {"image_id", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: image_id, label")

    print("Indexing source images (may take a bit)...")
    idx = build_index(SRC_ROOT)
    print(f"Indexed: {len(idx):,} files")

    missing = []
    placed = 0

    for row in df.itertuples(index=False):
        image_id = getattr(row, "image_id")
        label = getattr(row, "label")

        src = idx.get(image_id)
        if src is None:
            missing.append(image_id)
            continue

        dst = OUT_ROOT / str(label) / image_id
        place_file(src, dst, MODE)
        placed += 1

    print(f"Placed: {placed:,} / {len(df):,} ({MODE})")
    if missing:
        print(f"Missing: {len(missing)}")
        # сохраним список
        miss_path = OUT_ROOT / "missing_image_ids.txt"
        ensure_dir(OUT_ROOT)
        miss_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"Missing list saved to: {miss_path}")

if __name__ == "__main__":
    main()
