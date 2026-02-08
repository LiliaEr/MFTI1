# face_pipeline.py
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core import PipelineConfig, FacePipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Face pipeline (CPU) + embedding compare")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--img", type=str, help="Path to a single image (infer + save debug)")
    g.add_argument("--img1", type=str, help="Path to image #1 (compare mode)")

    p.add_argument("--img2", type=str, default=None, help="Path to image #2 (compare mode; required if --img1 is used)")
    p.add_argument("--out", type=str, default=None, help="Output debug folder (default: no debug)")
    p.add_argument("--mode", type=str, default="arcface", choices=["arcface", "ce"], help="Embedding model mode")
    p.add_argument("--det_th", type=float, default=0.80, help="Detector confidence threshold (0..1)")
    p.add_argument("--max_faces", type=int, default=1, help="How many faces to use from each image (top by score)")
    p.add_argument("--hourglass", type=str, default="stacked_hourglass_best.pt", help="Hourglass checkpoint path")
    p.add_argument("--arcface", type=str, default="arcface_best_fr.pt", help="ArcFace backbone checkpoint path")
    p.add_argument("--ce", type=str, default="ce_best_700.pt", help="CE checkpoint path")
    return p


def run_single(args) -> int:
    cfg = PipelineConfig(
        hourglass_ckpt=args.hourglass,
        arcface_ckpt=args.arcface if args.mode == "arcface" else None,
        ce_ckpt=args.ce if args.mode == "ce" else None,
        device="cpu",
        fp16=False,
        det_conf_th=float(args.det_th),
        debug_dir=args.out,
        debug_max_faces=max(1, int(args.max_faces)),
    )

    pipe = FacePipeline(cfg, embed_mode=args.mode)

    res = pipe.infer(args.img)
    print("faces:", len(res))
    if len(res) == 0:
        raise RuntimeError(f"No face detected: {args.img}")

    r0 = res[0]
    print("first face score:", r0.score)
    print("first embedding shape:", r0.embedding.shape)
    if args.out:
        print("debug saved to:", str(Path(args.out).resolve()))
    return 0


def run_compare(args) -> int:
    if not args.img2:
        raise SystemExit("In compare mode you must provide --img2")

    out_root = Path(args.out) if args.out else None
    out1 = str(out_root / "img1") if out_root else None
    out2 = str(out_root / "img2") if out_root else None

    cfg1 = PipelineConfig(
        hourglass_ckpt=args.hourglass,
        arcface_ckpt=args.arcface if args.mode == "arcface" else None,
        ce_ckpt=args.ce if args.mode == "ce" else None,
        device="cpu",
        fp16=False,
        det_conf_th=float(args.det_th),
        debug_dir=out1,
        debug_max_faces=max(1, int(args.max_faces)),
    )
    cfg2 = PipelineConfig(
        hourglass_ckpt=args.hourglass,
        arcface_ckpt=args.arcface if args.mode == "arcface" else None,
        ce_ckpt=args.ce if args.mode == "ce" else None,
        device="cpu",
        fp16=False,
        det_conf_th=float(args.det_th),
        debug_dir=out2,
        debug_max_faces=max(1, int(args.max_faces)),
    )

    pipe1 = FacePipeline(cfg1, embed_mode=args.mode)
    pipe2 = FacePipeline(cfg2, embed_mode=args.mode)

    res1 = pipe1.infer(args.img1)
    if len(res1) == 0:
        raise RuntimeError(f"No face detected in img1: {args.img1}")

    res2 = pipe2.infer(args.img2)
    if len(res2) == 0:
        raise RuntimeError(f"No face detected in img2: {args.img2}")

    e1 = res1[0].embedding
    e2 = res2[0].embedding

    cosine_dist = FacePipeline.distance(e1, e2, metric="cosine")
    l2_dist = FacePipeline.distance(e1, e2, metric="l2")

    print("=== FACE EMBEDDING COMPARISON ===")
    print("image 1:", args.img1)
    print("image 2:", args.img2)
    print(f"cosine_dist = {cosine_dist:.4f}")
    print(f"l2_dist     = {l2_dist:.4f}")

    if args.out:
        print("debug saved to:", str(out_root.resolve()))
        print(" - img1 debug:", out1)
        print(" - img2 debug:", out2)

    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.img:
        return run_single(args)
    return run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
