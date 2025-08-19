import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

FINGERS = ["index", "middle", "ring", "pinky", "thumb"]
SIDES = ["left", "right"]

SIDE_FINGER_RE = re.compile(
    r'(left|right)[-_]?(index|middle|ring|pinky|thumb)',
    re.IGNORECASE
)


def parse_side_finger(path: Path) -> Optional[Tuple[str, str]]:
    """
    Inspect all parts of a path and try to extract (side, finger),
    e.g. ('right', 'index'). Returns None if not found.
    """
    # Try each part of the path string
    for part in path.parts[::-1]:  # search from deepest upwards
        m = SIDE_FINGER_RE.search(part)
        if m:
            side = m.group(1).lower()
            finger = m.group(2).lower()
            return side, finger
    # Try on the full lower-cased path string as a fallback
    m = SIDE_FINGER_RE.search(str(path).lower())
    if m:
        return m.group(1).lower(), m.group(2).lower()
    return None


def find_runs_with_models(src_date_dir: Path) -> List[Path]:
    """
    Return a list of directories that contain a 'models' subdirectory.
    """
    runs = []
    for root, dirs, files in os.walk(src_date_dir):
        if 'models' in dirs:
            runs.append(Path(root))
    return runs


def copy_or_move(src: Path, dst: Path, move: bool, dry_run: bool, keep_existing: bool):
    if dst.exists():
        if keep_existing:
            print(f"[skip] {dst} exists (keep-existing)")
            return
        else:
            # Overwrite by removing file first to avoid partial copy issues
            print(f"[rm]   {dst}")
            if not dry_run:
                dst.unlink()
    action = "move" if move else "copy"
    print(f"[{action}] {src} -> {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))


def latest_by_mtime(paths: List[Path]) -> Optional[Path]:
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Reorganize checkpoints by finger into a compact layout.")
    parser.add_argument("--src", required=True, type=Path, help="Source date folder, e.g., out/2025.08.17")
    parser.add_argument("--dst", required=True, type=Path, help="Destination date folder, e.g., checkpoint_reorganized/2025.08.17")
    parser.add_argument("--all-models", action="store_true", help="Copy all *.pt files (default: only *_best.pt)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only")
    parser.add_argument("--keep-existing", action="store_true", help="Do not overwrite if the destination file exists")
    args = parser.parse_args()

    src_date_dir: Path = args.src
    dst_date_dir: Path = args.dst
    dst_date_dir.mkdir(parents=True, exist_ok=True)

    runs = find_runs_with_models(src_date_dir)
    if not runs:
        print(f"No runs with 'models' found under {src_date_dir}")
        return

    print(f"Found {len(runs)} runs with 'models' under {src_date_dir}\n")

    # We keep track of best checkpoints seen per (side, finger)
    kept_best = {}

    for run_dir in runs:
        sf = parse_side_finger(run_dir)
        if not sf:
            print(f"[warn] Could not parse side/finger from: {run_dir}")
            continue
        side, finger = sf
        if side not in SIDES or finger not in FINGERS:
            print(f"[warn] Unrecognized side/finger in: {run_dir}")
            continue

        src_models = run_dir / "models"
        if not src_models.is_dir():
            print(f"[warn] Missing models dir for: {run_dir}")
            continue

        # Destination layout
        subdir_name = f"{side}_{finger}"
        dst_sub = dst_date_dir / subdir_name
        dst_models = dst_sub / "models"
        dst_hydra = dst_sub / ".hydra"

        # 1) Copy hydra files if present
        src_hydra = run_dir / ".hydra"
        hydra_files = ["config.yaml", "hydra.yaml", "overrides.yaml"]
        if src_hydra.is_dir():
            for hf in hydra_files:
                s = src_hydra / hf
                if s.exists():
                    d = dst_hydra / hf
                    copy_or_move(s, d, move=args.move, dry_run=args.dry_run, keep_existing=args.keep_existing)

        # 2) Copy dataset_stats.pkl if present in run or one level up
        candidates = [
            run_dir / "dataset_stats.pkl",
            run_dir.parent / "dataset_stats.pkl",
        ]
        chosen_stats = latest_by_mtime([c for c in candidates if c.exists()])
        if chosen_stats:
            copy_or_move(chosen_stats, dst_sub / "dataset_stats.pkl", move=args.move, dry_run=args.dry_run, keep_existing=args.keep_existing)

        # 3) Copy models
        pt_files = list(src_models.glob("*.pt"))
        if not pt_files:
            print(f"[warn] No .pt files in {src_models}")
            continue

        if args.all_models:
            for p in pt_files:
                d = dst_models / p.name
                copy_or_move(p, d, move=args.move, dry_run=args.dry_run, keep_existing=args.keep_existing)
        else:
            # Only best
            enc_best = [p for p in pt_files if p.name.startswith("encoder_") and p.name.endswith("best.pt")]
            dec_best = [p for p in pt_files if p.name.startswith("decoder_") and p.name.endswith("best.pt")]
            # If multiple, keep the latest by mtime
            enc_choice = latest_by_mtime(enc_best) if enc_best else None
            dec_choice = latest_by_mtime(dec_best) if dec_best else None

            if not enc_choice and not dec_choice:
                print(f"[warn] No *_best.pt in {src_models}; nothing copied for {side}_{finger}")
                continue

            # If multiple runs produce the same finger, keep the latest we see overall
            key = (side, finger)
            prev = kept_best.get(key, {})
            # Compare times and overwrite later if needed
            if enc_choice:
                prev_enc = prev.get("encoder")
                if (not prev_enc) or (enc_choice.stat().st_mtime > prev_enc.stat().st_mtime):
                    kept_best.setdefault(key, {})["encoder"] = enc_choice
                enc_to_copy = enc_choice
            else:
                enc_to_copy = None

            if dec_choice:
                prev_dec = prev.get("decoder")
                if (not prev_dec) or (dec_choice.stat().st_mtime > prev_dec.stat().st_mtime):
                    kept_best.setdefault(key, {})["decoder"] = dec_choice
                dec_to_copy = dec_choice
            else:
                dec_to_copy = None

            # Copy the chosen ones
            if enc_to_copy:
                copy_or_move(enc_to_copy, dst_models / enc_to_copy.name, move=args.move, dry_run=args.dry_run, keep_existing=args.keep_existing)
            if dec_to_copy:
                copy_or_move(dec_to_copy, dst_models / dec_to_copy.name, move=args.move, dry_run=args.dry_run, keep_existing=args.keep_existing)

    print("\nDone.")


if __name__ == "__main__":
    main()


''' Commands
python reorganize_checkpoints.py   --src ./checkpoints/control-trainings/out/2025.08.17   --dst ./checkpoint_reorganized/2025.08.17   --all-models
'''