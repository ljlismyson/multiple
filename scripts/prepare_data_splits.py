"""Split 0623_multiples paired raw volumes into train/val/test bin files.

The split is along the shot axis (ns): 704/88/88 for an 8:1:1 split of
the original (880, 481, 3300) volumes.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import BinaryIO


FREE_SURFACE_MEMBER = "0623_multiples/model_v1/data_free_surface_ns880ng481nt3300.bin"
SIM_ABS_GHOST_MEMBER = "0623_multiples/model_v1/data_sim_abs_ghost_ns880ng481nt3300.bin"


def _copy_exact(src: BinaryIO, dst: BinaryIO, n_bytes: int) -> None:
    remaining = int(n_bytes)
    while remaining > 0:
        chunk = src.read(min(1024 * 1024, remaining))
        if not chunk:
            raise EOFError(f"Unexpected EOF with {remaining} bytes left to copy.")
        dst.write(chunk)
        remaining -= len(chunk)


def _split_member(
    zf: zipfile.ZipFile,
    *,
    member: str,
    out_dir: Path,
    stem: str,
    split_counts: dict[str, int],
    ng: int,
    nt: int,
    dtype_bytes: int,
    overwrite: bool,
) -> None:
    shot_bytes = int(ng) * int(nt) * int(dtype_bytes)
    with zf.open(member) as src:
        for split_name, ns in split_counts.items():
            split_dir = out_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            out_path = split_dir / f"{stem}_ns{ns}ng{ng}nt{nt}.bin"
            if out_path.exists():
                if not overwrite:
                    raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it.")
                out_path.unlink()
            with out_path.open("wb") as dst:
                _copy_exact(src, dst, ns * shot_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 8:1:1 train/val/test bin splits.")
    parser.add_argument("--zip", default="D:/edge_download/0623_multiples.zip", help="Path to 0623_multiples.zip.")
    parser.add_argument("--out-dir", default="data", help="Output directory that will contain train/val/test.")
    parser.add_argument("--ns", type=int, default=880)
    parser.add_argument("--ng", type=int, default=481)
    parser.add_argument("--nt", type=int, default=3300)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    n_train = int(args.ns * 0.8)
    n_val = int(args.ns * 0.1)
    n_test = int(args.ns) - n_train - n_val
    split_counts = {"train": n_train, "val": n_val, "test": n_test}

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        for split_name in split_counts:
            split_dir = out_dir / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)

    with zipfile.ZipFile(Path(args.zip)) as zf:
        _split_member(
            zf,
            member=FREE_SURFACE_MEMBER,
            out_dir=out_dir,
            stem="free_surface",
            split_counts=split_counts,
            ng=args.ng,
            nt=args.nt,
            dtype_bytes=args.dtype_bytes,
            overwrite=args.overwrite,
        )
        _split_member(
            zf,
            member=SIM_ABS_GHOST_MEMBER,
            out_dir=out_dir,
            stem="sim_abs_ghost",
            split_counts=split_counts,
            ng=args.ng,
            nt=args.nt,
            dtype_bytes=args.dtype_bytes,
            overwrite=args.overwrite,
        )

    for split_name, ns in split_counts.items():
        print(f"{split_name}: ns={ns}, ng={args.ng}, nt={args.nt}, dir={out_dir / split_name}")


if __name__ == "__main__":
    main()
