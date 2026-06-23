"""Check 0623_multiples raw binary geometry without extracting the full zip."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


_NAME_RE = re.compile(r"_ns(?P<ns>\d+)ng(?P<ng>\d+)nt(?P<nt>\d+)\.bin$")


def _shape_from_name(name: str) -> tuple[int, int, int]:
    match = _NAME_RE.search(Path(name).name)
    if match is None:
        raise ValueError(f"Cannot parse ns/ng/nt from {name!r}.")
    return int(match["ns"]), int(match["ng"]), int(match["nt"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Check free_surface/sim_abs_ghost volume shape.")
    parser.add_argument("--zip", default="D:/edge_download/0623_multiples.zip", help="Path to 0623_multiples.zip.")
    parser.add_argument("--dtype-bytes", type=int, default=4, help="Bytes per sample; float32 is 4.")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            info for info in zf.infolist()
            if info.filename.endswith(".bin") and ("free_surface" in info.filename or "sim_abs_ghost" in info.filename)
        ]
        if len(members) != 2:
            raise RuntimeError(f"Expected 2 data members, found {len(members)}: {[m.filename for m in members]}")
        for info in sorted(members, key=lambda x: x.filename):
            shape = _shape_from_name(info.filename)
            expected = shape[0] * shape[1] * shape[2] * args.dtype_bytes
            ok = "OK" if expected == info.file_size else "SIZE_MISMATCH"
            print(f"{Path(info.filename).name}: shape={shape}, bytes={info.file_size}, expected={expected}, {ok}")


if __name__ == "__main__":
    main()
