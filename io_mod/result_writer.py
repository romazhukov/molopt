from pathlib import Path
from typing import List, Tuple, Dict, Any
import json

Atom = str
Coord = Tuple[float, float, float]


def save_best_xyz(outdir: Path, atoms: List[Atom], coords: List[Coord], tag: str = "best") -> Path:
    outpath = outdir / f"{tag}.xyz"
    with open(outpath, "w") as f:
        f.write(f"{len(atoms)}\n{tag}\n")
        for a, (x, y, z) in zip(atoms, coords):
            f.write(f"{a:2s}  {x: .6f}  {y: .6f}  {z: .6f}\n")
    return outpath


def save_all_xyz(outdir: Path, conformers: List[Tuple[List[Atom], List[Coord]]], tag: str = "all") -> Path:
    outpath = outdir / f"{tag}_conformers.xyz"
    with open(outpath, "w") as f:
        for i, (atoms, coords) in enumerate(conformers, start=1):
            f.write(f"{len(atoms)}\nconf {i}\n")
            for a, (x, y, z) in zip(atoms, coords):
                f.write(f"{a:2s}  {x: .6f}  {y: .6f}  {z: .6f}\n")
    return outpath


def save_report(outdir: Path, data: Dict[str, Any]) -> Path:
    outpath = outdir / "report.json"
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2)
    return outpath
