# conformers/crest_generator.py
from __future__ import annotations
import subprocess as sp
from pathlib import Path
from typing import List, Tuple

from io_mod.xyz_reader import read_xyz_many, write_xyz

Atom = str
Coords = List[Tuple[float, float, float]]


class CRESTGenerator:
    """
    Генератор конформеров через CREST.
    """

    def __init__(self, level: str = "gfn2", debug: bool = False):
        self.level = level
        self.debug = debug

    def generate(
        self,
        atoms: List[Atom],
        coords0: Coords,
        charge: int = 0,
        multiplicity: int = 1,
        workdir: str | Path = "results",
        tag: str = "crest",
        ewin_kcal: float = 6.0,
    ) -> List[Tuple[List[Atom], Coords]]:
        """
        Запускает CREST, возвращает список [(atoms, coords)].
        """
        workdir = Path(workdir).absolute()
        wd = workdir / tag
        wd.mkdir(parents=True, exist_ok=True)

        inp = wd / "input.xyz"
        write_xyz(inp, atoms, coords0, comment="input for CREST")

        cmd = [
            "crest",
            str(inp),
            f"--{self.level}",
            "--chrg", str(charge),
            "--uhf", str(multiplicity - 1),
            "--ewin", str(ewin_kcal),
        ]

        if self.debug:
            print("CREST CMD:", " ".join(cmd))

        res = sp.run(cmd, cwd=wd, capture_output=True, text=True)
        if self.debug:
            print(res.stdout)
            print(res.stderr)
        if res.returncode != 0:
            # Сохраним stderr для отладки
            (wd / "stderr_crest.txt").write_text(res.stderr or "", encoding="utf-8")
            raise RuntimeError(f"CREST rc={res.returncode}; см. {wd/'stderr_crest.txt'}")

        xyz_path = wd / "crest_conformers.xyz"
        if not xyz_path.exists():
            raise RuntimeError(f"Файл {xyz_path} не найден после CREST.")

        confs = read_xyz_many(xyz_path)
        if not confs:
            raise RuntimeError(f"Файл {xyz_path} пуст или не распознан.")

        if self.debug:
            print(f"CREST дал конформеров: {len(confs)}")

        return confs
