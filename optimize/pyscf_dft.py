# optimize/pyscf_dft.py
# -*- coding: utf-8 -*-
"""
DFT-оптимизатор на базе PySCF для r2scan-подобных расчётов + D4/GCP.
Теперь поддерживает:
- пользовательские .gbs-базисы (для всех атомов);
- constraints в стиле geomeTRIC;
- режим переходного состояния (transition=True) через geomeTRIC.

Требуется: pip install geometric
Опционально: dftd4, gcp (внешние бинарники)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shlex
import subprocess
from typing import Iterable, List, Sequence, Tuple, Dict, Any

from pyscf import gto, dft

# --------------------------- утилиты ---------------------------------


def _write_xyz_local(xyz_path: Path,
                     atoms: Sequence[str],
                     coords: Sequence[Iterable[float]],
                     comment: str = "") -> None:
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    with open(xyz_path, "w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for a, (x, y, z) in zip(atoms, coords):
            f.write(f"{a:<2s} {x:16.10f} {y:16.10f} {z:16.10f}\n")


def _parse_d4_energy(stdout: str) -> float | None:
    m = re.search(r"Dispersion energy:\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*Eh", stdout)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    m2 = re.search(r"EDISP\s*=\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", stdout)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            return None
    return None


def _run(cmd: Sequence[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


# --------------------- работа с базисами (.gbs) ----------------------


def _looks_like_gbs_path(s: str) -> bool:
    sl = s.lower()
    return sl.endswith(".gbs") or "/" in s or "\\" in s or os.path.isfile(s)


def _find_candidate_gbs_files(name: str) -> list[Path]:
    stem = name.replace("_", "-").lower()
    candidates = [
        Path(f"{stem}.gbs"),
        Path(f"{stem}.1.gbs"),
        Path("basis") / f"{stem}.gbs",
        Path("basis") / f"{stem}.1.gbs",
        Path.cwd() / f"{stem}.gbs",
        Path.cwd() / f"{stem}.1.gbs",
        Path(__file__).resolve().parent.parent / "basis" / f"{stem}.gbs",
        Path(__file__).resolve().parent.parent / "basis" / f"{stem}.1.gbs",
    ]
    uniq, seen = [], set()
    for p in candidates:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def _load_gbs_map(path: Path) -> dict:
    txt = path.read_text()
    return gto.basis.parse(txt)


# --------------------------- класс оптимизатора ----------------------


@dataclass
class PySCFR2SCAN3c:
    do_opt: bool = True
    solvent: str | None = None
    nthreads: int | None = None
    debug: bool = False
    d4_exe: str | None = None
    gcp_exe: str | None = None
    basis: str = "def2-SVP"      # имя встроенного или путь/алиас на .gbs
    grid_level: int = 3          # PySCF grids.level

    # Новые поля:
    constraints: str | Path | list[dict] | None = None  # форматы geomeTRIC
    transition: bool = False       # True -> искать седло (TS)
    coordsys: str = "tric"         # geomeTRIC internal coords (TRIC ~ по умолчанию)

    # ---------------------- приватные методы -------------------------

    def _resolve_basis(self) -> str | dict:
        b = self.basis.strip()
        if _looks_like_gbs_path(b):
            p = Path(b)
            if not p.exists():
                raise FileNotFoundError(f"Не найден .gbs файл: {p}")
            if self.debug: print(f"[DEBUG] GBS: {p}")
            return _load_gbs_map(p)
        if any(tag in b.lower() for tag in ["mtzvpp", "tzvpp", "gbs"]):
            for cand in _find_candidate_gbs_files(b):
                if cand.exists():
                    if self.debug: print(f"[DEBUG] найден .gbs по алиасу '{b}': {cand}")
                    return _load_gbs_map(cand)
            if self.debug: print(f"[DEBUG] алиас '{b}' не сопоставился с .gbs; пробую как встроенный")
        return b

    def _build_mol(self,
                   atoms: Sequence[str],
                   coords: Sequence[Iterable[float]],
                   charge: int,
                   multiplicity: int) -> gto.Mole:
        spin = int(multiplicity) - 1  # 2S; singlet=0
        atom_spec = [[a, (float(x), float(y), float(z))] for a, (x, y, z) in zip(atoms, coords)]
        basis_obj = self._resolve_basis()
        mol = gto.M(
            atom=atom_spec, unit="Angstrom", charge=charge, spin=spin,
            basis=basis_obj, verbose=4 if self.debug else 0,
            max_memory=int(os.environ.get("PYSCF_MAXMEM_MB", "4000")),
        )
        return mol

    def _make_scf(self, mol: gto.Mole):
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf.xc = "r2scan"
        mf.grids.level = self.grid_level
        mf.conv_tol = 1e-9
        if self.nthreads:
            os.environ["OMP_NUM_THREADS"] = str(self.nthreads)
        return mf

    def _resolve_constraints(self) -> str | list[dict] | None:
        """
        Возвращает constraints в формате, который понимает geomeTRIC.optimize:
        - None
        - строка с блоками $freeze/$set/$scan/...
        - список dict'ов (geomeTRIC Python-API)
        - путь к файлу -> загруженная строка
        """
        c = self.constraints
        if c is None:
            return None
        if isinstance(c, list):
            return c
        if isinstance(c, Path):
            return c.read_text()
        if isinstance(c, str):
            p = Path(c)
            if p.suffix.lower() == ".txt" and p.exists():
                return p.read_text()
            return c
        # на всякий — вернём как есть
        return c

    def _optimize_geometry(self, mf):
        """
        Оптимизация через geomeTRIC.
        transition=True включает поиск седла (TS).
        """
        try:
            from pyscf.geomopt.geometric_solver import optimize as geometric_optimize
        except ModuleNotFoundError:
            raise RuntimeError("geometric не установлен. Установи: pip install geometric")

        kwargs = {
            "coordsys": self.coordsys,
            "transition": bool(self.transition),
        }
        cons = self._resolve_constraints()
        if cons is not None:
            kwargs["constraints"] = cons
            if self.debug:
                print("[DEBUG] constraints → geomeTRIC:")
                if isinstance(cons, str):
                    print(cons.strip()[:1000] + ("\n... (truncated)" if len(cons) > 1000 else ""))
                else:
                    print(cons)

        mol_opt = geometric_optimize(mf, **kwargs)
        mf_opt = mf.__class__(mol_opt)
        _ = mf_opt.kernel()
        return mol_opt, mf_opt

    def _d4_energy(self, xyz: Path, wd: Path, charge: int | None = None) -> float:
        if not self.d4_exe:
            return 0.0
        cmd = [self.d4_exe, "-f", "r2scan"]
        if charge is not None:
            cmd += ["-c", str(charge)]
        cmd += [str(xyz.name)]
        if self.debug:
            print("[DEBUG] run D4:", " ".join(shlex.quote(x) for x in cmd))
        res = _run(cmd, cwd=wd)
        e = _parse_d4_energy(res.stdout)
        if e is None:
            raise RuntimeError(
                "Не удалось распарсить энергию D4 из вывода dftd4\n"
                f"stdout:\n{res.stdout}\n\nstderr:\n{res.stderr}"
            )
        return float(e)

    def _gcp_energy(self, xyz: Path, wd: Path, charge: int | None = None) -> float:
        if not self.gcp_exe:
            return 0.0
        cmd = [self.gcp_exe, str(xyz.name)]
        if charge is not None:
            cmd += ["-c", str(charge)]
        if self.debug:
            print("[DEBUG] run GCP:", " ".join(shlex.quote(x) for x in cmd))
        res = _run(cmd, cwd=wd)
        m = re.search(r"GCP\s*energy\s*[:=]\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*Eh", res.stdout)
        return float(m.group(1)) if m else 0.0

    # ------------------------- публичный API --------------------------

    def optimize(self,
                 atoms: Sequence[str],
                 coords: Sequence[Iterable[float]],
                 *,
                 charge: int,
                 multiplicity: int,
                 workdir: Path,
                 index: int | None = None) -> Tuple[List[str], List[Tuple[float, float, float]], float, Dict[str, Any]]:
        workdir = Path(workdir)
        wd = workdir / f"pyscf_{index:03d}" if index is not None else workdir / "pyscf"
        wd.mkdir(parents=True, exist_ok=True)
        start_xyz = wd / "start.xyz"
        final_xyz = wd / "final.xyz"

        mol = self._build_mol(atoms, coords, charge, multiplicity)
        mf = self._make_scf(mol)
        e_scf = float(mf.kernel())

        _write_xyz_local(start_xyz, atoms, coords, "start geometry")

        if self.do_opt:
            mol_opt, mf_opt = self._optimize_geometry(mf)
            e_scf = float(mf_opt.e_tot) if hasattr(mf_opt, "e_tot") else float(mf_opt.kernel())
            at_fin = [a[0] if isinstance(a, (list, tuple)) else a for a in mol_opt._atom]
            crd_fin = [tuple(xyz) for _, xyz in mol_opt._atom]
        else:
            at_fin = list(atoms)
            crd_fin = [tuple(map(float, c)) for c in coords]

        _write_xyz_local(final_xyz, at_fin, crd_fin, "final geometry for D4/gCP")

        e_d4 = self._d4_energy(final_xyz, wd, charge=charge) if self.d4_exe else 0.0
        e_gcp = self._gcp_energy(final_xyz, wd, charge=charge) if self.gcp_exe else 0.0
        e_total = float(e_scf) + float(e_d4) + float(e_gcp)

        meta = {
            "e_scf_Eh": float(e_scf),
            "e_d4_Eh": float(e_d4),
            "e_gcp_Eh": float(e_gcp),
            "e_total_Eh": float(e_total),
            "workdir": str(wd),
            "basis": self.basis,
            "grid_level": self.grid_level,
            "used_geometric": True,
            "transition": bool(self.transition),
            "coordsys": self.coordsys,
        }
        return at_fin, crd_fin, e_total, meta
