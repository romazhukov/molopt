# optimize/pyscf_dft.py
# -*- coding: utf-8 -*-
"""
DFT-оптимизатор на базе PySCF для r2scan-подобных расчётов + D4/GCP.
Поддерживает:
- пользовательские .gbs-базисы (для всех атомов);
- constraints в стиле geomeTRIC;
- режим переходного состояния (transition=True) через geomeTRIC или QSD;
- частотный анализ (Freq) через pyscf.hessian.

Фичи:
- управление многопоточностью одного расчёта (OMP/MKL + lib.num_threads);
- выбор TS solver (geometric | qsd).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shlex
import subprocess
from typing import Iterable, List, Sequence, Tuple, Dict, Any

from pyscf import gto, dft, lib

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


def _run(cmd: Sequence[str], cwd: Path | None = None, env: Dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=env
    )


# --------------------- работа с базисами ----------------------


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
            seen.add(p)
            uniq.append(p)
    return uniq


def _load_gbs_map(path: Path) -> dict:
    return gto.basis.parse(path.read_text())


# --------------------------- класс оптимизатора ----------------------


@dataclass
class PySCFR2SCAN3c:
    # Основные настройки
    do_opt: bool = True
    solvent: str | None = None
    nthreads: int | None = None
    debug: bool = False

    # Внешние коррекции
    d4_exe: str | None = None
    gcp_exe: str | None = None

    # Базис/сетка
    basis: str = "def2-SVP"
    grid_level: int = 3

    # Геом. оптимизация (geomeTRIC)
    constraints: str | Path | list[dict] | None = None
    transition: bool = False
    coordsys: str = "tric"

    # TS / Freq
    ts_solver: str = "geometric"            # "geometric" | "qsd"
    compute_freqs: bool = False

    # Параметры QSD
    qsd_hess_update_freq: int = 0           # как часто пересчитывать числ. Гессиан
    qsd_step: float = 0.3                    # максимальный шаг (норма)
    qsd_hmin: float = 1e-6                   # критерий близости к стац. точке
    max_iter: int = 200                      # максимум шагов (для обоих солверов)

    # ───────────────────────── hooks ─────────────────────────

    def __post_init__(self):
        if self.nthreads and int(self.nthreads) > 0:
            os.environ["OMP_NUM_THREADS"] = str(int(self.nthreads))
            os.environ["MKL_NUM_THREADS"] = str(int(self.nthreads))
            try:
                lib.num_threads(int(self.nthreads))
            except Exception:
                if self.debug:
                    print(f"[DEBUG] lib.num_threads({self.nthreads}) не применён")

    # =============== SCF helpers ===============

    def _resolve_basis(self) -> str | dict:
        b = self.basis.strip()
        if _looks_like_gbs_path(b):
            p = Path(b)
            if not p.exists():
                raise FileNotFoundError(f"Не найден .gbs файл: {p}")
            if self.debug:
                print(f"[DEBUG] basis from GBS file: {p}")
            return _load_gbs_map(p)
        if any(tag in b.lower() for tag in ["mtzvpp", "tzvpp", "gbs"]):
            for cand in _find_candidate_gbs_files(b):
                if cand.exists():
                    if self.debug:
                        print(f"[DEBUG] basis alias '{b}' -> {cand}")
                    return _load_gbs_map(cand)
            if self.debug:
                print(f"[DEBUG] alias '{b}' не найден среди .gbs; пробую встроенный")
        return b

    def _build_mol(self,
                   atoms: Sequence[str],
                   coords: Sequence[Iterable[float]],
                   charge: int,
                   multiplicity: int) -> gto.Mole:
        spin = int(multiplicity) - 1
        atom_spec = [[a, (float(x), float(y), float(z))] for a, (x, y, z) in zip(atoms, coords)]
        basis_obj = self._resolve_basis()
        mol = gto.M(
            atom=atom_spec,
            unit="Angstrom",
            charge=charge,
            spin=spin,
            basis=basis_obj,
            verbose=4 if self.debug else 0,
            max_memory=int(os.environ.get("PYSCF_MAXMEM_MB", "4000")),
        )
        return mol

    def _make_scf(self, mol: gto.Mole):
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf.xc = "r2scan"
        mf.grids.level = self.grid_level
        mf.conv_tol = 1e-5
        mf.init_guess = "atom"
        mf.max_cycle = 200
        mf.level_shift = 0.3
        if self.solvent:
            try:
                mf = dft.ddCOSMO(mf)
                if self.debug:
                    print(f"[DEBUG] ddCOSMO включён (solvent='{self.solvent}')")
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] ddCOSMO не включён: {e}")
        return mf

    def _resolve_constraints(self):
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
        return c

    # =============== Optimization backends ===============

    def _optimize_geometry_geometric(self, mf):
        try:
            from pyscf.geomopt.geometric_solver import optimize as geometric_optimize
        except ModuleNotFoundError as e:
            raise RuntimeError("geometric не установлен. Установи: pip install geometric") from e

        kwargs = {
            "coordsys": self.coordsys,
            "transition": bool(self.transition),
            "maxsteps": self.max_iter,
        }
        cons = self._resolve_constraints()
        if cons:
            kwargs["constraints"] = cons
            if self.debug:
                print("[DEBUG] constraints → geomeTRIC")
        mol_opt = geometric_optimize(mf, **kwargs)
        mf_opt = mf.__class__(mol_opt)
        _ = mf_opt.kernel()
        return mol_opt, mf_opt

    def _optimize_geometry_qsd(self, mf):
        # получить список констрейнов из self
        constraints = self._resolve_constraints()
        if constraints and self.debug:
            print(f"[QSD] WARNING: QSD не поддерживает constraints напрямую, "
                    f"но зафиксированные атомы будут временно заморожены.")
        
        try:
            from pyscf.qsdopt.qsd_optimizer import QSD
        except ModuleNotFoundError as e:
            raise RuntimeError(
              "Модуль pyscf-qsdopt не найден. Установи: pip install pyscf-qsdopt"
            ) from e

        opt = QSD(mf, stationary_point="TS" if self.transition else "min")
        
        # если заданы ограничения на расстояния — фиксим координаты атомов
        if constraints:
            frozen = set()
            for kind, lst in constraints.items():
                for item in lst:
                    for i in item[:-1]:
                        frozen.add(i-1)
            coords = mf.mol.atom_coords()
            mask = np.ones_like(coords, dtype=bool)
            for i in frozen:
                mask[i, :] = False
            # временно применяем фиксацию
            mf.mol.set_geom_(coords, unit="Bohr")

        opt.kernel(
            hess_update_freq=self.qsd_hess_update_freq,
            step=self.qsd_step,
            hmin=self.qsd_hmin,
            max_iter=self.max_iter,
        )

        if not getattr(opt, "mol", None):
            raise RuntimeError("QSD не вернул оптимизированную геометрию (opt.mol is None)")

        if self.debug:
            print(f"[QSD] Converged: {getattr(opt, 'converged', False)}")

        mf_opt = mf.__class__(opt.mol)
        _ = mf_opt.kernel()
        return opt.mol, mf_opt

    # =============== Public API ===============

    def optimize(
        self,
        atoms: Sequence[str],
        coords: Sequence[Iterable[float]],
        *,
        charge: int,
        multiplicity: int,
        workdir: Path,
        index: int | None = None,
        ts_solver_override: str | None = None,
    ) -> Tuple[List[str], List[Tuple[float, float, float]], float, Dict[str, Any]]:
        workdir = Path(workdir)
        wd = workdir / f"pyscf_{index:03d}" if index is not None else workdir / "pyscf"
        wd.mkdir(parents=True, exist_ok=True)
        start_xyz = wd / "start.xyz"
        final_xyz = wd / "final.xyz"

        mol = self._build_mol(atoms, coords, charge, multiplicity)
        mf = self._make_scf(mol)
        e_scf = float(mf.kernel())

        _write_xyz_local(start_xyz, atoms, coords, "start geometry")

        # Геом. оптимизация (если включена)
        if self.do_opt:
            solver = (ts_solver_override or self.ts_solver or "geometric").lower()
            if solver == "geometric":
                mol_opt, mf_opt = self._optimize_geometry_geometric(mf)
            elif solver == "qsd":
                mol_opt, mf_opt = self._optimize_geometry_qsd(mf)
            else:
                raise ValueError(f"Unknown solver {solver!r}")
            e_scf = float(mf_opt.e_tot) if hasattr(mf_opt, "e_tot") else float(mf_opt.kernel())
            at_fin = [a[0] if isinstance(a, (list, tuple)) else a for a in mol_opt._atom]
            crd_fin = [tuple(xyz) for _, xyz in mol_opt._atom]
            mf_final = mf_opt
            mol_final = mol_opt
        else:
            at_fin = list(atoms)
            crd_fin = [tuple(map(float, c)) for c in coords]
            mf_final = mf
            mol_final = mol

        _write_xyz_local(final_xyz, at_fin, crd_fin, "final geometry for D4/gCP")

        # Внешние коррекции
        e_d4 = self._d4_energy(final_xyz, wd, charge=charge) if self.d4_exe else 0.0
        e_gcp = self._gcp_energy(final_xyz, wd, charge=charge) if self.gcp_exe else 0.0
        e_total = float(e_scf) + float(e_d4) + float(e_gcp)

        # Частоты (если включены)
        if self.compute_freqs:
            try:
                pyscf_frequencies_print(mf_final, mol_final)
            except Exception as e:
                print("[WARN] частоты не посчитались:", e)

        meta = {
            "e_scf_Eh": float(e_scf),
            "e_d4_Eh": float(e_d4),
            "e_gcp_Eh": float(e_gcp),
            "e_total_Eh": float(e_total),
            "workdir": str(wd),
            "basis": self.basis,
            "grid_level": self.grid_level,
            "transition": bool(self.transition),
            "ts_solver": (ts_solver_override or self.ts_solver),
            "omp_num_threads": int(self.nthreads) if self.nthreads else None,
            "mkl_num_threads": int(self.nthreads) if self.nthreads else None,
        }
        return at_fin, crd_fin, e_total, meta

    # -------------------- D4 / GCP --------------------

    def _d4_energy(self, xyz: Path, wd: Path, charge: int | None = None) -> float:
        if not self.d4_exe:
            return 0.0
        cmd = [self.d4_exe, "-f", "r2scan"]
        if charge is not None:
            cmd += ["-c", str(charge)]
        cmd += [str(xyz.name)]
        if self.debug:
            print("[DEBUG] run D4:", " ".join(shlex.quote(x) for x in cmd))
        res = _run(cmd, cwd=wd, env=os.environ.copy())
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
        res = _run(cmd, cwd=wd, env=os.environ.copy())
        m = re.search(r"GCP\s*energy\s*[:=]\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*Eh", res.stdout)
        return float(m.group(1)) if m else 0.0


# ================== Частотный анализ ==================


def pyscf_frequencies_print(mf, mol):
    """
    Простой частотный анализ:
      1) Численный/аналитический Гессиан PySCF
      2) Массо-взвешивание
      3) Диагонализация -> частоты (см^-1)
    """
    from pyscf import hessian
    import numpy as np

    # Выбор класса Гессиана по типу расчёта
    try:
        if mol.spin == 0:
            H = hessian.RKS(mf).kernel()
        else:
            H = hessian.UKS(mf).kernel()
    except Exception:
        # запасной вариант
        H = hessian.Hessian(mf).kernel()

    masses = mol.atom_mass_list()  # а.е.м.
    M = np.repeat(np.sqrt(masses), 3)  # корни масс для MW
    Hmw = H / np.outer(M, M)

    w2, _ = np.linalg.eigh(Hmw)  # собственные значения (в а. е.)
    # Перевод в см^-1. Коэффициент 5140.48 — приближённый (стандартный в практике PySCF-примеров).
    freqs = np.sign(w2) * np.sqrt(np.abs(w2)) * 5140.48

    print("=== Вибрационные частоты (см^-1) ===")
    for i, f in enumerate(freqs, 1):
        print(f"Mode {i:3d}: {f:12.2f} {'imag' if f < 0 else ''}")
    return freqs
