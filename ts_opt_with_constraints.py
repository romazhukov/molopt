#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TS optimization with constraints for PySCF + geomeTRIC

- PySCF (DFT, r2scan по умолчанию)
- geomeTRIC для оптимизации (включая TS через transition=True)
- Базисы: def2-SVP по умолчанию или Basis Set Exchange (--gbs),
  или готовый JSON (--gbs-json)
- Констрейны читаем из файла (--cinp) с блоками $constrain ... $end
- Поддержка частотного анализа (--freq) через PySCF hessian
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Iterable, Any

try:
    import basis_set_exchange as bse
    HAVE_BSE = True
except Exception:
    HAVE_BSE = False

from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize as geometric_optimize
from pyscf import hessian

# ===================== ЧТЕНИЕ XYZ =====================

def read_xyz(xyz_path: str) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    atoms: List[str] = []
    coords: List[Tuple[float, float, float]] = []
    with open(xyz_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(parts[0])
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms, coords

# ===================== БАЗИСЫ =====================

def load_pyscf_basis_from_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        basis_dict = json.load(f)
    return basis_dict

def make_pyscf_basis_from_bse(name: str, atoms: Sequence[str]) -> Dict[str, Any]:
    if not HAVE_BSE:
        raise RuntimeError("basis_set_exchange не установлен (pip install basis_set_exchange)")
    basis_dict: Dict[str, Any] = {}
    unique = sorted(set(atoms))
    for symb in unique:
        text = bse.get_basis(name, elements=[symb], fmt="nwchem")
        basis_dict[symb] = gto.basis.parse(text)
    return basis_dict

# ===================== КОНСТРЕЙНЫ =====================

def parse_cinp_constraints(cinp_path: str) -> Dict[str, Any]:
    data = {"distance": [], "angle": [], "dihedral": []}
    if not cinp_path:
        return data
    text = Path(cinp_path).read_text(encoding="utf-8")
    constr_blocks = re.findall(r'(?is)^\s*\$constrain\b(.*?)^\s*\$end\b', text, flags=re.M)
    for block in constr_blocks:
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^distance\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*([^\s,]+)$', line, flags=re.I)
            if m:
                data["distance"].append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
                continue
            m = re.match(r'^angle\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([^\s,]+)$', line, flags=re.I)
            if m:
                data["angle"].append((int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))))
                continue
            m = re.match(r'^dihedral\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([^\s,]+)$', line, flags=re.I)
            if m:
                data["dihedral"].append((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))))
                continue
    return data

def _in_range(*idxs: int, natoms: int) -> bool:
    return all(1 <= i <= natoms for i in idxs)

def validate_constraints(constraints: Dict[str, Any], natoms: int) -> Dict[str, Any]:
    out = {"distance": [], "angle": [], "dihedral": []}
    for (i, j, val) in constraints.get("distance", []):
        if _in_range(i, j, natoms=natoms):
            out["distance"].append((i, j, val))
    for (i, j, k, val) in constraints.get("angle", []):
        if _in_range(i, j, k, natoms=natoms):
            out["angle"].append((i, j, k, val))
    for (i, j, k, l, val) in constraints.get("dihedral", []):
        if _in_range(i, j, k, l, natoms=natoms):
            out["dihedral"].append((i, j, k, l, val))
    return out

def write_geometric_constraints_file(constraints: Dict[str, Any], path: str) -> None:
    lines: List[str] = []
    lines.append("$set")
    for (i, j, val) in constraints.get("distance", []):
        lines.append(f"distance {i} {j} {val}")
    for (i, j, k, val) in constraints.get("angle", []):
        lines.append(f"angle {i} {j} {k} {val}")
    for (i, j, k, l, val) in constraints.get("dihedral", []):
        lines.append(f"dihedral {i} {j} {k} {l} {val}")
    lines.append("$end")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

# ===================== PYSCF =====================

def build_pyscf(
    atoms: Sequence[str],
    coords: Sequence[Iterable[float]],
    *,
    charge: int = 0,
    multiplicity: int = 1,
    basis: Any = "def2-SVP",
    debug: bool = False
):
    spin = int(multiplicity) - 1
    atom_spec = [[a, (float(x), float(y), float(z))] for a, (x, y, z) in zip(atoms, coords)]
    mol = gto.M(
        atom=atom_spec,
        unit="Angstrom",
        charge=charge,
        spin=spin,
        basis=basis,
        verbose=4 if debug else 0,
    )
    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    mf.xc = "r2scan"

    mf.grids.level = 5
    mf.max_cycle = 200
    mf.level_shift = 0.5
    mf.damp = 0.2
    mf.diis_space = 12
    mf.conv_tol = 1e-6
    return mol, mf

# ===================== ЗАПУСК =====================

def run_job(
    xyz: str,
    *,
    gbs_name: str | None,
    gbs_json: str | None,
    cinp: str | None,
    ts: bool,
    charge: int,
    multiplicity: int,
    debug: bool,
    freq: bool,
    maxsteps: int,
) -> None:
    atoms, coords = read_xyz(xyz)

    # Базис
    if gbs_json:
        basis = load_pyscf_basis_from_json(gbs_json)
    elif gbs_name:
        basis = make_pyscf_basis_from_bse(gbs_name, atoms)
    else:
        basis = "def2-SVP"

    # PySCF
    mol, mf = build_pyscf(atoms, coords, charge=charge, multiplicity=multiplicity, basis=basis, debug=debug)

    # Констрейны
    constraints_raw = parse_cinp_constraints(cinp) if cinp else {"distance": [], "angle": [], "dihedral": []}
    constraints = validate_constraints(constraints_raw, natoms=len(atoms))

    tmp_dir = tempfile.TemporaryDirectory(prefix="geom_constraints_")
    constraints_file = os.path.join(tmp_dir.name, "constraints.txt")
    constraints_arg = None
    if any(constraints[k] for k in ("distance", "angle", "dihedral")):
        write_geometric_constraints_file(constraints, constraints_file)
        constraints_arg = constraints_file
        print(f"[INFO] constraints file written: {constraints_file}")
    else:
        print("[INFO] no valid constraints; продолжаем без ограничений")

    tmp_input = os.path.join(tmp_dir.name, "geom.inp")
    Path(tmp_input).write_text("# dummy input for geomeTRIC\n", encoding="utf-8")

    print(f"[INFO] запускаем geomeTRIC (transition={bool(ts)}, maxsteps={maxsteps})...")

    # --- автоостановка при застревании ---
    energies = []
    def callback(env):
        E = getattr(env, "e_tot", None)
        if E is not None:
            energies.append(E)
            if len(energies) > 5:
                dE = abs(energies[-1] - energies[-5])
                if dE < 1e-6:
                    print(f"[STOP] ΔE за 5 шагов < 1e-6 → останавливаю оптимизацию.")
                    raise RuntimeError("Optimization stalled")

    mol_opt = geometric_optimize(
        mf,
        constraints=constraints_arg,
        transition=bool(ts),
        maxsteps=maxsteps,
        callback=callback,
    )

    # частотный анализ
    if freq:
        print("[INFO] считаем частоты...")
        hess = hessian.Hessian(mf).kernel()
        import numpy as np
        from scipy.linalg import eigh
        m = mol.atom_mass_list()
        mw = np.repeat(m, 3)
        w, v = eigh(hess, np.diag(mw))
        freqs = (w * 219474.6) ** 0.5  # перевод в см^-1
        imags = [f for f in freqs if f < 0]
        if ts and len(imags) == 1:
            print(f"[CHECK] TS подтверждён: одна мнимая частота ({imags[0]:.2f} cm^-1)")
        elif ts and len(imags) != 1:
            print(f"[WARN] TS неверный: {len(imags)} мнимых частот")
        elif not ts and imags:
            print(f"[WARN] минимум не подтверждён: {len(imags)} мнимых частот")

    # сохранить xyz
    try:
        coords_fin = mol_opt.atom_coords(unit='Angstrom')
        xyz_out = Path("optimized.xyz")
        with xyz_out.open("w", encoding="utf-8") as f:
            f.write(f"{len(atoms)}\noptimized by geomeTRIC (TS={bool(ts)})\n")
            for a, (x, y, z) in zip([a[0] if isinstance(a, (list, tuple)) else a for a in mol_opt._atom], coords_fin):
                f.write(f"{a:<2} {x:16.10f} {y:16.10f} {z:16.10f}\n")
        print(f"[RESULT] Финальная геометрия сохранена: {xyz_out.resolve()}")
    except Exception as e:
        print(f"[WARN] не удалось сохранить optimized.xyz: {e}")

    tmp_dir.cleanup()

# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(description="TS optimization with constraints (PySCF + geomeTRIC)")
    ap.add_argument("xyz", help="Входной XYZ файл")
    ap.add_argument("--gbs", dest="gbs_name", help="Имя базиса из Basis Set Exchange (например, def2-TZVP)")
    ap.add_argument("--gbs-json", dest="gbs_json", help="Локальный JSON с PySCF-базисом")
    ap.add_argument("--cinp", help="Файл с $constrain блоками (input.inp)")
    ap.add_argument("--ts", action="store_true", help="Режим поиска переходного состояния")
    ap.add_argument("--charge", type=int, default=0, help="Заряд системы")
    ap.add_argument("--mult", type=int, default=1, help="Мультиплетность (2S+1)")
    ap.add_argument("--maxsteps", type=int, default=200, help="Максимум шагов геом. оптимизации")
    ap.add_argument("--freq", action="store_true", help="Считать частоты")
    ap.add_argument("--debug", action="store_true", help="Отладочный вывод")
    args = ap.parse_args()

    run_job(
        args.xyz,
        gbs_name=args.gbs_name,
        gbs_json=args.gbs_json,
        cinp=args.cinp,
        ts=args.ts,
        charge=args.charge,
        multiplicity=args.mult,
        debug=args.debug,
        freq=args.freq,
        maxsteps=args.maxsteps,
    )

if __name__ == "__main__":
    main()
