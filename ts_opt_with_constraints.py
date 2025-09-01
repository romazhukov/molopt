#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS оптимизация с констрейнтами (PySCF + geomeTRIC).
Поддержка:
- базисных сетов через JSON (--gbs),
- автоматическая подгрузка с Basis Set Exchange (--gbs-name),
- конфиг файла с констрейнтами (--cinp),
- поиск переходных состояний (--ts).
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from pyscf import gto, dft
import basis_set_exchange as bse


# ============ Утилиты =====================

def read_xyz(path: str) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    atoms, coords = [], []
    with open(path, "r") as f:
        lines = f.readlines()[2:]  # пропускаем первые две строки
    for line in lines:
        parts = line.split()
        atoms.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms, coords


def load_pyscf_basis_from_json(json_path: str):
    """Загрузить PySCF-базис напрямую из JSON-файла."""
    with open(json_path, "r", encoding="utf-8") as f:
        basis_dict = json.load(f)
    return basis_dict


def make_pyscf_basis_from_bse(name: str, atoms: List[str]):
    """Скачать базис с BSE и преобразовать в формат PySCF."""
    basis_dict = {}
    for symb in set(atoms):
        basis_str = bse.get_basis(name, elements=[symb], fmt="nwchem")
        basis_dict[symb] = gto.basis.parse(basis_str)
    return basis_dict


def parse_constraints(cinp_path: str):
    """Простейший парсер input.inp с секциями $constrain ... $end."""
    constraints = {"distance": [], "angle": [], "dihedral": [], "metadyn": []}
    with open(cinp_path, "r") as f:
        lines = f.readlines()
    block = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("$constrain"):
            block = "constrain"
            continue
        if line.lower().startswith("$metadyn"):
            block = "metadyn"
            continue
        if line.lower().startswith("$end"):
            block = None
            continue
        if block == "constrain":
            if line.startswith("distance:"):
                _, rest = line.split(":")
                i, j, d = rest.split(",")
                constraints["distance"].append((int(i), int(j), float(d)))
            elif line.startswith("angle:"):
                _, rest = line.split(":")
                i, j, k, ang = rest.split(",")
                constraints["angle"].append((int(i), int(j), int(k), float(ang)))
            elif line.startswith("dihedral:"):
                _, rest = line.split(":")
                i, j, k, l, dih = rest.split(",")
                constraints["dihedral"].append((int(i), int(j), int(k), int(l), float(dih)))
            elif "force constant" in line:
                constraints["force_const"] = float(line.split("=")[1])
        elif block == "metadyn":
            if line.startswith("atoms:"):
                _, rest = line.split(":")
                groups = []
                for g in rest.split(","):
                    if "-" in g:
                        a, b = g.split("-")
                        groups.extend(list(range(int(a), int(b) + 1)))
                    else:
                        groups.append(int(g))
                constraints["metadyn"].append(groups)
    return constraints


# ============ Основной расчёт =====================

def build_pyscf(atoms, coords, charge, multiplicity, basis, debug=False):
    spin = multiplicity - 1
    mol = gto.M(
        atom=[[a, xyz] for a, xyz in zip(atoms, coords)],
        unit="Angstrom",
        charge=charge,
        spin=spin,
        basis=basis,
        verbose=4 if debug else 0,
    )
    if spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = "r2scan"
    mf.conv_tol = 1e-9
    return mol, mf


def optimize_with_constraints(mf, ts=False, constraints=None):
    """Запуск оптимизации через geomeTRIC с констрейнтами."""
    from pyscf.geomopt.geometric_solver import optimize as geometric_optimize
    import geometric

    # Конфиг geomeTRIC
    geom_options = {}
    if ts:
        geom_options["transition"] = True
    if constraints:
        geom_options["constraints"] = constraints

    mol_opt = geometric_optimize(mf, **geom_options)
    mf_opt = mf.__class__(mol_opt)
    _ = mf_opt.kernel()
    return mol_opt, mf_opt


def run_job(xyz, gbs_json=None, gbs_name=None, cinp=None, ts=False, debug=False):
    atoms, coords = read_xyz(xyz)

    # --- базис ---
    if gbs_json is not None:
        basis = load_pyscf_basis_from_json(gbs_json)
    elif gbs_name is not None:   # <-- исправлено
        basis = make_pyscf_basis_from_bse(gbs_name, atoms)
    else:
        basis = "def2-SVP"

    # --- констрейны ---
    constraints = parse_constraints(cinp) if cinp else None

    # --- PySCF ---
    mol, mf = build_pyscf(atoms, coords, charge=0, multiplicity=1, basis=basis, debug=debug)

    # --- геом. оптимизация ---
    mol_opt, mf_opt = optimize_with_constraints(mf, ts=ts, constraints=constraints)

    print("Энергия SCF:", mf_opt.e_tot, "Eh")
    return mol_opt, mf_opt


# ============ CLI =====================

def main():
    parser = argparse.ArgumentParser(description="TS оптимизация с констрейнтами (PySCF+geomeTRIC)")
    parser.add_argument("xyz", help="входной XYZ файл")
    parser.add_argument("--gbs", help="JSON файл с базисом")
    parser.add_argument("--gbs-name", dest="gbs_name", help="имя базиса из Basis Set Exchange (например def2-mTZVPP)")
    parser.add_argument("--cinp", help="input.inp файл с констрейнтами")
    parser.add_argument("--ts", action="store_true", help="поиск переходного состояния")
    parser.add_argument("--debug", action="store_true", help="подробный вывод")
    args = parser.parse_args()

    run_job(
        args.xyz,
        gbs_json=args.gbs,
        gbs_name=args.gbs_name,
        cinp=args.cinp,
        ts=args.ts,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
