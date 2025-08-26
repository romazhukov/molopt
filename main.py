from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import typer

from utils.log import get_logger
from utils.unique import dedup_conformers_by_rmsd
from io_mod.xyz_reader import read_xyz
from io_mod.result_writer import save_best_xyz, save_report, save_all_xyz
from conformers.rdkit_generator import RDKitConformerGenerator
from conformers.crest_generator import CRESTGenerator
from optimize.xtb_optimizer import XTBOptimizer


Atom = str
Coord = Tuple[float, float, float]

app = typer.Typer(add_completion=False)
log = get_logger()


def _optimize_xtb_block(
    conformers: List[Tuple[List[Atom], List[Coord]]],
    charge: int,
    multiplicity: int,
    outdir: Path,
    debug: bool,
    level: str = "gfn2",
    opt: str = "tight",
):
    """
    Прогнать XTB-оптимизацию по всем конформерам и вернуть (best, results).
    """
    optm = XTBOptimizer(level=level, opt=opt, debug=debug)
    results: List[dict] = []

    for i, (a, c) in enumerate(conformers, start=1):
        log.info(f"[{i}/{len(conformers)}] XTB оптимизация…")
        a_opt, c_opt, energy, meta = optm.optimize(
            a, c,
            charge=charge,
            multiplicity=multiplicity,
            workdir=outdir,
            index=i,
        )
        if energy is None:
            log.error(f"[{i}] XTB не вернул энергию. Папка: {meta.get('workdir')}")
        else:
            log.info(f"[{i}] энергия: {energy:.6f} Eh")

        results.append(
            {"index": i, "atoms": a_opt, "coords": c_opt, "energy": energy, "meta": meta}
        )

    valid = [r for r in results if r["energy"] is not None]
    if not valid:
        log.error("Нет результатов с энергией — всё упало")
        raise typer.Exit(code=1)

    best = min(valid, key=lambda x: x["energy"])
    log.info(f"Минимальная энергия: {best['energy']:.6f} Eh")
    return best, results


@app.command()
def run(
    xyz: Path = typer.Argument(..., help="входной .xyz"),
    charge: int = typer.Option(0, "--charge", "-q"),
    multiplicity: int = typer.Option(1, "--multiplicity", "-m"),
    # RDKit параметры
    nconfs: int = typer.Option(0, "--nconfs", "-n"),
    prune_rms: float = typer.Option(0.15, "--prune-rms"),
    batch: int = typer.Option(100, "--batch"),
    max_rounds: int = typer.Option(50, "--max-rounds"),
    # движок
    engine: str = typer.Option("rdkit", "--engine", help="rdkit | crest"),
    # CREST параметры
    crest_level: str = typer.Option("gfn2", "--crest-level"),
    crest_ewin: float = typer.Option(6.0, "--crest-ewin"),
    # реоптимизация
    reopt_xtb: bool = typer.Option(True, "--reopt-xtb"),
    # прочее
    outdir: Path = typer.Option(Path("results"), "--out", "-o"),
    debug: bool = typer.Option(False, "--debug"),
):
    """
    Конформер-поиск (RDKit или CREST) → фильтр RMSD → (опционально) xTB-оптимизация → выбор best.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    atoms, coords0 = read_xyz(xyz)
    log.info(f"вход: {xyz.name}, atoms={len(atoms)}, charge={charge}, mult={multiplicity}")
    log.info(f"engine={engine}, reopt_xtb={reopt_xtb}")

    # --- генерация ---
    if engine.lower() == "rdkit":
        gen = RDKitConformerGenerator(forcefield="MMFF", seed=7)
        if nconfs and nconfs > 0:
            conformers = gen.generate_fixed(atoms, coords0, nconfs=nconfs, prune_rms=prune_rms, charge=charge)
        else:
            conformers = gen.generate_all(atoms, coords0, prune_rms=prune_rms, batch=batch, max_rounds=max_rounds, charge=charge)
    elif engine.lower() == "crest":
        gen = CRESTGenerator(level=crest_level, debug=debug)
        conformers = gen.generate(atoms, coords0, charge=charge, multiplicity=multiplicity, workdir=str(outdir), tag="crest", ewin_kcal=crest_ewin)
    else:
        log.error(f"Неизвестный engine: {engine}")
        raise typer.Exit(code=2)

    if not conformers:
        log.error("Генерация не дала ни одного конформера.")
        raise typer.Exit(code=3)

    log.info(f"получено конформеров (до RMSD-фильтра): {len(conformers)}")

    # --- RMSD фильтр ---
    idxs_unique = dedup_conformers_by_rmsd(conformers, rmsd_thresh=prune_rms)
    conformers = [conformers[i] for i in idxs_unique]
    log.info(f"после фильтра RMSD осталось: {len(conformers)}")

    # сохраняем ансамбль
    all_xyz = save_all_xyz(outdir, conformers, tag="unique")
    log.info(f"Сохранил ансамбль уникальных конформеров в {all_xyz}")

    # --- XTB оптимизация ---
    if reopt_xtb:
        best, results = _optimize_xtb_block(conformers, charge, multiplicity, outdir, debug, level="gfn2", opt="tight")
        best_atoms = best["atoms"]
        best_coords = best["coords"]
        best_energy = best["energy"]
    else:
        log.warning("reopt_xtb=False → возвращаю первый конформер без оптимизации")
        best_atoms, best_coords = conformers[0]
        best_energy = None
        results = [{"index": i + 1, "atoms": a, "coords": c, "energy": None, "meta": {"note": "no_xtb"}} for i, (a, c) in enumerate(conformers)]

    # --- вывод ---
    best_xyz = save_best_xyz(outdir, best_atoms, best_coords, tag="best")
    save_report(
        outdir,
        {
            "engine": engine,
            "best_energy_Eh": best_energy,
            "n_conformers_unique": len(conformers),
            "xtb_used": bool(reopt_xtb),
            "inputs": {"xyz": str(xyz), "charge": charge, "multiplicity": multiplicity},
            "optimized": [{"index": r["index"], "energy": r["energy"], "workdir": (r["meta"] or {}).get("workdir")} for r in results],
        },
    )

    if best_energy is not None:
        log.info(f"Готово! best: {best_xyz} | Ebest: {best_energy:.6f} Eh | отчёт: {outdir/'report.json'}")
    else:
        log.info(f"Готово! best (без XTB): {best_xyz} | отчёт: {outdir/'report.json'}")


if __name__ == "__main__":
    app()
