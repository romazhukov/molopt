# main.py
from pathlib import Path
import typer

from utils.log import get_logger
from io_mod.xyz_reader import read_xyz
from io_mod.result_writer import save_best_xyz, save_report
from utils.unique import dedup_conformers_by_rmsd

from conformers.rdkit_generator import RDKitConformerGenerator
from conformers.crest_generator import CRESTGenerator

from optimize.xtb_optimizer import XTBOptimizer
from optimize.pyscf_dft import PySCFR2SCAN3c  # наш DFT оптимизатор (r2scan3c)

app = typer.Typer(add_completion=False)
log = get_logger()


@app.command()
def run(
    xyz: Path = typer.Argument(..., help="входной .xyz"),
    charge: int = typer.Option(0, "--charge", "-q"),
    multiplicity: int = typer.Option(1, "--multiplicity", "-m"),

    # генерация
    nconfs: int = typer.Option(0, "--nconfs", "-n", help="если >0 — фиксированное число конформеров (RDKit)"),
    prune_rms: float = typer.Option(0.15, "--prune-rms", help="RMSD отсев дубликатов (Å)"),
    batch: int = typer.Option(100, "--batch", help="размер партии для RDKit полного перебора"),
    max_rounds: int = typer.Option(50, "--max-rounds", help="макс. раундов генерации RDKit"),

    # пайплайн/логи
    outdir: Path = typer.Option(Path("results"), "--out", "-o"),
    debug: bool = typer.Option(False, "--debug"),

    # выбор движка конформеров и уровня оптимизации
    engine: str = typer.Option("rdkit", "--engine", help="rdkit | crest"),
    crest_level: str = typer.Option("gfn2", "--crest-level", help="уровень теории для CREST (gfn1/gfn2/...)"),
    opt_level: str = typer.Option("xtb", "--opt-level", help="xtb | r2scan3c"),
    solvent: str = typer.Option(None, "--solvent", help="растворитель для PCM (PySCF), например: toluene")
):
    """
    Конвейер: генерация конформеров (RDKit/CREST) → оптимизация (XTB или DFT r2scan3c) → выбор минимума.
    """
    # ────────────────────────────── подготовка ──────────────────────────────
    outdir.mkdir(parents=True, exist_ok=True)
    atoms, coords0 = read_xyz(xyz)
    log.info(f"вход: {xyz.name}, atoms={len(atoms)}, charge={charge}, mult={multiplicity}")
    log.info(f"engine={engine}, opt_level={opt_level}, solvent={solvent or '-'}")

    engine = engine.lower().strip()
    opt_level = opt_level.lower().strip()

    if engine not in {"rdkit", "crest"}:
        raise typer.BadParameter("engine должен быть rdkit или crest")
    if opt_level not in {"xtb", "r2scan3c"}:
        raise typer.BadParameter("opt-level должен быть xtb или r2scan3c")

    # ───────────────────────── генерация конформеров ────────────────────────
    if engine == "crest":
        gen = CRESTGenerator(level=crest_level, debug=debug)
        # CREST сам решает, сколько конформеров оставить
        conformers = gen.generate(
            atoms, coords0,
            charge=charge, multiplicity=multiplicity,
            workdir=str(outdir), tag="crest"
        )
    else:
        gen = RDKitConformerGenerator(forcefield="MMFF", seed=7)
        if nconfs and nconfs > 0:
            conformers = gen.generate_fixed(
                atoms, coords0,
                nconfs=nconfs, prune_rms=prune_rms, charge=charge
            )
        else:
            conformers = gen.generate_all(
                atoms, coords0,
                prune_rms=prune_rms, batch=batch, max_rounds=max_rounds, charge=charge
            )

    log.info(f"получено конформеров: {len(conformers)}")

    # удаляем дубликаты по RMSD (на всякий случай и для RDKit, и для CREST)
    idxs_unique = dedup_conformers_by_rmsd(conformers, rmsd_thresh=0.1)
    conformers = [conformers[i] for i in idxs_unique]
    log.info(f"после фильтра RMSD уникальных конформеров: {len(conformers)}")

    if not conformers:
        log.error("Конформеров не получено.")
        raise typer.Exit(code=1)

    # ───────────────────────────── оптимизатор ──────────────────────────────
    if opt_level == "xtb":
        optimizer = XTBOptimizer(level="gfn2", opt="tight", debug=debug)
    else:
        # DFT r2scan3c через PySCF, опционально в растворителе
        optimizer = PySCFR2SCAN3c(solvent=solvent, debug=debug)

    # ───────────────────────────── оптимизация ──────────────────────────────
    results = []
    for i, (a, c) in enumerate(conformers, start=1):
        log.info(f"[{i}/{len(conformers)}] оптимизация {opt_level.upper()}…")
        a_opt, c_opt, energy, meta = optimizer.optimize(
            a, c,
            charge=charge, multiplicity=multiplicity,
            workdir=outdir, index=i
        )
        if energy is None:
            log.error(f"[{i}] оптимизатор не вернул энергию. Папка: {meta.get('workdir')}")
        else:
            unit = "Eh" if opt_level == "xtb" else "Eh"  # сохраняем в Eh; конверсию делаем в оптимизаторе при нужде
            log.info(f"[{i}] энергия: {energy:.6f} {unit}")
        results.append({"index": i, "atoms": a_opt, "coords": c_opt, "energy": energy, "meta": meta})

    # ───────────────────────── выбор минимума и отчёт ───────────────────────
    valid = [r for r in results if r["energy"] is not None]
    if not valid:
        log.error("Нет валидных результатов с энергией — всё упало")
        raise typer.Exit(code=1)

    best = min(valid, key=lambda x: x["energy"])
    best_energy = best["energy"]
    log.info(f"Минимальная энергия: {best_energy:.6f} Eh")

    # сохраним xyz и JSON-отчёт
    best_xyz = save_best_xyz(outdir, best["atoms"], best["coords"], tag="best")

    save_report(outdir, {
        "engine": engine,
        "crest_level": crest_level if engine == "crest" else None,
        "opt_level": opt_level,
        "solvent": solvent,
        "best_energy_Eh": best_energy,
        "n_conformers": len(conformers),
        "inputs": {"xyz": str(xyz), "charge": charge, "multiplicity": multiplicity},
        "optimized": [
            {"index": r["index"], "energy": r["energy"], "workdir": (r["meta"] or {}).get("workdir")}
            for r in results
        ]
    })

    log.info(
        f"Готово! best: {best_xyz} | Ebest: {best_energy:.6f} Eh | отчёт: {outdir/'report.json'}"
    )


if __name__ == "__main__":
    app()
