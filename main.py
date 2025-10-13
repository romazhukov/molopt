from pathlib import Path
import typer

from utils.log import get_logger
from io_mod.xyz_reader import read_xyz
from io_mod.result_writer import save_best_xyz, save_report
from utils.unique import dedup_conformers_by_rmsd
from conformers.crest_generator import CRESTGenerator

from optimize.xtb_optimizer import XTBOptimizer
from optimize.pyscf_dft import PySCFR2SCAN3c

app = typer.Typer(add_completion=False)
log = get_logger()

@app.command(help="Пайплайн: Intermediate (CREST→XTB→DFT) или TS (XTB→DFT с констрейнами)")
def run(
    xyz: Path = typer.Argument(..., help="Входной .xyz файл"),
    task: str = typer.Option("intermediate", "--task", "-T",
                             help="Тип задачи: intermediate | ts"),
    charge: int = typer.Option(0, "--charge", "-q"),
    multiplicity: int = typer.Option(1, "--multiplicity", "-m"),

    # генерация (только для intermediate)
    nconfs: int = typer.Option(0, "--nconfs", "-n"),
    prune_rms: float = typer.Option(0.15, "--prune-rms"),
    batch: int = typer.Option(100, "--batch"),
    max_rounds: int = typer.Option(50, "--max-rounds"),

    # общие опции
    outdir: Path = typer.Option(Path("results"), "--out", "-o"),
    debug: bool = typer.Option(False, "--debug"),

    crest_level: str = typer.Option("gfn2", "--crest-level"),
    opt_level: str = typer.Option("xtb", "--opt-level", help="xtb | r2scan3c | both"),

    solvent: str = typer.Option(None, "--solvent"),
    threads: int = typer.Option(1, "--threads", "-t"),

    # TS
    constraints: Path = typer.Option(None, "--constraints", "-C",
                                     help="Файл с констрейнами для предоптимизации (TS)"),
    ts_solver: str = typer.Option("geometric", "--ts-solver",
                                  help="geometric | qsd | both"),

    # частоты
    freq: bool = typer.Option(False, "--freq", help="Частотный анализ")
):
    outdir.mkdir(parents=True, exist_ok=True)
    atoms, coords0 = read_xyz(xyz)
    log.info(f"Вход: {xyz.name}, atoms={len(atoms)}, charge={charge}, mult={multiplicity}")
    log.info(f"task={task}, opt_level={opt_level}, solvent={solvent or '-'}, threads={threads}")

    task = task.lower().strip()
    opt_level = opt_level.lower().strip()
    if opt_level not in {"xtb","r2scan3c","both"}:
        raise typer.BadParameter("opt-level: xtb | r2scan3c | both")
    if task not in {"intermediate","ts"}:
        raise typer.BadParameter("task: intermediate | ts")

    results = []

    # ───────────── INTERMEDIATE ─────────────
    if task == "intermediate":
        # CREST генерация
        gen = CRESTGenerator(level=crest_level, debug=debug)
        conformers = gen.generate(
            atoms, coords0,
            charge=charge, multiplicity=multiplicity,
            workdir=str(outdir), tag="crest"
        )
        log.info(f"получено конформеров: {len(conformers)}")
        idxs = dedup_conformers_by_rmsd(conformers, rmsd_thresh=prune_rms)
        conformers = [conformers[i] for i in idxs]
        log.info(f"после RMSD фильтра: {len(conformers)}")

        if not conformers:
            raise typer.Exit(code=1)

        # Оптимизация (XTB → DFT)
        if opt_level == "xtb":
            optimizer = XTBOptimizer(level="gfn2", opt="tight", debug=debug, nthreads=threads)
        elif opt_level == "r2scan3c":
            optimizer = PySCFR2SCAN3c(solvent=solvent, debug=debug, nthreads=threads,
                                      transition=False, compute_freqs=freq)
        else:  # both
            optimizer = (
                XTBOptimizer(level="gfn2", opt="tight", debug=debug, nthreads=max(1, threads // 2)),
                PySCFR2SCAN3c(solvent=solvent, debug=debug, nthreads=threads,
                              transition=False, compute_freqs=freq)
            )

        for i, (a, c) in enumerate(conformers, start=1):
            log.info(f"[{i}/{len(conformers)}] оптимизация {opt_level.upper()}…")
            if opt_level == "both":
                a_xtb, c_xtb, e_xtb, m_xtb = optimizer[0].optimize(
                    a, c, charge=charge, multiplicity=multiplicity, workdir=outdir, index=i, freq=freq
                )
                log.info(f"[{i}] XTB энергия: {e_xtb:.6f} Eh")
                a_opt, c_opt, energy, meta = optimizer[1].optimize(
                    a_xtb, c_xtb, charge=charge, multiplicity=multiplicity, workdir=outdir, index=i
                )
            else:
                a_opt, c_opt, energy, meta = optimizer.optimize(
                    a, c, charge=charge, multiplicity=multiplicity, workdir=outdir, index=i, freq=freq
                )
            results.append({"index": i, "atoms": a_opt, "coords": c_opt, "energy": energy, "meta": meta})

    # ───────────── TRANSITION STATE ─────────────
    elif task == "ts":
        log.info("TS поиск: предоптимизация XTB + DFT (geometric/QSD)")

        # Предоптимизация XTB (с констрейнами, если заданы)
        xtb = XTBOptimizer(level="gfn2", opt="tight", debug=debug, nthreads=threads)
        a_xtb, c_xtb, e_xtb, m_xtb = xtb.optimize(
            atoms, coords0, charge=charge, multiplicity=multiplicity,
            workdir=outdir, index=1, constraints=constraints, freq=freq
        )
        log.info(f"XTB предоптимизация завершена (E={e_xtb:.6f} Eh)")

        # DFT поиск TS
        dft = PySCFR2SCAN3c(solvent=solvent, debug=debug, nthreads=threads,
                            transition=True, ts_solver=ts_solver, compute_freqs=freq)
        a_opt, c_opt, energy, meta = dft.optimize(
            a_xtb, c_xtb, charge=charge, multiplicity=multiplicity, workdir=outdir, index=1
        )
        results.append({"index": "TS", "atoms": a_opt, "coords": c_opt, "energy": energy, "meta": meta})

    # ───────────── выбор минимума ─────────────
    valid = [r for r in results if r["energy"] is not None]
    if not valid:
        raise typer.Exit(code=1)
    best = min(valid, key=lambda x: x["energy"])
    best_energy = best["energy"]
    log.info(f"Минимальная энергия: {best_energy:.6f} Eh")

    best_xyz = save_best_xyz(outdir, best["atoms"], best["coords"], tag="best")
    save_report(outdir, {
        "task": task,
        "opt_level": opt_level,
        "solvent": solvent,
        "threads": threads,
        "best_energy_Eh": best_energy,
        "inputs": {"xyz": str(xyz), "charge": charge, "multiplicity": multiplicity},
        "optimized": [
            {"index": r["index"], "energy": r["energy"], "workdir": (r["meta"] or {}).get("workdir")}
            for r in results
        ]
    })
    log.info(f"Готово! best: {best_xyz} | Ebest: {best_energy:.6f} Eh | отчёт: {outdir/'report.json'}")

if __name__ == "__main__":
    app()
