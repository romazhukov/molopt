# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import subprocess
# import argparse
# import os
# import json


# def threads_for(task, total_cores: int, nprocs: int):
#     if task["opt_level"] == "r2scan3c":
#         return max(1, total_cores // nprocs)
#     return 1  # XTB всегда по 1 ядру


# def worker(task: dict, outdir: Path, total_cores: int, nprocs: int):
#     threads = threads_for(task, total_cores, nprocs)

#     env = os.environ.copy()
#     env["OMP_NUM_THREADS"] = str(threads)

#     cmd = [
#         "python", "molopt/run.py",
#         str(task["file"]),
#         "--engine", task["engine"],
#         "--opt-level", task["opt_level"],
#         "--charge", str(task["charge"]),
#         "--multiplicity", str(task["multiplicity"]),
#         "--out", str(outdir / Path(task["file"]).stem)
#     ]

#     subprocess.run(cmd, check=True, env=env)
#     return Path(task["file"]).name, threads


# def load_tasks(taskfile: Path):
#     with open(taskfile, "r", encoding="utf-8") as f:
#         return json.load(f)


# def main():
#     parser = argparse.ArgumentParser(description="Кастомный параллельный запуск molopt")
#     parser.add_argument("--tasks", type=str, help="JSON файл с задачами")
#     parser.add_argument("--outdir", type=str, default="results_parallel", help="Папка для результатов")
#     parser.add_argument("--nprocs", type=int, default=4, help="Число параллельных процессов")

#     args = parser.parse_args()
#     outdir = Path(args.outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     if args.tasks:
#         tasks = load_tasks(Path(args.tasks))
#     else:
#         tasks = [
#             {"file": "molecules/ethanol.xyz", "engine": "crest", "opt_level": "xtb", "charge": 0, "multiplicity": 1},
#             {"file": "molecules/butane.xyz",  "engine": "rdkit", "opt_level": "r2scan3c", "charge": 0, "multiplicity": 1},
#             {"file": "molecules/h2o.xyz",     "engine": "crest", "opt_level": "xtb", "charge": -1, "multiplicity": 2},
#         ]

#     total_cores = os.cpu_count() or 1
#     print(f"💻 Найдено {total_cores} CPU-ядер")
#     print(f"▶️ Загружено {len(tasks)} задач, max {args.nprocs} процессов")

#     with ProcessPoolExecutor(max_workers=args.nprocs) as pool:
#         futures = [pool.submit(worker, t, outdir, total_cores, args.nprocs) for t in tasks]

#         for future in as_completed(futures):
#             try:
#                 name, threads = future.result()
#                 print(f"✅ {name} готов (OMP_NUM_THREADS={threads})")
#             except Exception as e:
#                 print(f"❌ Ошибка: {e}")


# if __name__ == "__main__":
#     main()
