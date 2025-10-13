# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import json
# from pathlib import Path

# def ask(prompt, default=None, cast=str):
#     if default is not None:
#         prompt = f"{prompt} [{default}]: "
#     else:
#         prompt = f"{prompt}: "
#     val = input(prompt).strip()
#     return cast(val) if val else default

# def main():
#     tasks = []

#     print("=== Генератор tasks.json для parallel_run_custom.py ===")
#     print("Введи параметры для каждой молекулы. Оставь имя файла пустым, чтобы закончить.\n")

#     while True:
#         file = input("Файл молекулы (.xyz, пусто = закончить): ").strip()
#         if not file:
#             break

#         engine = ask("Engine (rdkit/crest)", default="rdkit")
#         opt_level = ask("Opt-level (xtb/r2scan3c)", default="xtb")
#         charge = ask("Charge", default=0, cast=int)
#         mult = ask("Multiplicity", default=1, cast=int)

#         task = {
#             "file": file,
#             "engine": engine,
#             "opt_level": opt_level,
#             "charge": charge,
#             "multiplicity": mult
#         }
#         tasks.append(task)
#         print(f"✅ Добавлена задача: {task}\n")

#     if not tasks:
#         print("⚠️ Не добавлено ни одной задачи.")
#         return

#     outfile = ask("Имя файла для сохранения", default="tasks.json")
#     Path(outfile).write_text(json.dumps(tasks, indent=2), encoding="utf-8")
#     print(f"\n💾 Сохранено {len(tasks)} задач в {outfile}")

# if __name__ == "__main__":
#     main()
