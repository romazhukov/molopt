from pathlib import Path
import subprocess, shutil, os
from io_mod.xyz_reader import write_xyz
from energy.xtb_parser import parse_total_energy

class XTBOptimizer:
    def __init__(self, level="gfn2", opt="tight", debug=False, nthreads: int = 1):
        self.level = level
        self.opt = opt
        self.debug = debug
        self.nthreads = nthreads

    def _check_xtb(self):
        if not shutil.which("xtb"):
            raise RuntimeError("xtb не найден в PATH (проверь установку и PATH)")

    def optimize(self, atoms, coords, charge=0, multiplicity=1, workdir=".", index=1,
                 constraints: Path | None = None, freq: bool = False):
        """
        Оптимизация с помощью xTB:
        - atoms, coords → conf.xyz
        - запускаем xtb
        - если constraints != None → передаём в --input
        - если freq=True → делаем частотный анализ
        """
        self._check_xtb()
        uhf = max(0, int(multiplicity) - 1)
        wd = Path(workdir) / f"xtb_{index:03d}"
        wd.mkdir(parents=True, exist_ok=True)

        inp = wd / "conf.xyz"
        write_xyz(inp, atoms, coords, comment=f"conf {index}")

        cmd = ["xtb", "conf.xyz", "--opt", self.opt, "--chrg", str(charge), "--uhf", str(uhf)]
        if self.level:
            cmd.append(f"--{self.level}")
        if constraints:
            cmd += ["--input", str(constraints)]
        if freq:
            cmd.append("--hess")  # XTB умеет частоты через гессиан

        (wd / "cmd.txt").write_text(" ".join(cmd))

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.nthreads)
        res = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, env=env)
        (wd / "stdout.txt").write_text(res.stdout or "")
        (wd / "stderr.txt").write_text(res.stderr or "")

        energy = None
        log = wd / "xtbopt.log"
        if log.exists():
            energy = parse_total_energy(log.read_text())
        if energy is None:
            energy = parse_total_energy(res.stdout + "\n" + res.stderr)

        atoms_opt, coords_opt = atoms, coords
        opt_xyz = wd / "xtbopt.xyz"
        if opt_xyz.exists():
            lines = opt_xyz.read_text().splitlines()
            try:
                n = int(lines[0].strip())
                A, C = [], []
                for line in lines[2:2+n]:
                    s = line.split()
                    A.append(s[0]); C.append((float(s[1]), float(s[2]), float(s[3])))
                atoms_opt, coords_opt = A, C
            except Exception:
                pass

        meta = {
            "status": "ok" if energy is not None else "failed",
            "workdir": str(wd),
            "returncode": res.returncode,
            "cmd": " ".join(cmd),
            "threads": self.nthreads,
        }
        return atoms_opt, coords_opt, energy, meta
