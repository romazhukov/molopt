# conformers/rdkit_generator.py
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
import tempfile, subprocess, os, shutil


def _mol_from_xyz_rdkit(atoms, coords, charge=0):
    """Fallback: восстановление связей только средствами RDKit."""
    xyz_block = f"{len(atoms)}\nxyz via molopt\n" + "\n".join(
        f"{a} {x:.8f} {y:.8f} {z:.8f}" for a, (x, y, z) in zip(atoms, coords)
    )
    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise RuntimeError("RDKit: не смог прочитать XYZ-блок.")
    # Восстановим связи по координатам и заряду
    rdDetermineBonds.DetermineBonds(mol, charge=int(charge))
    Chem.SanitizeMol(mol)
    return mol


def _mol_from_xyz_with_obabel(atoms, coords, charge=0):
    """
    Основной путь: XYZ -> OpenBabel (восстановит связи) -> MOL -> RDKit Mol.
    Если obabel недоступен, бросаем исключение, чтобы сработал RDKit fallback.
    """
    if not shutil.which("obabel"):
        raise FileNotFoundError("obabel not found")

    xyz_text = f"{len(atoms)}\nxyz via molopt\n" + "\n".join(
        f"{a} {x:.8f} {y:.8f} {z:.8f}" for a, (x, y, z) in zip(atoms, coords)
    )

    with tempfile.TemporaryDirectory() as tmp:
        xyz_file = os.path.join(tmp, "mol.xyz")
        mol_file = os.path.join(tmp, "mol.mol")
        with open(xyz_file, "w") as f:
            f.write(xyz_text)

        res = subprocess.run(
            ["obabel", "-ixyz", xyz_file, "-omol", "-O", mol_file, "--addh"],
            capture_output=True, text=True
        )
        if res.returncode != 0 or not os.path.exists(mol_file):
            raise RuntimeError(
                "OpenBabel не смог сконвертировать XYZ в MOL.\n"
                f"rc={res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
            )

        rdmol = Chem.MolFromMolFile(mol_file, sanitize=True, removeHs=False)

    if rdmol is None:
        raise RuntimeError("RDKit: не смог прочитать MOL после obabel.")
    Chem.SanitizeMol(rdmol)
    return rdmol


def _build_mol_robust(atoms, coords, charge=0):
    """Пробуем через obabel, если его нет/ошибка — используем RDKit fallback."""
    try:
        return _mol_from_xyz_with_obabel(atoms, coords, charge=charge)
    except Exception:
        return _mol_from_xyz_rdkit(atoms, coords, charge=charge)


class RDKitConformerGenerator:
    """
    Генератор конформеров RDKit: сначала восстанавливаем связи из XYZ
    (OpenBabel → RDKit), при недоступности obabel — fallback на RDKit.
    """
    def __init__(self, forcefield: str = "MMFF", seed: int = 7):
        self.ff = forcefield.upper()
        self.seed = seed

    def _post_opt(self, mol):
        try:
            if self.ff == "MMFF" and AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
            else:
                AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0)
        except Exception:
            pass

    def _dump(self, mol):
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        out = []
        for conf in mol.GetConformers():
            pos = conf.GetPositions()
            coords = [(float(x), float(y), float(z)) for x, y, z in pos]
            out.append((symbols, coords))
        return out

    def _embed(self, mol, num, prune_rms, seed):
        p = AllChem.ETKDGv3()
        p.randomSeed = int(seed)
        p.pruneRmsThresh = float(prune_rms)
        AllChem.EmbedMultipleConfs(mol, numConfs=int(num), params=p)

    def generate_fixed(self, atoms, coords, n: int, prune_rms: float = 0.5, charge: int = 0):
        mol = _build_mol_robust(atoms, coords, charge=charge)
        self._embed(mol, n, prune_rms, self.seed)
        self._post_opt(mol)
        return self._dump(mol)

    def generate_all(
        self,
        atoms,
        coords,
        prune_rms: float = 0.5,
        batch: int = 100,
        max_rounds: int = 50,
        charge: int = 0,
    ):
        mol = _build_mol_robust(atoms, coords, charge=charge)
        prev = -1
        rounds = 0
        while mol.GetNumConformers() > prev and rounds < max_rounds:
            prev = mol.GetNumConformers()
            self._embed(mol, batch, prune_rms, seed=17 + rounds)
            rounds += 1
            if mol.GetNumConformers() == prev:
                break
        self._post_opt(mol)
        return self._dump(mol)
