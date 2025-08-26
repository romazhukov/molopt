# utils/unique.py
from __future__ import annotations
from typing import List, Tuple, Dict
import math
import numpy as np

CoordList = List[Tuple[float, float, float]]

def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    RMSD с удалением переноса и поворота (Kabsch).
    P, Q: [N,3] (одинаковый порядок атомов!)
    """
    # центрируем
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    # вычисляем оптимальный поворот
    C = np.dot(Pc.T, Qc)
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        V[:, -1] *= -1.0

    U = np.dot(V, Wt)
    P_rot = np.dot(Pc, U)

    diff = P_rot - Qc
    rmsd = math.sqrt((diff * diff).sum() / P.shape[0])
    return float(rmsd)


def dedup_conformers_by_rmsd(
    conformers: List[Tuple[List[str], CoordList]],
    rmsd_thresh: float = 0.1,
) -> List[int]:
    """
    Принимает список конформеров (atoms, coords).
    Возвращает индексы уникальных конформеров по порогу RMSD (Å).
    """
    keep: List[int] = []
    coords_np = [np.array(c[1], dtype=float) for c in conformers]

    for i in range(len(conformers)):
        is_dup = False
        for j in keep:
            rmsd = _kabsch_rmsd(coords_np[i], coords_np[j])
            if rmsd < rmsd_thresh:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
    return keep


def dedup_results_by_rmsd_energy(
    results: List[Dict],
    rmsd_thresh: float = 0.1,
    dE_thresh_Eh: float = 1e-5,  # ~0.006 ккал/моль
) -> List[int]:
    """
    results: список объектов типа
      {
        "atoms": [...],
        "coords": [...],        # до оптимизации (опционально)
        "atoms_opt": [...],
        "coords_opt": [...],
        "energy": float,        # в Eh (электрон-вольт не используем)
        ...
      }
    Возвращает индексы уникальных по энергии и RMSD (используем coords_opt).
    """
    keep: List[int] = []
    coords_np = [np.array(r.get("coords_opt", r["coords"]), dtype=float) for r in results]
    energies = [float(r["energy"]) if r.get("energy") is not None else float("inf") for r in results]

    for i in range(len(results)):
        is_dup = False
        for j in keep:
            dE = abs(energies[i] - energies[j])
            if dE <= dE_thresh_Eh:
                rmsd = _kabsch_rmsd(coords_np[i], coords_np[j])
                if rmsd <= rmsd_thresh:
                    is_dup = True
                    break
        if not is_dup:
            keep.append(i)
    return keep
