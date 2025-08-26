# io_mod/xyz_reader.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Iterable

Atom = str
Coord = Tuple[float, float, float]
Frame = Tuple[List[Atom], List[Coord]]


def _parse_xyz_stream(lines: Iterable[str]) -> List[Frame]:
    """
    Разбор .xyz, поддерживает мульти‑XYZ (несколько фреймов подряд).
    Возвращает список фреймов [(atoms, coords)].
    """
    frames: List[Frame] = []
    it = iter(lines)

    while True:
        try:
            header = next(it).strip()
        except StopIteration:
            break

        if not header:
            # пустые строки между блоками
            continue

        # первая строка: число атомов
        try:
            n = int(header)
        except ValueError:
            # если файл не начинается с числа — формат не XYZ
            raise ValueError(f"XYZ parse error: expected atom count, got: {header!r}")

        # вторая строка — комментарий (пропускаем/читаем)
        try:
            _comment = next(it)
        except StopIteration:
            raise ValueError("XYZ parse error: unexpected EOF after atom count")

        atoms: List[Atom] = []
        coords: List[Coord] = []

        for _ in range(n):
            try:
                line = next(it)
            except StopIteration:
                raise ValueError("XYZ parse error: unexpected EOF inside atom block")

            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"XYZ parse error: bad atom line: {line!r}")

            atoms.append(parts[0])
            x, y, z = map(float, parts[1:4])
            coords.append((x, y, z))

        frames.append((atoms, coords))

    return frames


def read_xyz(path: Path | str) -> Frame:
    """
    Прочитать ОДИН фрейм из .xyz (если во входе несколько, берём первый).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        frames = _parse_xyz_stream(f)
    if not frames:
        raise ValueError(f"Файл пуст или не распознан как XYZ: {path}")
    return frames[0]


def read_xyz_many(path: Path | str) -> List[Frame]:
    """
    Прочитать ВСЕ фреймы из мульти‑XYZ.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        frames = _parse_xyz_stream(f)
    return frames


def write_xyz(path: Path | str, atoms: List[Atom], coords: List[Coord], comment: str = "") -> Path:
    """
    Записать один фрейм XYZ.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write((comment or "generated") + "\n")
        for a, (x, y, z) in zip(atoms, coords):
            f.write(f"{a:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")
    return path
