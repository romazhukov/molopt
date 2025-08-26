from abc import ABC, abstractmethod

class GeometryOptimizer(ABC):
    @abstractmethod
    def optimize(self, atoms, coords, charge: int = 0, multiplicity: int = 1, workdir: str = "."):
        """Возвращает (atoms, coords_opt, energy_hartree, metadata)"""
        ...
