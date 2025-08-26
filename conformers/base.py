from abc import ABC, abstractmethod

class ConformerGenerator(ABC):
    @abstractmethod
    def generate_fixed(self, atoms, coords, n: int, prune_rms: float = 0.15): ...
    @abstractmethod
    def generate_all(self, atoms, coords, prune_rms: float = 0.15, batch: int = 100, max_rounds: int = 50): ...
