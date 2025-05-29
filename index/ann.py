from dataclasses import dataclass


@dataclass(eq=False)
class SearchNeighbor:
    id: int
    score: float

    def __eq__(self, other):
        if isinstance(other, SearchNeighbor):
            return self.id == other.id
        elif isinstance(other, int):
            return self.id == other
        else:
            raise NotImplementedError
