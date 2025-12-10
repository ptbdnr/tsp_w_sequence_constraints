from abc import ABC, abstractmethod

from schemas.route import Route


class IterativeOptimiser(ABC):
    """Abstract base class for iterative optimisers."""

    @abstractmethod
    def add_seed_route(self, route: Route) -> None:
        """Add a seed route for iterative optimisation."""
        raise NotImplementedError

    @abstractmethod
    def optimise(self) -> list[Route]:
        """Perform iterative optimisation to create a route."""
        raise NotImplementedError
