import json
from pathlib import Path

from schemas.route import Route


class Callback:
    """Base class for optimiser callbacks."""

    iterations: list[dict[str, float]]
    routes: dict[int, Route]

    def __init__(self) -> None:
        """Initialise the callback instance."""
        self.iterations = []
        self.routes = {}

    def on_iteration(
        self,
        *,
        iteration: int,
        current_value: float,
        best_value: float,
        improved: bool,
    ) -> None:
        """Call at the end of each iteration of the optimiser.

        Args:
            iteration: The current iteration number.
            current_value: The value of the current solution.
            best_value: The value of the best solution found so far.
            improved: Whether the best solution was improved in this iteration.

        """
        self.iterations.append({
            "iteration": iteration,
            "current_value": current_value,
            "best_value": best_value,
            "improved": float(improved),
        })

    def save_route(
        self, *,
        iteration: int,
        route: Route,
    ) -> None:
        """Save the route at a given iteration.

        Args:
            iteration: The iteration number.
            route: The route to save.

        """
        self.routes[iteration] = route

    def iterations_to_file(self, *, filepath: Path) -> None:
        """Export the callback records to a JSON file.

        Args:
            filepath: The path to the output JSON file.

        """

        with Path(filepath).open("w", encoding="utf-8") as f:
            json.dump(self.iterations, f, indent=4)

    def routes_to_file(self, *, filepath: Path) -> None:
        """Export the saved routes to a JSON file.

        Args:
            filepath: The path to the output JSON file.

        """

        route_sequences = {
            iteration: [node.id for node in route.sequence]
            for iteration, route in self.routes.items()
        }

        with Path(filepath).open("w", encoding="utf-8") as f:
            json.dump(route_sequences, f, indent=4)
