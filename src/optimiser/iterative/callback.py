import json
from pathlib import Path

import matplotlib.pyplot as plt

from schemas.route import Route


class Callback:
    """Base class for optimiser callbacks."""

    iterations: list[dict[str, float]]
    routes: dict[int, Route]

    def __init__(self) -> None:
        """Initialise the callback instance."""
        self.iterations = []
        self.routes = {}

    def load_alns_result_statistics(
        self,
        *,
        statistics: dict[str, list[float]],
    ) -> None:
        """Load ALNS result statistics into the callback.

        Args:
            statistics: A dictionary containing lists of statistics
                such as objectives and runtimes.

        """
        best_value = statistics.objectives[0]
        for i, curr_value in enumerate(statistics.objectives):
            self.iterations.append({
                "iteration": i,
                "current_value": float(curr_value),
                "best_value": float(best_value),
                "improved": float(curr_value < best_value),
            })
            best_value = min(best_value, curr_value)
        for i, val in enumerate(statistics.runtimes):
            if i < len(self.iterations):
                self.iterations[i]["runtime"] = float(val)

    def on_iteration(
        self,
        *,
        iteration: int,
        current_value: float,
        best_value: float,
        improved: bool,
        runtime: float | None = None,
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
            "runtime": runtime,
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

    def plot_iterations(
        self,
        filepath: Path,
        title: str | None = None,
    ) -> None:
        """Plot the optimisation iterations to a file.

        Args:
            title: The title of the plot.
            filepath: The path to the output image file.

        """
        iterations = [record["iteration"] for record in self.iterations]
        current_values = [record["current_value"] for record in self.iterations]
        best_values = [record["best_value"] for record in self.iterations]

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, current_values, label="Current Value", color="blue")
        plt.plot(iterations, best_values, label="Best Value", color="orange")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.title(title or "Optimisation Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(filepath)
        plt.close()
