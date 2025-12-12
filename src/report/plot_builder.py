import matplotlib.pyplot as plt

from schemas.node import Node
from schemas.route import Route
from utils.logger import Logger


class IterationPlotBuilder:
    """Builder class for plotting iteration statistics."""

    logger: Logger
    figure: plt.Figure

    def __init__(
            self,
            logger: Logger | None,
        ) -> None:
        """Initialize the PlotBuilder."""
        self.logger = logger or Logger(__name__)
        self.figure = plt.figure(figsize=(10, 6))

    def iterations_to_file(
            self,
            iterations: list[dict],
            filepath: str,
            title: str | None = None,
        ) -> plt.Figure:
        """Rebuild the plot builder."""
        self._clear_plot()
        self._plot_iterations(iterations=iterations)
        self._save_plot(
            filepath=filepath,
            title=title,
        )
        return self

    def _clear_plot(self) -> None:
        """Clear the current plot."""
        plt.clf()

    def _plot_iterations(self, iterations: list[dict]) -> None:
        """Plot the iteration statistics."""
        iteration_nums = [item["iteration"] for item in iterations]
        current_values = [item["current_value"] for item in iterations]
        best_values = [item["best_value"] for item in iterations]

        plt.plot(iteration_nums, current_values, label="Current Value", color="blue")
        plt.plot(iteration_nums, best_values, label="Best Value", color="green")
        plt.xlabel("Iteration (#)")
        plt.ylabel("Objective Value")
        plt.legend()
        plt.grid()

    def _save_plot(
            self,
            filepath: str,
            title: str | None = None,
        ) -> None:
        """Save the plot to the specified filepath."""
        title = title or "Iteration Statistics"
        plt.title(title)
        plt.savefig(filepath)
        plt.close()

class RoutePlotBuilder:
    """Builder class for plotting routes."""

    logger: Logger
    figure: plt.Figure
    nodes: list[Node]

    def __init__(
            self,
            nodes: list[Node],
            logger: Logger | None,
        ) -> None:
        """Initialize the PlotBuilder."""
        self.nodes = nodes
        self.logger = logger or Logger(__name__)
        self.figure = plt.figure(figsize=(10, 6))

    def route_to_file(
            self,
            route: Route,
            filepath: str,
            title: str | None = None,
        ) -> plt.Figure:
        """Rebuild the plot builder."""
        self._clear_plot()
        self._plot_nodes()
        self._plot_route(route=route)
        self._save_plot(
            filepath=filepath,
            title=title,
        )
        return self

    def _clear_plot(self) -> None:
        """Clear the current plot."""
        plt.clf()

    def _plot_nodes(self) -> None:
        """Plot the nodes."""
        x_coords = [node.x for node in self.nodes]
        y_coords = [node.y for node in self.nodes]

        plt.scatter(x_coords, y_coords, color="blue")
        for i, node in enumerate(self.nodes):
            plt.text(node.x, node.y, str(node.id))

    def _plot_route(self, route: Route) -> None:
        """Plot the given route."""
        x_coords = [node.x for node in route.sequence]
        y_coords = [node.y for node in route.sequence]

        plt.plot(x_coords, y_coords, color="red", marker="o")

    def _save_plot(
            self,
            filepath: str,
            title: str | None = None,
        ) -> None:
        """Save the plot to the specified filepath."""
        title = title or "Route Plot"
        plt.title(title)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid()
        plt.savefig(filepath)
        plt.close()
