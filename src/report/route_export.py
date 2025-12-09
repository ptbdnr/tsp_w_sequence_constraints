from pathlib import Path

from eval.route_eval import RouteEvaluator
from report.plot_builder import PlotBuilder
from schemas.node import Node
from schemas.route import Route
from utils.logger import Logger


class RouteExporter:
    """A class evaluating a route."""

    logger: Logger
    route_eval: RouteEvaluator
    plot_builder: PlotBuilder

    def __init__(
            self,
            route_eval: RouteEvaluator,
            nodes: list[Node],
            logger: Logger | None = None,
        ) -> None:
        """Initialize class."""
        self.logger = logger or Logger(__name__)
        self.route_eval = route_eval
        self.plot_builder = PlotBuilder(
            nodes=nodes,
            logger=self.logger,
        )

    def report_format(self, route: Route) -> str:
        """Get a report representation of the route.

        ```text
        Route:0-3-1-2
        Total Distance: 849.25
        Delta Value: 12.38
        ```
        """
        total_distance, distances = self.route_eval.total_distance_and_distances(route=route)
        max_distance = max(distances) if distances else 0.0
        min_distance = min(distances) if distances else 0.0
        delta = max_distance - min_distance

        route_sequence_ids = "-".join([str(node.id) for node in route.sequence])
        return (
            f"Route:{route_sequence_ids}\n"
            f"Total Distance: {total_distance:.2f}\n"
            f"Delta Value: {delta:.2f}"
        )

    def report_to_file(self, route: Route, filepath: str) -> None:
        content = self.report_format(route=route)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with Path(filepath).open("w", encoding="UTF8") as file:
            file.write(content)

    def plot_to_file(
            self,
            route: Route,
            filepath: str,
            title: str | None = None,
        ) -> None:
        return self.plot_builder.route_to_file(
            route=route,
            filepath=filepath,
            title=title,
        )