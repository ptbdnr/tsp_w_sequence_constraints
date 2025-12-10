
from datastore.distance_manager import EuclidianDistanceManager
from datastore.edge_manager import EdgeManager
from datastore.node_manager import NodeManager
from eval.route_eval import RouteEvaluator
from optimiser.iterative.operations.operation import Operation
from optimiser.iterative.operations.relocate import Relocate
from optimiser.iterative.operations.three_opt_swap import ThreeOptSwap
from optimiser.iterative.operations.two_opt_swap import TwoOptSwap
from optimiser.iterative.termination import Termination
from schemas.route import Route
from utils.logger import Logger
from optimiser.iterative.callback import Callback

RANDOM_SEED_VALUE = 42


class LocalSearchImprover:
    """Class for iterative local search algorithm."""

    node_manager: NodeManager
    edge_manager: EdgeManager
    distance_manager: EuclidianDistanceManager
    callback: Callback

    seed_routes: list[Route]

    operations: list[Operation]
    termination: Termination
    route_eval: RouteEvaluator

    logger: Logger

    def __init__(
            self,
            logger: Logger,
            node_manager: NodeManager,
            edge_manager: EdgeManager,
            distance_manager: EuclidianDistanceManager,
            termination: Termination,
            callback: Callback,
        ) -> None:
        """Initialise the instance."""
        self.logger = logger
        self.node_manager = node_manager
        self.edge_manager = edge_manager
        self.distance_manager = distance_manager
        self.route_eval = RouteEvaluator(
            logger=logger,
            node_manager=self.node_manager,
            edge_manager=self.edge_manager,
            distance_manager=distance_manager,
        )
        self.seed_routes = []
        self.operations = [
            TwoOptSwap(route_eval=self.route_eval, logger=logger, rnd_seed=RANDOM_SEED_VALUE),
            ThreeOptSwap(route_eval=self.route_eval, logger=logger, rnd_seed=RANDOM_SEED_VALUE),
            Relocate(route_eval=self.route_eval, logger=logger, rnd_seed=RANDOM_SEED_VALUE),
        ]
        self.termination = termination
        self.callback = callback

    def add_seed_route(self, route: Route) -> None:
        """Add a seed route for iterative optimisation."""
        self.seed_routes.append(route)

    def optimise(self) -> list[Route]:
        """Perform naive iterative optimisation to create a route."""
        if not self.seed_routes:
            self.logger.warning("No seed routes available for optimisation.")
            return Route(sequence=[])

        # Naive iterative improvement: return the best seed route
        best_routes = []
        best_route_value = float("inf")
        for route in self.seed_routes:
            self.logger.debug(f"Seed route with {len(route.sequence)} nodes.")
            route_value = self.route_eval.calculate_objective_value(route=route)
            if route_value < best_route_value:
                best_route_value = route_value
                best_routes = [route]

        iteration_count = 0
        for route in best_routes:
            route_value = self.route_eval.calculate_objective_value(route=route)
            while not self.termination.should_terminate(iteration_count=iteration_count):
                operation = self.operations[iteration_count % len(self.operations)]
                new_route = operation.apply_first_improvement(route=route)
                new_route_value = self.route_eval.calculate_objective_value(route=new_route)
                self.callback.on_iteration(
                    iteration=iteration_count,
                    current_value=new_route_value,
                    best_value=route_value,
                    improved=new_route_value < route_value,
                )
                if new_route_value < route_value:
                    self.logger.debug(
                        f"Improved route found with objective value {new_route_value} "
                        f"using operation {operation.__class__.__name__}.",
                    )
                    route_value = new_route_value
                    best_routes = [new_route]
                    self.callback.save_route(
                        iteration=iteration_count,
                        route=new_route,
                    )
                iteration_count += 1

        self.logger.debug(f"Selected best route with value: {route_value}.")
        return [best_routes[0]]
