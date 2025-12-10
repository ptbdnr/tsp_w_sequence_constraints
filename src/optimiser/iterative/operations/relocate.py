from random import SystemRandom

from eval.route_eval import RouteEvaluator
from optimiser.iterative.operations.operation import Operation
from schemas.route import Route
from utils.logger import Logger

MIN_ROUTE_LENGTH = 4


class Relocate(Operation):
    """Class for relocate operation.

    The relocate operation removes a node or segment from its current position
    and inserts it at a different position in the route.

    procedure relocate(route, v1, v2, insert_pos) {
        1. Extract segment from route[v1:v2+1]
        2. Remove this segment from the route
        3. Insert the segment at insert_pos
        return new_route;
    }

    For single-node relocation (v1 == v2), this moves a single node.
    For segment relocation (v1 < v2), this moves multiple consecutive nodes.
    """

    logger: Logger
    rnd_seed: int
    rnd_generator: SystemRandom
    route_eval: RouteEvaluator

    def __init__(
            self,
            route_eval: RouteEvaluator,
            logger: Logger | None = None,
            rnd_seed: int = 42,
        ) -> None:
        """Initialise the operation."""
        self.route_eval = route_eval
        self.rnd_seed = rnd_seed
        self.rnd_generator = SystemRandom(x=self.rnd_seed)
        self.logger = logger or Logger(__name__)

    def apply(
            self,
            route: Route,
            v1: int | None = None,
            v2: int | None = None,
            insert_pos: int | None = None,
            *,
            inplace: bool = False,
        ) -> Route:
        """Apply the relocate operation to the given route.

        Args:
            route: The route to apply the operation to
            v1: Start index of segment to relocate (if None, randomly selected)
            v2: End index of segment to relocate (if None, same as v1 for single node)
            insert_pos: Position to insert the segment (if None, randomly selected)
            inplace: If True, modify the route in place; otherwise create a new route

        Returns:
            The modified route (either new or the same object if inplace=True)

        """
        route_length = len(route.sequence)

        # Need at least 4 nodes to perform relocate
        if route_length < MIN_ROUTE_LENGTH:
            self.logger.warning(f"Route too short for relocate (length={route_length})")
            return route if inplace else route.copy()

        # Generate random indices if not provided
        # We exclude the first and last nodes (depot nodes)
        if v1 is None or v2 is None:
            v1 = self.rnd_generator.randint(1, route_length - 2)
            v2 = self.rnd_generator.randint(v1, route_length - 2)
        elif v1 > v2:
            # Ensure v1 <= v2
            v1, v2 = v2, v1

        # Validate v1 and v2
        if v1 < 1 or v2 >= route_length - 1 or v1 > v2:
            self.logger.error(
                f"Invalid indices for relocate: v1={v1}, v2={v2}, "
                f"route_length={route_length}",
            )
            return route if inplace else route.copy()

        # Generate random insert position if not provided
        if insert_pos is None:
            # Can insert at any position except within the extracted segment
            valid_positions = [i for i in range(route_length - (v2 - v1 + 1))
                             if not (v1 <= i <= v2)]
            if not valid_positions:
                self.logger.warning("No valid insertion positions available")
                return route if inplace else route.copy()
            insert_pos = self.rnd_generator.choice(valid_positions)
        elif insert_pos < 0 or insert_pos >= route_length - (v2 - v1 + 1):
            # Validate insert_pos
            self.logger.error(
                f"Invalid insertion position: {insert_pos} for route_length={route_length}",
            )
            return route if inplace else route.copy()

        segment_length = v2 - v1 + 1
        self.logger.debug(
            f"Applying relocate: moving segment [{v1}:{v2}] (length={segment_length}) "
            f"to position {insert_pos}",
        )

        # Extract the segment to relocate
        segment = route.sequence[v1:v2 + 1]

        # Create new sequence without the segment
        new_sequence = route.sequence[:v1] + route.sequence[v2 + 1:]

        # Adjust insert position if needed (after removal, positions shift)
        adjusted_insert_pos = insert_pos if insert_pos < v1 else insert_pos - segment_length

        # Insert the segment at the new position
        new_sequence = (
            new_sequence[:adjusted_insert_pos] +
            segment +
            new_sequence[adjusted_insert_pos:]
        )

        # Apply the change
        if inplace:
            route.sequence = new_sequence
            self.logger.debug(
                f"Applied relocate in place: segment [{v1}:{v2}] to position {insert_pos}",
            )
            return route

        # Create a new route
        new_route = Route(name=route.name, sequence=new_sequence, logger=self.logger)
        self.logger.debug(
            f"Created new route with relocate: segment [{v1}:{v2}] to position {insert_pos}",
        )
        return new_route

    def apply_best_improvement(
            self,
            route: Route,
            *,
            only_valid: bool = True,
    ) -> Route:
        """Apply the best relocate improvement to the route.

        Tries all possible relocate moves and returns the one with the best improvement.

        Args:
            route: The route to improve
            only_valid: If True, only consider valid routes

        Returns:
            The improved route (or original if no improvement found)

        """
        best_route = route.copy()
        orig_value = best_value = self.route_eval.calculate_objective_value(route)
        improved = False

        route_length = len(route.sequence)
        evaluations = 0

        # Try all possible relocate moves
        for v1 in range(1, route_length - 2):
            for v2 in range(v1, route_length - 2):
                segment_length = v2 - v1 + 1
                # Determine valid insertion positions
                for insert_pos in range(0, route_length - segment_length):
                    # Skip positions within or adjacent to the segment
                    if v1 <= insert_pos <= v2 + 1:
                        continue

                    # Create new route with this relocate
                    new_route = self.apply(
                        route,
                        v1=v1,
                        v2=v2,
                        insert_pos=insert_pos,
                        inplace=False,
                    )
                    if only_valid and not self.route_eval.is_valid_route(route=new_route):
                        continue
                    new_value = self.route_eval.calculate_objective_value(route=new_route)
                    evaluations += 1

                    # Check if this is an improvement
                    if new_value < best_value:
                        best_route = new_route
                        best_value = new_value
                        improved = True
                        self.logger.debug(
                            f"Found improvement with relocate [{v1}:{v2}] to {insert_pos}: "
                            f"value reduced to {new_value:.2f}",
                        )

        if improved:
            self.logger.info(
                f"Best relocate improvement found after {evaluations} evaluations: "
                f"value reduced from {orig_value:.2f} to {best_value:.2f}",
            )
        else:
            self.logger.debug(
                f"No relocate improvement found after {evaluations} evaluations",
            )

        return best_route

    def apply_first_improvement(
            self,
            route: Route,
            *,
            only_valid: bool = True,
        ) -> Route:
        """Apply the first relocate improvement found.

        Stops as soon as an improvement is found (faster than best improvement).

        Args:
            route: The route to improve
            only_valid: If True, only consider valid routes

        Returns:
            The improved route (or original if no improvement found)

        """
        curr_value = self.route_eval.calculate_objective_value(route)
        route_length = len(route.sequence)
        evaluations = 0

        # Try relocate moves until we find an improvement
        for v1 in range(1, route_length - 2):
            for v2 in range(v1, route_length - 2):
                segment_length = v2 - v1 + 1
                for insert_pos in range(0, route_length - segment_length):
                    # Skip positions within or adjacent to the segment
                    if v1 <= insert_pos <= v2 + 1:
                        continue

                    # Create new route with this relocate
                    new_route = self.apply(
                        route,
                        v1=v1,
                        v2=v2,
                        insert_pos=insert_pos,
                        inplace=False,
                    )
                    if only_valid and not self.route_eval.is_valid_route(route=new_route):
                        continue
                    new_value = self.route_eval.calculate_objective_value(route=new_route)
                    evaluations += 1

                    # Return immediately if we find an improvement
                    if new_value < curr_value:
                        self.logger.info(
                            f"First relocate improvement found at [{v1}:{v2}] "
                            f"to {insert_pos} after {evaluations} evaluations: "
                            f"value reduced from {curr_value:.2f} to {new_value:.2f}",
                        )
                        return new_route

        self.logger.debug(f"No relocate improvement found after {evaluations} evaluations")
        return route
