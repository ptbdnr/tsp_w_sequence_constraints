from random import SystemRandom

from eval.route_eval import RouteEvaluator
from optimiser.iterative.operations.operation import Operation
from schemas.route import Route
from utils.logger import Logger

MIN_ROUTE_LENGTH = 6


class ThreeOptSwap(Operation):
    """Class for three-opt swap operation.

    The 3-opt swap removes three edges and reconnects the route in different ways.
    This is more powerful than 2-opt but computationally more expensive.

    Given three cut points (v1, v2, v3) that divide the route into 4 segments:
    - Segment A: route[0:v1]
    - Segment B: route[v1:v2]
    - Segment C: route[v2:v3]
    - Segment D: route[v3:end]

    There are 8 possible ways to reconnect these segments:
    1. A-B-C-D (original, no change)
    2. A-B-revC-D (2-opt on C)
    3. A-revB-C-D (2-opt on B)
    4. A-C-B-D (swap B and C)
    5. A-revB-revC-D (reverse B and C)
    6. A-C-revB-D (swap B and C, reverse B)
    7. A-revC-B-D (swap B and C, reverse C)
    8. A-revC-revB-D (swap and reverse both B and C)
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

    def _get_all_reconnections(
            self,
            segment_a: list,
            segment_b: list,
            segment_c: list,
            segment_d: list,
        ) -> list[list]:
        """Generate all possible 3-opt reconnections.

        Args:
            segment_a: First segment
            segment_b: Second segment
            segment_c: Third segment
            segment_d: Fourth segment

        Returns:
            List of all possible reconnections

        """
        reconnections = [
            # 1. Original (no change)
            segment_a + segment_b + segment_c + segment_d,
            # 2. Reverse C (2-opt on C)
            segment_a + segment_b + list(reversed(segment_c)) + segment_d,
            # 3. Reverse B (2-opt on B)
            segment_a + list(reversed(segment_b)) + segment_c + segment_d,
            # 4. Swap B and C
            segment_a + segment_c + segment_b + segment_d,
            # 5. Reverse both B and C
            segment_a + list(reversed(segment_b)) + list(reversed(segment_c)) + segment_d,
            # 6. Swap B and C, reverse B
            segment_a + segment_c + list(reversed(segment_b)) + segment_d,
            # 7. Swap B and C, reverse C
            segment_a + list(reversed(segment_c)) + segment_b + segment_d,
            # 8. Swap and reverse both B and C
            segment_a + list(reversed(segment_c)) + list(reversed(segment_b)) + segment_d,
        ]
        return reconnections

    def apply(
            self,
            route: Route,
            v1: int | None = None,
            v2: int | None = None,
            v3: int | None = None,
            reconnection_type: int | None = None,
            *,
            inplace: bool = False,
        ) -> Route:
        """Apply the 3-opt swap operation to the given route.

        Args:
            route: The route to apply the operation to
            v1: First cut point (if None, randomly selected)
            v2: Second cut point (if None, randomly selected)
            v3: Third cut point (if None, randomly selected)
            reconnection_type: Type of reconnection (0-7, if None, randomly selected)
            inplace: If True, modify the route in place; otherwise create a new route

        Returns:
            The modified route (either new or the same object if inplace=True)

        """
        route_length = len(route.sequence)

        # Need at least 6 nodes to perform 3-opt (start, 3 intermediate segments, end)
        if route_length < MIN_ROUTE_LENGTH:
            self.logger.warning(f"Route too short for 3-opt swap (length={route_length})")
            return route if inplace else route.copy()

        # Generate random indices if not provided
        # We exclude the first and last nodes (depot nodes)
        if v1 is None or v2 is None or v3 is None:
            v1 = self.rnd_generator.randint(1, route_length - 5)
            v2 = self.rnd_generator.randint(v1 + 1, route_length - 3)
            v3 = self.rnd_generator.randint(v2 + 1, route_length - 2)
        else:
            # Ensure v1 < v2 < v3
            indices = sorted([v1, v2, v3])
            v1, v2, v3 = indices

            # Validate indices
            if v1 < 1 or v3 >= route_length - 1 or not (v1 < v2 < v3):
                self.logger.error(
                    f"Invalid indices for 3-opt: v1={v1}, v2={v2}, v3={v3}, "
                    f"route_length={route_length}",
                )
                return route if inplace else route.copy()

        # Select reconnection type (0-7)
        if reconnection_type is None:
            reconnection_type = self.rnd_generator.randint(1, 7)  # Skip 0 (original)
        elif not 0 <= reconnection_type <= 7:
            self.logger.error(f"Invalid reconnection_type: {reconnection_type}, must be 0-7")
            return route if inplace else route.copy()

        self.logger.debug(
            f"Applying 3-opt swap at indices [{v1}, {v2}, {v3}] "
            f"with reconnection type {reconnection_type}",
        )

        # Extract segments
        segment_a = route.sequence[:v1]
        segment_b = route.sequence[v1:v2]
        segment_c = route.sequence[v2:v3]
        segment_d = route.sequence[v3:]

        # Get all possible reconnections
        all_reconnections = self._get_all_reconnections(
            segment_a, segment_b, segment_c, segment_d,
        )

        # Select the desired reconnection
        new_sequence = all_reconnections[reconnection_type]

        # Apply the change
        if inplace:
            route.sequence = new_sequence
            self.logger.debug(
                f"Applied 3-opt swap in place at indices [{v1}, {v2}, {v3}] "
                f"type {reconnection_type}",
            )
            return route

        # Create a new route
        new_route = Route(name=route.name, sequence=new_sequence, logger=self.logger)
        self.logger.debug(
            f"Created new route with 3-opt swap at indices [{v1}, {v2}, {v3}] "
            f"type {reconnection_type}",
        )
        return new_route

    def apply_best_improvement(
            self,
            route: Route,
            *,
            only_valid: bool = True,
        ) -> Route:
        """Apply the best 3-opt improvement to the route.

        Tries all possible 3-opt swaps and returns the one with the best improvement.

        Args:
            route: The route to improve

        Returns:
            The improved route (or original if no improvement found)

        """
        best_route = route.copy()
        orig_value = best_value = self.route_eval.calculate_objective_value(route)
        improved = False

        route_length = len(route.sequence)
        evaluations = 0

        # Try all possible 3-opt swaps
        for v1 in range(1, route_length - 4):
            for v2 in range(v1 + 1, route_length - 2):
                for v3 in range(v2 + 1, route_length - 1):
                    # Try all reconnection types (skip 0 which is original)
                    for reconnection_type in range(1, 8):
                        # Create new route with this swap
                        new_route = self.apply(
                            route,
                            v1=v1,
                            v2=v2,
                            v3=v3,
                            reconnection_type=reconnection_type,
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
                                f"Found improvement with 3-opt [{v1}, {v2}, {v3}] "
                                f"type {reconnection_type}: value reduced to {new_value:.2f}",
                            )

        if improved:
            self.logger.info(
                f"Best 3-opt improvement found after {evaluations} evaluations: "
                f"value reduced from {orig_value:.2f} to {best_value:.2f}",
            )
        else:
            self.logger.debug(
                f"No 3-opt improvement found after {evaluations} evaluations",
            )

        return best_route

    def apply_first_improvement(
            self,
            route: Route,
            *,
            only_valid: bool = True,
        ) -> Route:
        """Apply the first 3-opt improvement found.

        Stops as soon as an improvement is found (faster than best improvement).

        Args:
            route: The route to improve

        Returns:
            The improved route (or original if no improvement found)

        """
        curr_value = self.route_eval.calculate_objective_value(route)
        route_length = len(route.sequence)
        evaluations = 0

        # Try 3-opt swaps until we find an improvement
        for v1 in range(1, route_length - 4):
            for v2 in range(v1 + 1, route_length - 2):
                for v3 in range(v2 + 1, route_length - 1):
                    # Try all reconnection types (skip 0 which is original)
                    for reconnection_type in range(1, 8):
                        # Create new route with this swap
                        new_route = self.apply(
                            route,
                            v1=v1,
                            v2=v2,
                            v3=v3,
                            reconnection_type=reconnection_type,
                            inplace=False,
                        )
                        if only_valid and not self.route_eval.is_valid_route(route=new_route):
                            continue
                        new_value = self.route_eval.calculate_objective_value(route=new_route)
                        evaluations += 1

                        # Return immediately if we find an improvement
                        if new_value < curr_value:
                            self.logger.info(
                                f"First 3-opt improvement found at [{v1}, {v2}, {v3}] "
                                f"type {reconnection_type} after {evaluations} evaluations: "
                                f"value reduced from {curr_value:.2f} to {new_value:.2f}",
                            )
                            return new_route

        self.logger.debug(f"No 3-opt improvement found after {evaluations} evaluations")
        return route
