from functools import cache

from datastore.distance_manager import EuclidianDistanceManager
from datastore.node_manager import NodeManager
from schemas.node import Node
from utils.logger import Logger

MIN_ROUTE_NODES = 2  # Including start and end depot nodes


@cache
def get_l_value(
    node_manager: NodeManager,
    distance_manager: EuclidianDistanceManager,
) -> float:
    """Calculate L = max(d_ij) * n, where max(d_ij) is the maximum distance in the entire distance matrix.

    Args:
        node_manager: Manager containing all nodes
        distance_manager: Manager for distance calculations
    Returns:

        The L value

    """
    n = len(node_manager.nodes) - 2  # Excluding start (0) and end (n+1) depot

    max_distance_matrix = 0.0
    all_nodes = list(node_manager.nodes.values())
    for i, node1 in enumerate(all_nodes):
        for node2 in all_nodes[i + 1:]:
            dist = distance_manager.get_distance(node1, node2)
            max_distance_matrix = max(max_distance_matrix, dist)

    return max_distance_matrix * n


class Route:
    """A class representing a route consisting of multiple nodes."""

    logger: Logger
    nodes: list[Node]

    def __init__(self, nodes: list[Node], logger: Logger | None = None) -> None:
        """Initialize class."""
        self.logger = logger or Logger(__name__)
        self.nodes = nodes

    def total_distance_and_distances(
            self,
            distance_manager: EuclidianDistanceManager | None = None,
        ) -> tuple[float, list[float]]:
        """Calculate the total distance of the route."""

        if len(self.nodes) < MIN_ROUTE_NODES:
            return 0.0, []

        if not distance_manager:
            distance_manager = EuclidianDistanceManager(logger=self.logger)

        total_distance = 0.0
        distances: list[float] = []
        for i in range(len(self.nodes) - 1):
            curr_distance = distance_manager.get_distance(self.nodes[i], self.nodes[i + 1])
            distances.append(curr_distance)
            total_distance += curr_distance

        if not distances:
            return 0.0, []

        return total_distance, distances

    def total_distance(self, distance_manager: EuclidianDistanceManager | None = None) -> float:
        """Calculate the total distance of the route."""
        return self.total_distance_and_distances(distance_manager)[0]

    def calculate_objective_value(
        self,
        node_manager: NodeManager,
        distance_manager: EuclidianDistanceManager | None = None,
    ) -> float:
        """Calculate the objective function value: L·Δ + D.

        Where:
        - L = max(d_ij) * n, where max(d_ij) is the maximum distance in the entire distance matrix
        - Δ = delta value (maxD - minD for the route)
        - D = total distance of the route
        - n = number of nodes (excluding start and end depot nodes)

        Args:
            node_manager: Manager containing all nodes
            distance_manager: Manager for distance calculations

        Returns:
            The objective function value

        """
        if not distance_manager:
            distance_manager = EuclidianDistanceManager(logger=self.logger)

        # Calculate D (total distance)
        d_value, distances = self.total_distance_and_distances(distance_manager)

        # Calculate Δ (delta)
        max_distance = max(distances)
        min_distance = min(distances)
        delta = max_distance - min_distance
        self.logger.debug(f"Delta calculation: maxD={max_distance}, minD={min_distance}, Δ={delta}")

        # Calculate L
        l_value = get_l_value(node_manager, distance_manager)

        # Calculate objective value
        objective_value = l_value * delta + d_value

        self.logger.debug(
            f"Objective calculation: L={l_value:.2f}, Δ={delta:.2f}, D={d_value:.2f}, "
            f"Objective={objective_value:.2f}",
        )

        return objective_value

    def is_valid_route(
        self,
        node_manager: NodeManager,
    ) -> bool:
        """Check if the route is valid based on sequence constraints.

        Constraints:
        1. Route must start from node 0
        2. Route must end at node n+1
        3. Each intermediate node must be visited exactly once
        4. Even→Odd forbidden: i is even, j is odd, i < n/2
        5. Odd→Even forbidden: i is odd, j is even, i >= n/2

        Args:
            node_manager: Manager containing all nodes

        Returns:
            True if the route is valid, False otherwise

        """
        if len(self.nodes) < MIN_ROUTE_NODES:
            self.logger.warning("Route has fewer than 2 nodes")
            return False

        # Check if route starts at node 0
        if self.nodes[0].id not in {"0", 0}:
            self.logger.warning(f"Route does not start at node 0, starts at {self.nodes[0].id}")
            return False

        # Get all node IDs and sort them to determine n
        # assuming node IDs are integers
        all_node_ids = sorted([int(node_id) for node_id in node_manager.all_node_ids()])
        n = len(all_node_ids) - 2  # Excluding 0 and n+1

        # Check if route ends at node n+1
        expected_end = n + 1
        if self.nodes[-1].id not in {str(expected_end), expected_end}:
            self.logger.warning(f"Route does not end at node {expected_end}, ends at {self.nodes[-1].id}")
            return False

        # Check if all intermediate nodes are visited exactly once
        intermediate_nodes = [int(node.id) for node in self.nodes[1:-1]]
        expected_intermediate = [str(i) for i in range(1, n + 1)]

        if sorted(intermediate_nodes) != sorted(expected_intermediate):
            self.logger.warning("Not all intermediate nodes are visited exactly once")
            return False

        # Check sequence constraints for consecutive node pairs
        for i in range(len(self.nodes) - 1):
            current_node = self.nodes[i]
            next_node = self.nodes[i + 1]

            # Skip constraints for depot nodes (0 and n+1)
            if current_node.id == "0" or current_node.id == str(n + 1):
                continue
            if next_node.id == "0" or next_node.id == str(n + 1):
                continue

            current_id = int(current_node.id)
            next_id = int(next_node.id)

            # Constraint 1: Even→Odd forbidden when i < n/2
            if current_id % 2 == 0 and next_id % 2 == 1 and current_id < n / 2:
                self.logger.warning(
                    f"Constraint violated: Even→Odd transition from {current_id} to {next_id} "
                    f"with {current_id} < n/2 (n={n})",
                )
                return False

            # Constraint 2: Odd→Even forbidden when i >= n/2
            if current_id % 2 == 1 and next_id % 2 == 0 and current_id >= n / 2:
                self.logger.warning(
                    f"Constraint violated: Odd→Even transition from {current_id} to {next_id} "
                    f"with {current_id} >= n/2 (n={n})",
                )
                return False

        self.logger.info("Route is valid")
        return True

    def __str__(self) -> str:
        """Get the route as a string representation.

        Returns:
            String in format "0-3-1-2-4-5"

        """
        return "-".join([str(node.id) for node in self.nodes])

    def __repr__(self) -> str:
        """Representation the route as a string."""
        return f"Route({self.__str__()})"

    def __len__(self) -> int:
        """Return the number of nodes in the route."""
        return len(self.nodes)

    def copy(self) -> "Route":
        """Create a deep copy of the route."""
        return Route(nodes=self.nodes.copy(), logger=self.logger)

    def report_format(self) -> str:
        """Get a report representation of the route.

        ```text
        Route:0-3-1-2
        Total Distance: 849.25
        Delta Value: 12.38
        ```
        """
        total_distance, distances = self.total_distance_and_distances()
        max_distance = max(distances) if distances else 0.0
        min_distance = min(distances) if distances else 0.0
        delta = max_distance - min_distance

        return f"Route: {self}\nTotal Distance: {total_distance:.2f}\nDelta Value :{delta:.2f}"
