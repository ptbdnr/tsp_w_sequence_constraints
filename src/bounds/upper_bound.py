from datastore.distance_manager import EuclidianDistanceManager
from datastore.node_manager import NodeManager
from utils.logger import Logger


class UpperBoundCalculator:
    """Class to calculate upper bounds for TSP solutions."""

    logger: Logger

    def __init__(self, logger: Logger | None = None) -> None:
        """Initialize the upper bound calculator."""
        self.logger = logger or Logger(__name__)

    def calculate_upper_bound(
            self,
            node_manager: NodeManager,
            distance_manager: EuclidianDistanceManager,
        ) -> float:
        """Calculate an upper bound for the VRP solution.

        The upper bound is calculated as $L * max(d_ij) + max(d_ij) * n$,
        i.e., $max(d_ij) * n * ( max(d_ij) + 1 )$
        where:
            L = max(d_ij) * n, where max(d_ij) is the maximum distance in the entire distance matrix,
            max_distance is the maximum distance between any two nodes,
            and n is the total number of nodes including depots.

        Args:
            node_manager (NodeManager): The manager containing all nodes.
            distance_manager (EuclidianDistanceManager): The manager to calculate distances.

        Returns:
            float: The calculated upper bound.

        """

        node_ids = [int(x) for x in node_manager.all_node_ids()]
        n = len(node_ids)
        max_distance = 0.0  # Initialize max distance
        for i in range(n):
            for j in range(i + 1, n):
                node1 = node_manager.get_node(node_ids[i])
                node2 = node_manager.get_node(node_ids[j])
                if node1 and node2:
                    distance = distance_manager.get_distance(node1, node2)
                    max_distance = max(max_distance, distance)  # Update max distance
        self.logger.debug(f"Maximum distance between any two nodes: {max_distance}")
        self.logger.debug(f"Total number of nodes (n): {n}")
        upper_bound = max_distance * n * (max_distance + 1)
        self.logger.info(f"Calculated upper bound: {upper_bound}")
        return upper_bound
