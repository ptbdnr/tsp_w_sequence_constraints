from schemas.node import Node
from utils.logger import Logger


class EuclidianDistanceManager:
    """A manager for calculating Euclidian distances between nodes."""

    logger: Logger
    distances: dict[tuple[int, int], float]

    def __init__(
            self,
            nb_of_nodes: int,
            logger: Logger | None = None,
        ) -> None:
        """Initialize the distance manager."""
        self.logger = logger or Logger(__name__)
        self.distances = {
            (nb_of_nodes - 1, 0): 0.0,  # distance from last node to depot is zero
        }

    def get_distance(
            self,
            node1: Node,
            node2: Node,
        ) -> float:
        """Get the Euclidian distance between two nodes.

        Assumes undirected distances (distance from A to B is the same as from B to A).
        """

        min_id = min(node1.id, node2.id)
        if min_id == node1.id:
            node_min_id = node1
            node_max_id = node2
        else:
            node_min_id = node2
            node_max_id = node1
        key = (node_min_id.id, node_max_id.id)
        if key not in self.distances:
            distance = self.calculate_distance(node_min_id, node_max_id)
            self.distances[key] = distance
            self.logger.debug(f"Calculated distance between {node1.id} and {node2.id}: {distance}")
        return self.distances[key]

    @staticmethod
    def calculate_distance(
            node1: Node,
            node2: Node,
            precition_digits: int = 1,
        ) -> float:
        """Calculate the Euclidian distance between two nodes."""
        if node1.id == node2.id:
            return 0.0
        distance = ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** 0.5
        return round(distance, precition_digits)
