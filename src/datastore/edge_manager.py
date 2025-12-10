from schemas.node import Node
from utils.logger import Logger
from datastore.distance_manager import EuclidianDistanceManager


class EdgeManager:
    """A manager for edge objects.

    An edge is tuple of two nodes.
    """

    logger: Logger
    nodes: dict[str, Node] = {}
    edges: dict[tuple[str, str], bool] = {}
    respect_even_to_odd_travel_constraint: bool
    respect_odd_to_even_travel_constraint: bool

    def __init__(
            self,
            logger: Logger | None = None,
            *,
            respect_even_to_odd_travel_constraint: bool = True,
            respect_odd_to_even_travel_constraint: bool = True,
        ) -> None:
        """Initialize the node manager."""
        self.logger = logger or Logger(__name__)
        self.nodes = {}
        self.respect_even_to_odd_travel_constraint = respect_even_to_odd_travel_constraint
        self.respect_odd_to_even_travel_constraint = respect_odd_to_even_travel_constraint

    def add_node(self, node: Node) -> None:
        """Add a Node to the manager."""
        self.nodes[node.id] = node

    def is_edge_valid(self, node_from: Node, node_to: Node) -> bool:
        """Check if an edge between two nodes is valid."""
        if self.respect_even_to_odd_travel_constraint:
            n = len(self.nodes)
            # traveling form node $𝑖 ∈ 𝑁  \ {0, 𝑛 + 1}$ to node $𝑗 ∈ 𝑁 \ {0, 𝑛 + 1}$ is forbidden
            # when: (i) $𝑖$ is an even number, (ii) $𝑗$ is an odd number, and (iii) $𝑖 < 𝑛/2$, and
            if (
                int(node_from.id) % 2 == 0 and
                int(node_to.id) % 2 == 1 and
                int(node_from.id) < n / 2
            ):
                return False
        if self.respect_odd_to_even_travel_constraint:
            n = len(self.nodes)
            # * traveling form node $𝑖 ∈ 𝑁 \ {0, 𝑛 + 1}$ to node $𝑗 ∈ 𝑁 \ {0, 𝑛 + 1}$ is forbidden
            # when: (i) $𝑖$ is an odd number, (ii) $𝑗$ is an even number, and (iii) $𝑖 ≥ 𝑛/2$
            if (
                int(node_from.id) % 2 == 1 and
                int(node_to.id) % 2 == 0 and
                int(node_from.id) >= n / 2
            ):
                return False
        return True

    def neighbors(
            self,
            node_id: str,
            *,
            candidates: list[Node] | None = None,
            max_neighbors: int | None = None,
            sort_by_distance: bool = False,
            distance_manager: EuclidianDistanceManager | None = None,
        ) -> list[Node]:
        """Get all neighboring nodes for a given node ID."""
        if node_id not in self.nodes:
            self.logger.warning(f"Node ID {node_id} not found in EdgeManager.")
            return []
        if candidates and len(candidates) == 0:
            self.logger.info(f"No candidate nodes provided for neighbors of node ID {node_id}.")
            return []

        candidates = candidates or list(self.nodes.values())

        if self.nodes[node_id] in candidates:
            candidates.remove(self.nodes[node_id])

        if len(candidates) == 0:
            self.logger.info(f"No candidate nodes identified for neighbors of node ID {node_id}.")
            return []

        candidates = [n for n in candidates if self.is_edge_valid(self.nodes[node_id], n)]

        if sort_by_distance:
            distance_manager = distance_manager or EuclidianDistanceManager(logger=self.logger)
            candidates.sort(key=lambda node: distance_manager.get_distance(self.nodes[node_id], node))

        if max_neighbors is not None:
            return candidates[:max_neighbors]

        return candidates
