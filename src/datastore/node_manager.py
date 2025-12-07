from datastore.distance_manager import EuclidianDistanceManager
from schemas.node import Node
from utils.logger import Logger


class NodeManager:
    """A manager for Node objects."""

    logger: Logger
    nodes: dict[str, Node] = {}

    def __init__(
            self,
            logger: Logger | None = None,
        ) -> None:
        """Initialize the node manager."""
        self.logger = logger or Logger(__name__)
        self.nodes = {}

    def add_node(self, node: Node) -> None:
        """Add a Node to the manager."""
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Node | None:
        """Retrieve a Node by its ID."""
        return self.nodes.get(node_id)

    def all_node_ids(self) -> list[str]:
        """Get a list of all node IDs."""
        return list(self.nodes.keys())

    def all_nodes(self) -> list[Node]:
        """Get a list of all nodes."""
        return list(self.nodes.values())

    def get_closest_k_nodes(
            self,
            target_node: Node,
            k: int,
            distance_manager: EuclidianDistanceManager | None = None,
        ) -> list[Node]:
        """Get the k closest nodes to the target node."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda node: distance_manager.get_distance(target_node, node) if distance_manager else float("inf"),
        )
        return sorted_nodes[1:k + 1]  # Exclude the target node itself
