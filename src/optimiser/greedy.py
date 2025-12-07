from datastore.node_manager import NodeManager
from datastore.edge_manager import EdgeManager
from datastore.distance_manager import EuclidianDistanceManager
from utils.logger import Logger
from schemas.route import Route

class GreedyOptimiser:
    """Class for greedy optimisation algorithm."""

    logger: Logger

    def __init__(self, logger: Logger) -> None:
        """Initialise the instance."""
        self.logger = logger

    def optimise(
            self,
            node_manager: NodeManager,
            edge_manager: EdgeManager,
            distance_manager: EuclidianDistanceManager,
        ) -> Route:
        """Perform greedy optimisation to create a route."""

        unvisited_nodes = node_manager.all_nodes()
        if not unvisited_nodes:
            self.logger.warning("No nodes available for optimisation.")
            return Route(nodes=[])

        curr_node = unvisited_nodes[0]
        last_node = unvisited_nodes[-1]
        route_nodes = [curr_node]
        unvisited_nodes.remove(curr_node)
        unvisited_nodes.remove(last_node)
        while unvisited_nodes:
            """Find the closest unvisited node."""
            closest_unvisited_nodes = edge_manager.neighbors(
                node_id=curr_node.id,
                candidates=unvisited_nodes,
                max_neighbors=1,
                sort_by_distance=True,
                distance_manager=distance_manager,
            )
            closest_node = closest_unvisited_nodes[0] if closest_unvisited_nodes else None
            if closest_node is None:
                self.logger.debug("No more reachable unvisited nodes.")
                break
            route_nodes.append(closest_node)
            unvisited_nodes.remove(closest_node)
            curr_node = closest_node

        route_nodes.append(last_node)
        self.logger.debug(f"Optimised route with {len(route_nodes)} nodes.")
        return Route(nodes=route_nodes, logger=self.logger)
