import copy

import matplotlib.pyplot as plt
import networkx as nx
import numpy.random as rnd
from alns import ALNS
from alns.accept import LateAcceptanceHillClimbing
from alns.select import RouletteWheel
from alns.stop import MaxRuntime

from datastore.distance_manager import EuclidianDistanceManager
from datastore.edge_manager import EdgeManager
from eval.route_eval import RouteEvaluator
from optimiser.iterative.iterative import IterativeOptimiser
from optimiser.iterative.termination import Termination
from schemas.node import Node
from schemas.route import Route
from utils.logger import Logger

DEGREE_OF_DESTRUCTION = 0.1
LOOKBACK_PERIOD = 10
SEED = 42


class SolutionState:
    """Solution class."""

    nodes: list[Node]
    edges: dict[Node, Node]
    route_evaluator: RouteEvaluator
    edge_manager: EdgeManager
    distance_manager: EuclidianDistanceManager
    logger: Logger

    def __init__(
            self,
            route: Route,
            route_evaluator: RouteEvaluator,
            edge_manager: EdgeManager,
            distance_manager: EuclidianDistanceManager,
            logger: Logger | None = None,
        ) -> None:
        self.nodes = route.sequence
        self.edges = {}
        for idx in range(len(route.sequence) - 1):
            self.edges[route.sequence[idx]] = route.sequence[idx + 1]
        self.edges[route.sequence[-1]] = route.sequence[0]
        self.route_evaluator = route_evaluator
        self.edge_manager = edge_manager
        self.distance_manager = distance_manager
        self.logger = logger or Logger(__name__)

    def objective(self) -> float:
        """Calculate the objective value of the current solution."""
        if len(self.edges) != len(self.nodes):
            return float("inf")  # Incomplete solution
        sequence = self._reconstruct_sequence()
        if not sequence:
            return float("inf")  # Invalid solution with subcycles
        route = Route(name="temp", sequence=sequence)
        return self.route_evaluator.calculate_objective_value(route=route)

    def _reconstruct_sequence(self) -> list[Node]:
        sequence = []
        curr_node = self.nodes[0]
        visited_nodes = set()
        while len(sequence) < len(self.nodes):
            if curr_node in visited_nodes:
                break  # Cycle detected
            sequence.append(curr_node)
            visited_nodes.add(curr_node)

            if curr_node not in self.edges:
                break  # Edge missing
            curr_node = self.edges[curr_node]
        return sequence if len(sequence) == len(self.nodes) else []

    def to_route(self) -> Route:
        """Convert the solution state back to a Route object."""
        sequence = self._reconstruct_sequence()
        return Route(name="ALNS", sequence=sequence)

    def to_graph(self) -> nx.Graph:
        """NetworkX helper method."""
        graph = nx.Graph()

        for node in self.nodes:
            graph.add_node(node.id, pos=(node.x, node.y))

        for node_from, node_to in self.edges.items():
            graph.add_edge(node_from.id, node_to.id)

        return graph


def edges_to_remove(state: SolutionState) -> int:
    """Calculate the number of edges to remove based on the degree of destruction."""
    return int(len(state.edges) * DEGREE_OF_DESTRUCTION)


def worst_removal(current: SolutionState, rng: rnd.Generator) -> SolutionState:
    """Remove the edges with the largest distance."""
    destroyed = copy.deepcopy(current)

    if not destroyed.edges:
        return destroyed

    # Sort edges by distance (descending)
    sorted_edges = sorted(
        destroyed.edges.items(),
        key=lambda item: current.distance_manager.calculate_distance(item[0], item[1]),
        reverse=True,
    )

    # Remove the worst edges
    num_to_remove = edges_to_remove(current)
    for idx in range(min(num_to_remove, len(sorted_edges))):
        node_from = sorted_edges[idx][0]
        del destroyed.edges[node_from]

    return destroyed


def path_removal(current: SolutionState, rng: rnd.Generator) -> SolutionState:
    """Remove a consecutive sub-path."""
    destroyed = copy.deepcopy(current)

    if not destroyed.nodes or len(destroyed.nodes) < 3:
        return destroyed

    # Randomly select a starting node
    node_idx = rng.choice(len(destroyed.nodes))
    curr_node = destroyed.nodes[node_idx]

    # Remove a path of consecutive edges
    num_to_remove = edges_to_remove(current)
    for _ in range(num_to_remove):
        if curr_node not in destroyed.edges:
            break
        next_node = destroyed.edges[curr_node]
        del destroyed.edges[curr_node]
        curr_node = next_node

    return destroyed


def random_removal(current: SolutionState, rng: rnd.Generator) -> SolutionState:
    """Remove edges at random."""
    destroyed = copy.deepcopy(current)

    num_to_remove = edges_to_remove(current)
    node_indices = rng.choice(
        len(destroyed.nodes),
        min(num_to_remove, len(destroyed.nodes)),  # ✅ Ensure we don't remove more than exist
        replace=False,
    )

    for idx in node_indices:
        node = destroyed.nodes[idx]
        if node in destroyed.edges:
            del destroyed.edges[node]

    return destroyed


def would_form_subcycle(from_node: Node, to_node: Node, state: SolutionState) -> bool:
    """Check if adding an edge would form a subcycle in the current state.

    Notice the offsets: we do not count the current node under consideration,
    as it cannot yet be part of a cycle.
    """
    for step in range(1, len(state.nodes)):
        if to_node not in state.edges:
            return False

        to_node = state.edges[to_node]

        if from_node == to_node and step != len(state.nodes) - 1:
            return True

    return False


def greedy_repair(current: SolutionState, rng: rnd.Generator) -> SolutionState:
    """Repair the current solution greedily."""
    nodes_with_outgoing = set(current.edges.keys())
    orphaned_nodes = [node for node in current.nodes if node not in nodes_with_outgoing]

    if not orphaned_nodes:
        return current  # Already complete

    # Track visited nodes (those already in the tour)
    visited = set(current.edges.values())

    # This kind of randomness ensures we do not cycle between the same
    # destroy and repair steps every time.
    shuffled_idcs = rng.permutation(len(orphaned_nodes))
    orphaned_nodes = [orphaned_nodes[idx] for idx in shuffled_idcs]

    for orphan in orphaned_nodes:
        # Find valid unvisited nodes this orphan can connect to
        candidates = {
            other for other in current.nodes
            if other != orphan
            if other not in visited or other == current.nodes[0]  # Allow returning to start
            if not would_form_subcycle(orphan, other, current)
            if current.edge_manager.is_edge_valid(orphan, other)
        }

        if not candidates:
            # Fallback: connect to nearest valid node
            candidates = {
                other for other in current.nodes
                if other != orphan
                if current.edge_manager.is_edge_valid(orphan, other)
            }

        if candidates:
            # Connect to the nearest candidate
            nearest = min(
                candidates,
                key=lambda other: current.distance_manager.calculate_distance(orphan, other),
            )
            current.edges[orphan] = nearest
            visited.add(nearest)
        else:
            current.logger.warning(f"Could not repair edge for orphaned node {orphan.id}")

    return current


class ALNSWrapper(IterativeOptimiser):
    """Adaptive Large Neighbourhood Search optimiser wrapper."""

    logger: Logger
    init_sol: SolutionState
    route_evaluator: RouteEvaluator
    edge_manager: EdgeManager
    distance_manager: EuclidianDistanceManager
    termination: Termination

    def __init__(
            self,
            route_evaluator: RouteEvaluator,
            edge_manager: EdgeManager,
            distance_manager: EuclidianDistanceManager,
            termination: Termination,
            logger: Logger | None = None,
    ) -> None:
        """Initialise the ALNS optimiser."""
        self.logger = logger or Logger("ALNSWrapper")
        self.route_evaluator = route_evaluator
        self.edge_manager = edge_manager
        self.distance_manager = distance_manager
        self.termination = termination

    def add_seed_route(self, route: Route) -> None:
        """Add a seed route for iterative optimisation."""
        self.init_sol = SolutionState(
            route=route,
            route_evaluator=self.route_evaluator,
            distance_manager=self.distance_manager,
            edge_manager=self.edge_manager,
            logger=self.logger,
        )

    def optimise(self) -> list[Route]:
        """Run the ALNS optimiser and return the best solution found."""
        alns = ALNS(rnd.default_rng(SEED))

        alns.add_destroy_operator(random_removal)
        alns.add_destroy_operator(path_removal)
        alns.add_destroy_operator(worst_removal)

        alns.add_repair_operator(greedy_repair)
        select = RouletteWheel(
            scores=[3, 2, 1, 0.5],
            decay=0.8,
            num_destroy=3,
            num_repair=1,
        )
        accept = LateAcceptanceHillClimbing(lookback_period=LOOKBACK_PERIOD)
        stop = MaxRuntime(self.termination.max_seconds)

        self.result = alns.iterate(
            initial_solution=self.init_sol,
            op_select=select,
            accept=accept,
            stop=stop,
        )
        return [self.result.best_state.to_route()]

    def plot_result(self, title: str, filepath: str) -> None:
        """Plot the results of the optimisation to a file."""
        _, ax = plt.subplots(figsize=(12, 6))
        self.result.plot_objectives(ax=ax, lw=2)
        ax.set_title(f"ALNS optimisation progress: {title}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective value")
        plt.savefig(filepath)
