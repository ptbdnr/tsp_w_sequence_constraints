#!/usr/bin/env python3
import os

from dotenv import load_dotenv

from bounds.lower_bound import LowerBoundCalculator
from bounds.upper_bound import UpperBoundCalculator
from datastore.distance_manager import EuclidianDistanceManager
from datastore.edge_manager import EdgeManager
from datastore.node_manager import NodeManager
from input_processing.csv_parser import CSVParser
from input_processing.data_validation import NodeValidator
from schemas.node import Node
from utils.logger import Logger
from optimiser.greedy import GreedyOptimiser
from schemas.route import Route

load_dotenv()

# ==================
# CONFIGURATION

LOGGER_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = Logger(__name__, level=LOGGER_LEVEL)

# ==================
# LOAD DATA

nodes: list[Node] = CSVParser(logger=logger).parse(
    filepath=os.getenv("DATA_NODES_FILEPATH"),
)
logger.info(f"Parsed {len(nodes)} nodes from CSV file.")

# ==================
# PRECOMPUTE DATA

node_mngr = NodeManager(logger=logger)
edge_mngr = EdgeManager(logger=logger)
for node in nodes:
    if not NodeValidator.validate(node):
        logger.error(f"Invalid node data: {node}")
    else:
        node_mngr.add_node(node)
        edge_mngr.add_node(node)

distance_mngr = EuclidianDistanceManager(logger=logger)

# ==================
# COMPUTE LOWER AND UPPER BOUNDS

ub_calculator = UpperBoundCalculator(logger=logger)
ub = ub_calculator.calculate_upper_bound(
    node_manager=node_mngr,
    distance_manager=distance_mngr,
)
logger.info(f"Upper bound: {ub}")

lb_calculator = LowerBoundCalculator(logger=logger)
lb = lb_calculator.calculate_lower_bound(
    node_manager=node_mngr,
    distance_manager=distance_mngr,
)
logger.info(f"Lower bound: {lb}")

# ==================
# RUN OPTIMISATION

# Greedy Optimiser
greedy_optimiser = GreedyOptimiser(logger=logger)

route = greedy_optimiser.optimise(
    node_manager=node_mngr,
    edge_manager=edge_mngr,
    distance_manager=distance_mngr,
)

logger.info(f"Greedy output route: {route}")
print(route.report_format())
logger.info(f"Objective value: {route.calculate_objective_value(node_manager=node_mngr, distance_manager=distance_mngr)}")
if route.is_valid_route(node_manager=node_mngr):
    logger.info(route.report_format())
else:
    logger.error("The optimised route is invalid.")
