#!/usr/bin/env python3
import os
from pathlib import Path

from dotenv import load_dotenv

from bounds.lower_bound import LowerBoundCalculator
from bounds.upper_bound import UpperBoundCalculator
from datastore.distance_manager import EuclidianDistanceManager
from datastore.edge_manager import EdgeManager
from datastore.node_manager import NodeManager
from eval.route_eval import RouteEvaluator
from input_processing.csv_parser import CSVParser
from input_processing.data_validation import NodeValidator
from optimiser.initial.naive import NaiveSequencer
from optimiser.iterative.local_search import LocalSearchImprover
from schemas.node import Node
from utils.logger import Logger
from report.route_export import RouteExporter

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

logger.level = "INFO"
distance_mngr = EuclidianDistanceManager(logger=logger)

# ==================
# COMPUTE LOWER AND UPPER BOUNDS

logger.level = "INFO"
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
# EVALUATOR AND EXPORTERS

route_eval = RouteEvaluator(
    node_manager=node_mngr,
    edge_manager=edge_mngr,
    distance_manager=distance_mngr,
    logger=logger,
)

route_exporter = RouteExporter(
    route_eval=route_eval,
    nodes=node_mngr.all_nodes(),
)

# ==================
# CONSTRUCTION HEURISTIC

logger.level = "DEBUG"

# Naive Sequencer
naive_sequencer = NaiveSequencer(node_manager=node_mngr, logger=logger)
naive_route = naive_sequencer.optimise()
logger.info(f"Naive output route: {naive_route}")
logger.info(f"Objective value: {route_eval.calculate_objective_value(route=naive_route)}")
if not route_eval.is_valid_route(route=naive_route):
    logger.error("The route is invalid.")
else:
    print(route_exporter.report_format(route=naive_route))
route_exporter.plot_to_file(
    route=naive_route,
    title="NaiveSequencer",
    filepath=Path(os.getenv("OUTPUT_DIR"), "naive.png").absolute(),
)
logger.info("Route plot saved.")

# ==================
# OPTIMISATION: ITERATIVE IMPROVEMENT

local_search_improver = LocalSearchImprover(
    logger=logger,
    node_manager=node_mngr,
    edge_manager=edge_mngr,
    distance_manager=distance_mngr,
)
local_search_improver.add_seed_route(route=naive_route)
best_route = local_search_improver.optimise()
logger.info(f"Best found route: {best_route}")
if not route_eval.is_valid_route(route=best_route):
    logger.error("The route is invalid.")
else:
    print(route_exporter.report_format(route=best_route))
route_exporter.plot_to_file(
    route=best_route,
    title="LocalSearchImprover",
    filepath=Path(os.getenv("OUTPUT_DIR"), "naive.png").absolute(),
)
logger.info("Route plot saved.")
route_exporter.report_to_file(
    route=best_route,
    filepath=Path(os.getenv("OUTPUT_DIR"), "local_search.txt").absolute(),
)
