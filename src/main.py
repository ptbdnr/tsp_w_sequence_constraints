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
from optimiser.iterative.alns_wrapper import ALNSWrapper
from optimiser.iterative.callback import Callback
from optimiser.iterative.local_search import LocalSearchImprover
from optimiser.iterative.sa import SimulatedAnnealingImprover
from optimiser.iterative.termination import Termination
from report.route_export import RouteExporter
from schemas.node import Node
from utils.logger import Logger

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

Path(os.getenv("OUTPUT_DIR")).mkdir(parents=True, exist_ok=True)

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
distance_mngr = EuclidianDistanceManager(
    nb_of_nodes=len(nodes),
    logger=logger,
)

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

with Path(os.getenv("OUTPUT_DIR"), "bounds.txt").open("w", encoding="utf-8") as f:
    f.write(f"Upper bound: {ub}\n")
    f.write(f"Lower bound: {lb}\n")

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
title = "NaiveSequencer"
route_exporter.plot_route(
    route=naive_route,
    title=title,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}.png").absolute(),
)
logger.info("Route plot saved.")
route_exporter.report_to_file(
    route=naive_route,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}.txt").absolute(),
)

# ==================
# OPTIMISATION: ITERATIVE IMPROVEMENT

termination = Termination(
    max_iterations=int(os.getenv("TERMINATION_MAX_ITERATIONS")),
    max_seconds=float(os.getenv("TERMINATION_MAX_SECONDS")),
)

# ==================
# OPTIMISATION: ITERATIVE IMPROVEMENT: ALNS

alns_wrapper = ALNSWrapper(
    edge_manager=edge_mngr,
    distance_manager=distance_mngr,
    route_evaluator=route_eval,
    termination=termination,
    logger=logger,
)
alns_wrapper.add_seed_route(route=naive_route)
best_routes = alns_wrapper.optimise()
callback = Callback()
callback.load_alns_result_statistics(
    statistics=alns_wrapper.result.statistics,
)
best_route = best_routes[0]
logger.info(f"Best found route: {best_route}")
if not route_eval.is_valid_route(route=best_route):
    logger.error("The route is invalid.")
else:
    print(route_exporter.report_format(route=best_route))
title = "ALNSWrapper"
route_exporter.plot_route(
    route=best_route,
    title=title,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}.png").absolute(),
)
logger.info("Route plot saved.")
route_exporter.report_to_file(
    route=best_route,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}.txt").absolute(),
)
alns_wrapper.plot_result(
    title=title,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}_iter_default.png").absolute(),
)
callback.iterations_to_file(
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}_iter.json").absolute(),
)
callback.plot_iterations(
    title=title,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}_iter.png").absolute(),
)

# ==================
# OPTIMISATION: ITERATIVE IMPROVEMENT: LOCAL SEARCH

termination.reset()
callback = Callback()
improver = LocalSearchImprover(
    logger=logger,
    node_manager=node_mngr,
    edge_manager=edge_mngr,
    distance_manager=distance_mngr,
    termination=termination,
    callback=callback,
)
improver.add_seed_route(route=naive_route)
best_routes = improver.optimise()
best_route = best_routes[0]
logger.info(f"Best found route: {best_route}")
if not route_eval.is_valid_route(route=best_route):
    logger.error("The route is invalid.")
else:
    print(route_exporter.report_format(route=best_route))
title = "LocalSearchImprover"
route_exporter.plot_route(
    route=best_route,
    title=title,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}.png").absolute(),
)
logger.info("Route plot saved.")
route_exporter.report_to_file(
    route=best_route,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}.txt").absolute(),
)
callback.routes_to_file(
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}_routes.json").absolute(),
)
callback.iterations_to_file(
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}_iter.json").absolute(),
)
callback.plot_iterations(
    title=title,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}_iter.png").absolute(),
)

# ==================
# OPTIMISATION: ITERATIVE IMPROVEMENT: SIMULATED ANNEALING

termination.reset()
callback = Callback()
improver = SimulatedAnnealingImprover(
    logger=logger,
    node_manager=node_mngr,
    edge_manager=edge_mngr,
    distance_manager=distance_mngr,
    termination=termination,
    initial_temperature=1000.0,
    cooling_rate=0.95,
    min_temperature=0.01,
    callback=callback,
)
improver.add_seed_route(route=naive_route)
best_routes = improver.optimise()
best_route = best_routes[0]
logger.info(f"Best found route: {best_route}")
if not route_eval.is_valid_route(route=best_route):
    logger.error("The route is invalid.")
else:
    print(route_exporter.report_format(route=best_route))
title = "SimulatedAnnealingImprover"
route_exporter.plot_route(
    route=best_route,
    title=title,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}.png").absolute(),
)
logger.info("Route plot saved.")
route_exporter.report_to_file(
    route=best_route,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}.txt").absolute(),
)
callback.routes_to_file(
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}_routes.json").absolute(),
)
callback.iterations_to_file(
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}_iter.json").absolute(),
)
callback.plot_iterations(
    title=title,
    filepath=Path(os.getenv("OUTPUT_DIR"), f"{title}_iter.png").absolute(),
)
