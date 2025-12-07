from pathlib import Path

from schemas.node import Node
from utils.logger import Logger

SKIP_HEADER = True
EXPECTED_NUM_FIELDS = 3


class CSVParser:
    """A simple CSV parser."""

    logger: Logger

    def __init__(
            self,
            logger: Logger | None = None,
        ) -> None:
        """Initialize the CSV parser."""
        self.logger = logger or Logger(__name__)

    def parse(
            self,
            filepath: Path | str,
        ) -> list[Node]:
        """Parse a CSV file into a list of Node objects."""
        nodes: list[Node] = []
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with filepath.open("r") as file:
            for line_idx, line in enumerate(file):
                if line_idx == 0 and SKIP_HEADER:
                    continue  # Skip header line
                parts = line.strip().split(",")
                if len(parts) != EXPECTED_NUM_FIELDS:
                    self.logger.warning(
                        f"Skipping invalid line in CSV: {line.strip()}",
                    )
                node = Node(
                    id=parts[0],
                    x=float(parts[1]),
                    y=float(parts[2]),
                )
                nodes.append(node)
        return nodes
