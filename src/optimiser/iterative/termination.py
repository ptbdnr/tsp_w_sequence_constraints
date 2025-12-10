from __future__ import annotations

import time


class Termination:
    """Class for termination criteria of iterative algorithms."""

    max_iterations: int
    max_seconds: float
    start_time: float
    min_value: float
    max_value: float

    def __init__(
            self,
            max_iterations: int = -1,
            max_seconds: float = -1.0,
            min_value: float = -1 * float("inf"),
            max_value: float = float("inf"),
        ) -> None:
        """Initialise the instance."""
        self.start_time = time.time()
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.min_value = min_value
        self.max_value = max_value

    def reset(self) -> None:
        """Reset the termination criteria."""
        self.start_time = time.time()

    def should_terminate(
            self,
            iteration_count: int | None = None,
            value: float | None = None,
        ) -> bool:
        """Determine if the iterative algorithm should terminate."""
        if self.max_iterations < 0 and self.max_seconds < 0.0:
            return False
        elapsed_time = time.time() - self.start_time
        if self.max_seconds > 0.0 and elapsed_time >= self.max_seconds:
            return True
        if self.max_iterations > 0 and iteration_count >= self.max_iterations:
            return True
        if value is not None:
            if self.min_value != -1 * float("inf") and value <= self.min_value:
                return True
            if self.max_value != float("inf") and value >= self.max_value:
                return True
        return False
