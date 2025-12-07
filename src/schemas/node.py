from pydantic import BaseModel


class Node(BaseModel):
    """A node."""

    id: int
    x: float
    y: float
