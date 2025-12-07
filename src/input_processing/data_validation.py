from schemas.node import Node


class NodeValidator:
    """A validator for Node objects."""

    @staticmethod
    def validate(node: Node) -> bool:
        """Validate a Node object."""

        return isinstance(node.id, int) and node.id >= 0 and isinstance(node.x, float) and isinstance(node.y, float)
