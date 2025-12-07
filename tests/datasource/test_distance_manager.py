import pytest

from src.schemas.node import Node


@pytest.mark.parametrize(("node_source", "node_target", "expected_distance"), [
    (Node(id="A", x=0, y=0), Node(id="B", x=3, y=4), 5.0),
    (Node(id="C", x=1, y=1), Node(id="D", x=4, y=5), 5.0),
])
def test_get_distance(node_source: Node, node_target: Node, expected_distance: float):
    from src.datastore.distance_manager import EuclidianDistanceManager
    distance_mngr = EuclidianDistanceManager()
    dist = distance_mngr.get_distance(node_source, node_target)
    assert dist == expected_distance
