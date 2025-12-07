import pytest

from src.schemas.node import Node


@pytest.mark.parametrize(("nodes"), [
    [Node(id=str(i), x=i * 1.0, y=i * 1.0) for i in range(10)],
])
def test_get_closest_k_nodes(nodes: list[Node]):
    from src.datastore.distance_manager import EuclidianDistanceManager
    from src.datastore.node_manager import NodeManager
    node_mngr = NodeManager()
    distance_mngr = EuclidianDistanceManager()
    closest_nodes = node_mngr.get_closest_k_nodes(
        target_node=nodes[0],
        k=5,
        distance_manager=distance_mngr,
    )
    assert len(closest_nodes) == 5
