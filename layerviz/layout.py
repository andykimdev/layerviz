"""
Place architecture graph nodes by rank and draw container boxes around their leaves.

Purpose:

- Rank each leaf by its longest path from the input and center every rank on x.
- Compute one box per container that spans its placed leaves with a depth based margin.

Special Notes:

- Pure functions over names and edges. No torch and no file IO, so the browser only draws.
- The first leaf in execution order is the input node.
- Back edges from a reused module are ignored for ranking, so the longest path stays defined.
"""

from dataclasses import dataclass

MARGIN_PER_LEVEL = 0.3


@dataclass
class Placed:
    """
    One leaf node with its graph coordinates and its rank from the input.

    """

    name: str
    x: float
    y: float
    rank: int


@dataclass
class Box:
    """
    One container drawn as a rectangle around its leaves, with its depth from the root.

    """

    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    depth: int


def check_edges(leaf_names: list[str], edges: list[tuple[str, str]]) -> None:
    """
    Check that every edge joins two known leaves.

    Args:
        leaf_names: Leaf names in execution order.
        edges: (source, target) leaf name pairs.

    Raises:
        ValueError: When an edge names a node that is not in leaf_names.
    """
    known = set(leaf_names)
    for source, target in edges:
        for name in (source, target):
            if name not in known:
                raise ValueError(f"edge {source} -> {target} names {name}, which is not a leaf")


def rank_leaves(leaf_names: list[str], edges: list[tuple[str, str]]) -> dict[str, int | None]:
    """
    Rank each leaf by its longest path from the input over the given edges.

    Args:
        leaf_names: Leaf names in execution order, the first one being the input.
        edges: (source, target) pairs between known leaves.

    Returns:
        A dict from leaf name to rank, or None for a leaf the input cannot reach.

    Notes:
        An edge whose source runs at or after its target, as when a module is called twice,
        is skipped so ranks follow execution order.
    """
    order = {name: i for i, name in enumerate(leaf_names)}
    sources: dict[str, list[str]] = {name: [] for name in leaf_names}
    for source, target in edges:
        if order[source] < order[target]:
            sources[target].append(source)
    ranks: dict[str, int | None] = {}
    for i, name in enumerate(leaf_names):
        reached = [rank for s in sources[name] if (rank := ranks[s]) is not None]
        if i == 0:
            ranks[name] = 0
        elif reached:
            ranks[name] = max(reached) + 1
        else:
            ranks[name] = None
    return ranks


def place_leaves(leaf_names: list[str], ranks: dict[str, int | None]) -> list[Placed]:
    """
    Place leaves at y equal to rank and center each rank on x in execution order.

    Args:
        leaf_names: Leaf names in execution order.
        ranks: Rank per leaf, None for an unreachable leaf.

    Returns:
        One Placed per leaf in execution order.

    Notes:
        Unreachable leaves take rank 0 and sit after the reachable rank 0 leaves.
    """
    reachable = [name for name in leaf_names if ranks[name] is not None]
    unreachable = [name for name in leaf_names if ranks[name] is None]
    rows: dict[int, list[str]] = {}
    for name in reachable + unreachable:
        rank = ranks[name]
        rows.setdefault(0 if rank is None else rank, []).append(name)
    positions: dict[str, Placed] = {}
    for rank, names in rows.items():
        for position, name in enumerate(names):
            x = position - (len(names) - 1) / 2
            positions[name] = Placed(name, x, rank * 1.0, rank)
    return [positions[name] for name in leaf_names]


def ancestors(name: str, parents: dict[str, str | None]) -> list[str]:
    """
    List the containers above a node, nearest first.

    Args:
        name: A node name.
        parents: Map from node name to its parent name, or None at the root.

    Returns:
        The parent chain from the direct parent up to the root container.
    """
    chain: list[str] = []
    parent = parents.get(name)
    while parent is not None:
        chain.append(parent)
        parent = parents.get(parent)
    return chain


def container_boxes(placed: list[Placed], parents: dict[str, str | None]) -> list[Box]:
    """
    Build one box per container that holds at least one placed leaf.

    Args:
        placed: Placed leaves in execution order.
        parents: Map from node name to its parent name, or None at the root.

    Returns:
        Boxes sorted outer first, then by the execution order of their first leaf.

    Notes:
        The margin is 0.3 per level from the deepest container up, so outer boxes enclose
        inner ones.
    """
    members: dict[str, list[Placed]] = {}
    for leaf in placed:
        for container in ancestors(leaf.name, parents):
            members.setdefault(container, []).append(leaf)
    depths = {name: len(ancestors(name, parents)) for name in members}
    deepest = max(depths.values(), default=0)
    boxes = []
    for name, leaves in members.items():
        margin = MARGIN_PER_LEVEL * (deepest - depths[name] + 1)
        xs = [leaf.x for leaf in leaves]
        ys = [leaf.y for leaf in leaves]
        boxes.append(
            Box(name, min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin,
                depths[name])
        )
    first_seen = {name: i for i, name in enumerate(members)}
    return sorted(boxes, key=lambda box: (box.depth, first_seen[box.name]))


def layout_graph(
    leaf_names: list[str],
    edges: list[tuple[str, str]],
    parents: dict[str, str | None],
) -> tuple[list[Placed], list[Box]]:
    """
    Compute leaf coordinates and container boxes for the architecture graph.

    Args:
        leaf_names: Leaf names in execution order, the first one being the input.
        edges: (source, target) pairs between leaves, already projected from containers.
        parents: Map from node name to its parent name, or None at the root.

    Returns:
        The placed leaves in execution order and the container boxes, outer first.

    Raises:
        ValueError: When an edge names a node that is not in leaf_names.
    """
    check_edges(leaf_names, edges)
    placed = place_leaves(leaf_names, rank_leaves(leaf_names, edges))
    return placed, container_boxes(placed, parents)
