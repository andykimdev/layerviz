"""
Check that a layerviz bundle folder holds index.html and a graph that lists nodes.

Purpose:

- Read the graph from graph.json, or from the graph script embedded in a self contained page.
- Print the node count and exit 0, or print each problem and exit 1.

Special Notes:

- Usage: python check_bundle.py <bundle folder> [...].
- For snapshots, pass one step folder such as step_000200.
- Uses only the standard library, so it runs in any environment.
"""

import json
import re
import sys
from pathlib import Path

# The inline page stores the graph in this element. Its JSON escapes </ as <\/, which json reads.
EMBEDDED = re.compile(r'<script type="application/json" id="graph">(.*?)</script>', re.DOTALL)


def read_graph(folder: Path) -> dict:
    """
    Return the graph dict of a bundle folder.

    Args:
        folder: A bundle folder written by capture or compare.

    Returns:
        The parsed graph.json, or the graph embedded in index.html when there is no graph.json.

    Raises:
        ValueError: When neither graph.json nor an embedded graph exists.
    """
    graph_file = folder / "graph.json"
    if graph_file.is_file():
        return json.loads(graph_file.read_text(encoding="utf-8"))
    index = folder / "index.html"
    match = EMBEDDED.search(index.read_text(encoding="utf-8")) if index.is_file() else None
    if match is None:
        raise ValueError(f"{folder} has no graph.json and no graph embedded in index.html")
    return json.loads(match.group(1))


def bundle_problems(folder: Path) -> list[str]:
    """
    Return the problems of one bundle folder, empty when it passes.

    Args:
        folder: A bundle folder written by capture or compare.

    Returns:
        One line per problem: a missing index.html, a missing graph, or a graph with no nodes.
    """
    if not (folder / "index.html").is_file():
        return [f"{folder}: no index.html"]
    try:
        graph = read_graph(folder)
    except ValueError as error:
        return [str(error)]
    if not graph.get("nodes"):
        return [f"{folder}: the graph lists no nodes"]
    return []


def main(argv: list[str]) -> int:
    """
    Check each bundle folder named on the command line and print the result.

    Args:
        argv: Bundle folder paths, the command line arguments after the script name.

    Returns:
        0 when every folder passes, 1 when any folder fails or no folder is named.
    """
    failed = False
    for arg in argv:
        folder = Path(arg)
        problems = bundle_problems(folder)
        failed = failed or bool(problems)
        if not problems:
            print(f"{folder}: ok, {len(read_graph(folder)['nodes'])} nodes")
        for line in problems:
            print(line)
    return 1 if failed or not argv else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
