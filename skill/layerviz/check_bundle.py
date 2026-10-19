"""
Check that a layerviz bundle folder holds a page, a graph of nodes, and every file they name.

Purpose:

- Read the graph from graph.json, or from the graph script embedded in a self contained page.
- Check that the nodes are well formed, at least one stores a layer, and every layer, channel,
  and page script file exists and is not empty.
- Print the node and layer counts and exit 0, or print each problem and exit 1.

Special Notes:

- Usage: python check_bundle.py <bundle folder> [...].
- For snapshots, pass one step folder such as step_000200.
- Does not decode layer files or open a browser.
- Uses only the standard library, so it runs in any environment.
"""

import json
import re
import sys
from pathlib import Path

# The inline page stores the graph in this element. Its JSON escapes </ as <\/, which json reads.
EMBEDDED = re.compile(r'<script type="application/json" id="graph">(.*?)</script>', re.DOTALL)
# A one page bundle stores each layer file as base64 text in an element named by its path.
INLINE_FILE = re.compile(r'<script\b[^>]*\bdata-path="([^"]+)">([^<]*)</script>')
SCRIPT_SRC = re.compile(r'<script\b[^>]*\bsrc="([^"]+)"')


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


def node_paths(node: dict) -> list:
    """
    Return the layer and channel paths one node names, without repeats.

    Args:
        node: One graph node, which may name file, files.a, files.b, and channels.

    Returns:
        The non null path values in that order, empty when the node stores no layer.
    """
    files = node.get("files")
    pair = files if isinstance(files, dict) else {}
    paths = []
    for value in (node.get("file"), pair.get("a"), pair.get("b"), node.get("channels")):
        if value is not None and value not in paths:
            paths.append(value)
    return paths


def named_paths(nodes: object) -> list:
    """
    Return every path named by the dict nodes of a node list.

    Args:
        nodes: The graph nodes value, which may be malformed.

    Returns:
        The paths of each dict node, empty when nodes is not a list.
    """
    if not isinstance(nodes, list):
        return []
    return [path for node in nodes if isinstance(node, dict) for path in node_paths(node)]


def stores_layer(node: dict) -> bool:
    """
    Return whether a graph node stores a layer, through a file or a files pair.

    Args:
        node: One graph node. A node that names only channels stores no layer.

    Returns:
        True when file, files.a, or files.b is set.
    """
    files = node.get("files")
    pair = files if isinstance(files, dict) else {}
    return any(value is not None for value in (node.get("file"), pair.get("a"), pair.get("b")))


def is_node(node: object) -> bool:
    """
    Return whether a graph node is a dict that holds an id and a name.

    Args:
        node: One item of the graph nodes list.

    Returns:
        True when node is a dict with both keys.
    """
    return isinstance(node, dict) and "id" in node and "name" in node


def node_problems(nodes: object, available: set) -> list[str]:
    """
    Return the problems of a graph node list, given the paths the bundle holds.

    Args:
        nodes: The graph nodes value, which should be a non empty list of node dicts.
        available: The paths stored as non empty files or as inline page elements.

    Returns:
        One line per malformed node, per path not in available, and one when no node stores
        a layer. Empty when the nodes pass.
    """
    if not isinstance(nodes, list):
        return ["the graph nodes are not a list"]
    if not nodes:
        return ["the graph lists no nodes"]
    malformed = [f"node {i} is not an object with an id and a name"
                 for i, node in enumerate(nodes) if not is_node(node)]
    if malformed:
        return malformed
    problems = [f"node {node['id']} names {path}, which is missing or empty"
                for node in nodes for path in node_paths(node) if path not in available]
    if not any(stores_layer(node) for node in nodes):
        problems.append("no node stores a layer")
    return problems


def script_problems(sources: list, available: set) -> list[str]:
    """
    Return one problem per page script source that the bundle does not hold.

    Args:
        sources: The src values of the script elements of index.html.
        available: The sources that are non empty files next to index.html.

    Returns:
        One line per missing or empty script, empty when every script is present.
    """
    return [f"index.html loads {src}, which is missing or empty"
            for src in sources if src not in available]


def files_on_disk(folder: Path, paths: list) -> set:
    """
    Return the paths that are non empty files inside a bundle folder.

    Args:
        folder: A bundle folder.
        paths: Paths relative to the folder. Values that are not strings never match.

    Returns:
        The subset of paths whose file exists and holds at least one byte.
    """
    found = set()
    for path in paths:
        target = folder / path if isinstance(path, str) else None
        if target is not None and target.is_file() and target.stat().st_size > 0:
            found.add(path)
    return found


def inline_files(page: str) -> set:
    """
    Return the paths of the non empty layer elements embedded in a one page bundle.

    Args:
        page: The index.html text.

    Returns:
        The data-path values whose element holds text.
    """
    return {path for path, text in INLINE_FILE.findall(page) if text.strip()}


def bundle_problems(folder: Path) -> list[str]:
    """
    Return the problems of one bundle folder, empty when it passes.

    Args:
        folder: A bundle folder written by capture or compare.

    Returns:
        One line per problem, each starting with the folder, except a missing graph whose
        message already names it.
    """
    index = folder / "index.html"
    if not index.is_file():
        return [f"{folder}: no index.html"]
    try:
        graph = read_graph(folder)
    except ValueError as error:
        return [str(error)]
    page = index.read_text(encoding="utf-8")
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    stored = inline_files(page) | files_on_disk(folder, named_paths(nodes))
    sources = SCRIPT_SRC.findall(page)
    scripts = files_on_disk(folder, sources)
    problems = node_problems(nodes, stored) + script_problems(sources, scripts)
    return [f"{folder}: {line}" for line in problems]


def summary(nodes: list) -> str:
    """
    Return the success line body for a node list that passed the check.

    Args:
        nodes: The graph nodes, each a dict.

    Returns:
        Text of the form ok, N nodes, M layers, where M counts nodes with a file or files pair.
    """
    layers = sum(1 for node in nodes if stores_layer(node))
    return f"ok, {len(nodes)} nodes, {layers} layers"


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
            print(f"{folder}: {summary(read_graph(folder)['nodes'])}")
        for line in problems:
            print(line)
    return 1 if failed or not argv else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
