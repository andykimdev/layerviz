"""
Build the inline block that makes one self contained index.html from the web files.

Purpose:
- Rewrite each ES module into a function scope that hands its exports to its importers.
- Embed styles, the graph, base64 layer files, and text files such as licenses in the page.

Special Notes:
- Pure string transforms with no file IO. bundle.py reads the files and writes the page.
- Text files become HTML comments, so a vendor license ships with the single file page.
"""

import base64
import posixpath
import re

INLINE_SUFFIXES = (".js", ".css", ".txt")
IMPORT_RE = re.compile(
    r"(?:^|(?<=;))[ \t]*import(?=[\s{*\"'])"
    r"\s*(?:(?P<clause>[^;\"']*?)\s*from\s*)?[\"'](?P<spec>[^\"']+)[\"']\s*;?",
    re.M,
)
EXPORT_DECL_RE = re.compile(
    r"^export\s+((?:async\s+)?(?:function\s*\*?|const|let|var|class)\s*([A-Za-z_$][\w$]*))", re.M
)
# The lookahead leaves export { x } from for EXPORT_FROM_RE to reject, not a stray from clause.
EXPORT_LIST_RE = re.compile(r"\bexport\s*\{([^{}]*)\}(?!\s*from\b)\s*;?")
EXPORT_FROM_RE = re.compile(r"\bexport\s*(?:\*|\{[^{}]*\})\s*(?:as\s+[\w$]+\s*)?from\b")
# Browsers end a comment at --> or --!>, so text holding either would leak into the page.
COMMENT_END_RE = re.compile(r"--!?>")
# HTML ends a script at </script in any letter case, and <!-- can start an escaped state.
SCRIPT_END_RE = re.compile(r"</(script)", re.I)
GRAPH_LT_RE = re.compile(r"<(?=[/!])")
IDENT = r"[A-Za-z_$][\w$]*"


def cannot_inline(web: dict) -> list:
    """
    List the web files the inline page cannot embed.

    Args:
        web: Relative posix path to bytes of every shipped web file.

    Returns:
        Sorted paths whose suffix is not .js, .css, or .txt.
    """
    return sorted(rel for rel in web if not rel.endswith(INLINE_SUFFIXES))


def text_comment(rel: str, text: str) -> str:
    """
    Wrap a text file such as a license in an HTML comment headed by its path.

    Args:
        rel: The file's relative path in the web folder.
        text: The file's text.

    Returns:
        The comment, which keeps the text unchanged.

    Raises:
        ValueError: When the text holds --> or --!>, which would close the comment early.
    """
    if COMMENT_END_RE.search(text):
        raise ValueError(f"{rel} holds --> or --!>, which would end its HTML comment early")
    return f"<!-- {rel}\n{text}\n-->"


def _resolve_import(rel: str, spec: str, sources: dict) -> str:
    """Resolve a relative import of module rel to a module path, else raise ValueError."""
    target = posixpath.normpath(posixpath.join(posixpath.dirname(rel), spec))
    if not spec.startswith(".") or target not in sources:
        raise ValueError(f"{rel} imports {spec}, which is not a module of the web folder")
    return target


def _module_order(sources: dict) -> list:
    """Order module paths so each follows its imports, vendor first, raising on a cycle."""
    order, visiting = [], set()

    def visit(rel: str) -> None:
        if rel in order:
            return
        if rel in visiting:
            raise ValueError(f"web modules import each other in a cycle through {rel}")
        visiting.add(rel)
        for match in IMPORT_RE.finditer(sources[rel]):
            visit(_resolve_import(rel, match["spec"], sources))
        order.append(rel)

    for rel in sorted(sources, key=lambda r: (not r.startswith("vendor/"), r)):
        visit(rel)
    return order


def _module_var(rel: str) -> str:
    """Return the variable name that holds one inlined module's exports."""
    return "__lv_" + re.sub(r"\W", "_", rel)


def _import_binding(clause: str | None, var: str, rel: str) -> str:
    """
    Turn an import clause into a const that reads from a module variable.

    Supports named, namespace, default, and side effect imports, else raises ValueError.
    """
    if clause is None:
        return ""
    clause = clause.strip()
    if clause.startswith("{") and clause.endswith("}"):
        names = [name.strip() for name in clause[1:-1].split(",") if name.strip()]
        bindings = ", ".join(re.sub(r"\s+as\s+", ": ", name) for name in names)
        return f"const {{ {bindings} }} = {var};"
    star = re.fullmatch(rf"\*\s*as\s+({IDENT})", clause)
    if star:
        return f"const {star[1]} = {var};"
    if re.fullmatch(IDENT, clause):
        return f"const {clause} = {var}.default;"
    raise ValueError(f"{rel} has an unsupported import clause: {clause}")


def _strip_exports(rel: str, text: str) -> tuple:
    """
    Remove export syntax and return (source, [(public name, local name)]).

    Raises ValueError naming the file for an export from clause or any other export form
    this transform does not support.
    """
    if EXPORT_FROM_RE.search(text):
        raise ValueError(f"{rel} re-exports with an export from clause, which cannot inline")
    names = [(match[2], match[2]) for match in EXPORT_DECL_RE.finditer(text)]
    text = EXPORT_DECL_RE.sub(r"\1", text)
    for match in EXPORT_LIST_RE.finditer(text):
        for item in filter(None, (part.strip() for part in match[1].split(","))):
            local, _, public = item.partition(" as ")
            names.append(((public or local).strip(), local.strip()))
    text = EXPORT_LIST_RE.sub("", text)
    if re.search(r"^\s*export\b", text, re.M):
        raise ValueError(f"{rel} has an unsupported export form")
    return text, names


def _wrap_module(rel: str, sources: dict) -> str:
    """Wrap one module in a function scope and bind its exports to its module variable."""

    def bind(match: re.Match[str]) -> str:
        var = _module_var(_resolve_import(rel, match["spec"], sources))
        return _import_binding(match["clause"], var, rel)

    body, names = _strip_exports(rel, IMPORT_RE.sub(bind, sources[rel]))
    props = ", ".join(
        public if public == local else f"{public}: {local}" for public, local in names
    )
    return f"const {_module_var(rel)} = (() => {{\n{body}\nreturn {{ {props} }};\n}})();\n"


def _texts(web: dict, suffix: str) -> dict:
    """Decode the web files with one suffix as relative path to UTF-8 text."""
    return {rel: blob.decode("utf-8") for rel, blob in web.items() if rel.endswith(suffix)}


def inline_block(graph_json: str, files: dict, web: dict) -> str:
    """
    Build text comments, style tags, the graph, base64 layer files, and one module script.

    Args:
        graph_json: The graph.json text.
        files: Relative layer and channel file path to bytes.
        web: Relative web file path to bytes.

    Returns:
        The HTML that replaces the scripts marker of the page template.

    Raises:
        ValueError: When a web file cannot inline, a module cannot be rewritten, or a text
            file would close its comment early.
    """
    blocked = cannot_inline(web)
    if blocked:
        raise ValueError(f"web files {blocked} cannot be inlined, only {INLINE_SUFFIXES} can")
    sources = _texts(web, ".js")
    script = "".join(_wrap_module(rel, sources) for rel in _module_order(sources))
    tags = [text_comment(rel, text) for rel, text in _texts(web, ".txt").items()]
    # Escaping keeps a closing tag inside the JSON or code from ending its element.
    styles = (text.replace("</style", "<\\/style") for text in _texts(web, ".css").values())
    tags += [f"<style>\n{css}</style>" for css in styles]
    safe_json = GRAPH_LT_RE.sub(r"\\u003c", graph_json)
    safe_script = SCRIPT_END_RE.sub(r"<\\/\1", script).replace("<!--", "<\\!--")
    tags.append(f'<script type="application/json" id="graph">{safe_json}</script>')
    for path, blob in files.items():
        name = posixpath.basename(path).removesuffix(".bin")
        data = base64.b64encode(blob).decode("ascii")
        attrs = f'type="application/octet-stream" id="{name}" data-path="{path}"'
        tags.append(f"<script {attrs}>{data}</script>")
    tags.append(f'<script type="module">\n{safe_script}</script>')
    return "\n".join(tags)
