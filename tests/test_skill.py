"""
Test that the layerviz skill meets the skill format limits and matches the real command line.

Purpose:

- Check the SKILL.md name, description, body length, and links against the skill format.
- Parse every python -m layerviz command and backticked flag of the docs with the real parser.
- Check that the three eval scenarios exist and run_evals.sh runs and grades three models.
- Check that the adapter example captures through the command line and passes check_bundle.py.

Special Notes:

- The evals themselves call claude and are run by hand, so no test runs run_evals.sh.
"""

import argparse
import importlib
import json
import re
import shlex
from pathlib import Path

import pytest
import torch

from layerviz.__main__ import build_parser, main

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skill" / "layerviz"
SCENARIOS = ("cnn_2d.md", "spconv_remote.md", "training_snapshots.md")
DOCUMENTED = ("SKILL.md", "README.md", *SCENARIOS)
FENCE = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL)
ANY_FENCE = re.compile(r"```.*?```", re.DOTALL)
BACKTICKED = re.compile(r"`([^`\n]+)`")
FLAG = re.compile(r"--[a-z][a-z0-9-]*")
# A redirect, pipe, or background word ends the arguments the shell hands to layerviz.
SHELL_OPERATOR = re.compile(r"^\d*[<>|&;]")
# The skill format allows a name of lowercase words joined by hyphens.
NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s#]+)")
# A four digit year, or a model family followed by a version number, names a moment in time.
DATED = re.compile(r"\b(19|20)\d\d\b|\b(sonnet|opus|haiku|claude)[- ]?\d", re.IGNORECASE)


def skill_body() -> str:
    """Return SKILL.md without its frontmatter."""
    return (SKILL / "SKILL.md").read_text().split("---", 2)[2]


def skill_frontmatter() -> dict[str, str]:
    """Return the top level fields of the SKILL.md frontmatter."""
    header = (SKILL / "SKILL.md").read_text().split("---", 2)[1]
    fields = {}
    for line in header.strip().splitlines():
        key, sep, value = line.partition(":")
        if sep and not line.startswith((" ", "\t")):
            fields[key.strip()] = value.strip()
    return fields


def layerviz_commands(markdown: str) -> list[list[str]]:
    """Return the arguments after python -m layerviz of every command in bash fences."""
    commands = []
    for block in FENCE.findall(markdown):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line, comments=True)
            for start in range(len(words) - 2):
                if words[start:start + 3] == ["python", "-m", "layerviz"]:
                    commands.append(layerviz_arguments(words[start + 3:]))
    return commands


def layerviz_arguments(words: list[str]) -> list[str]:
    """Return the words up to the first shell redirect, pipe, or background operator."""
    for index, word in enumerate(words):
        if SHELL_OPERATOR.match(word):
            return words[:index]
    return words


def documented_text(name: str) -> str:
    """Return the text of SKILL.md, README.md, or one eval scenario by file name."""
    if name == "SKILL.md":
        return skill_body()
    return (ROOT / "README.md" if name == "README.md" else SKILL / "evals" / name).read_text()


def prose_flags(markdown: str) -> set[str]:
    """Return the flags outside code fences, in backticks or on lines naming python -m layerviz."""
    prose = ANY_FENCE.sub("", markdown)
    spans = BACKTICKED.findall(prose)
    spans += [line for line in prose.splitlines() if "python -m layerviz" in line]
    return {flag for span in spans for flag in FLAG.findall(span)}


def parse_documented(words: list[str]) -> argparse.Namespace:
    """Parse one documented command with the real parser and fail the test when it errors."""
    try:
        return build_parser().parse_args(words)
    except SystemExit:
        pytest.fail(f"python -m layerviz {shlex.join(words)} does not parse")


def subcommand_flags() -> dict[str, set[str]]:
    """Return each subcommand of the real parser with the option strings it accepts."""
    actions = build_parser()._actions
    choices = next(a.choices for a in actions if isinstance(a, argparse._SubParsersAction))
    return {name: {flag for action in sub._actions for flag in action.option_strings}
            for name, sub in choices.items()}


def misspelled_flags(words: list[str]) -> set[str]:
    """Return the flags of one documented command that its subcommand does not accept."""
    used = {word.split("=")[0] for word in words if word.startswith("--")}
    return used - subcommand_flags()[words[0]]


def test_skill_meets_the_skill_format_limits():
    """The name, description, body length, and local links fit the skill format."""
    fields = skill_frontmatter()
    assert NAME.match(fields["name"]) and len(fields["name"]) <= 64
    assert 0 < len(fields["description"]) <= 1024
    body = skill_body()
    assert len(body.strip("\n").splitlines()) <= 500
    links = [t for t in LINK.findall(body) if not t.startswith(("http://", "https://"))]
    assert links
    missing = [target for target in links if not (SKILL / target).exists()]
    assert missing == []


def test_skill_body_is_under_150_lines():
    """The body stays under 150 lines."""
    assert len(skill_body().strip("\n").splitlines()) < 150


@pytest.mark.parametrize("text", ["layerviz_adapter.py", "build_model(checkpoint=None)",
                                  "load_sample()", "ssh -N -L", "(adapter_example.py)",
                                  "(check_bundle.py)"])
def test_skill_body_covers_each_step(text):
    """The body names the adapter, remote serve, and check steps."""
    assert text in skill_body()


@pytest.mark.parametrize("name", DOCUMENTED)
def test_documented_commands_parse_against_the_real_parser(name):
    """Every python -m layerviz command of a doc parses and spells each flag exactly."""
    commands = layerviz_commands(documented_text(name))
    assert [parse_documented(words).command for words in commands] == [w[0] for w in commands]
    assert [misspelled_flags(words) for words in commands] == [set() for _ in commands]


@pytest.mark.parametrize("name, expected", [("SKILL.md", set(subcommand_flags())),
                                             ("README.md", {"capture", "serve"})])
def test_documented_commands_cover_the_expected_subcommands(name, expected):
    """SKILL.md shows every subcommand of the real parser and README.md shows two."""
    assert {words[0] for words in layerviz_commands(documented_text(name))} == expected


@pytest.mark.parametrize("name, taught", [
    ("SKILL.md", {"--include", "--state-key", "--inline-mb"}),
    ("spconv_remote.md", {"--checkpoint", "--port"}),
    ("README.md", set()), ("cnn_2d.md", set()), ("training_snapshots.md", set()),
])
def test_documented_prose_flags_exist_on_the_parser(name, taught):
    """Every flag a doc names in prose is a real option, and the docs name the flags taught."""
    known = set().union(*subcommand_flags().values())
    flags = prose_flags(documented_text(name))
    assert flags - known == set()
    assert taught <= flags


@pytest.mark.parametrize("name", ["SKILL.md", "run_evals.sh"])
def test_skill_names_no_date_or_model_version(name):
    """The skill body and the eval script stay free of dates and model versions."""
    assert DATED.search(documented_text(name)) is None


@pytest.mark.parametrize("name", SCENARIOS)
def test_eval_scenario_holds_a_prompt_and_a_checklist(name):
    """Each scenario file has a prompt, an expected outcome, and a checklist."""
    text = documented_text(name)
    assert "## Prompt" in text and "## Expected outcome" in text and "- [ ]" in text


def test_eval_script_runs_and_grades_three_models():
    """The script runs and grades three models over every scenario."""
    script = documented_text("run_evals.sh")
    assert re.search(r"MODELS=\(haiku sonnet opus\)", script)
    assert "claude -p" in script and '--model "$model"' in script
    assert 'SUMMARY="$EVALS_DIR/runs/$STAMP/summary.txt"' in script
    assert '"$PYTHON" "$SKILL_DIR/check_bundle.py" "${bundles[@]}"' in script
    assert re.search(r'grade_run "\$run" "\$scenario" "\$model"\n', script)
    assert '"$verdict" "$scenario" "$model" >> "$SUMMARY"' in script
    assert all(Path(name).stem in script for name in SCENARIOS)
    assert "runs/" in (SKILL / ".gitignore").read_text().splitlines()


@pytest.fixture
def skill_module(monkeypatch):
    """Return an importer of the skill folder's Python files by module name."""
    monkeypatch.syspath_prepend(str(SKILL))
    return importlib.import_module


@pytest.mark.parametrize("inline_mb", ["16", "0"])
def test_adapter_example_captures_and_passes_check(tmp_path, skill_module, capsys, inline_mb):
    """The example adapter captures through the command line and check_bundle passes it."""
    out = tmp_path / "bundle"
    capture_bundle(out, inline_mb)
    assert capsys.readouterr().out.strip() == str(out / "index.html")
    assert (out / "graph.json").is_file() == (inline_mb == "0")
    checker = skill_module("check_bundle")
    graph = checker.read_graph(out)
    names = {node["name"] for node in graph["nodes"]}
    assert {"stem.0", "layer1.conv1", "layer2.conv2", "fc"} <= names
    stored = [node for node in graph["nodes"] if node["file"]]
    assert {node["name"] for node in stored} >= {"stem.0", "fc"}
    assert checker.main([str(out)]) == 0
    expected = f"{out}: ok, {len(graph['nodes'])} nodes, {len(stored)} layers\n"
    assert capsys.readouterr().out == expected


def write_no_page(folder: Path) -> None:
    """Leave the bundle folder empty."""


def write_no_graph(folder: Path) -> None:
    """Write a page with no embedded graph and no graph.json."""
    (folder / "index.html").write_text("<html></html>")


def write_no_nodes(folder: Path) -> None:
    """Write a page and a graph.json that lists no nodes."""
    (folder / "index.html").write_text("<html></html>")
    (folder / "graph.json").write_text(json.dumps({"nodes": []}))


def write_channels_only(folder: Path) -> None:
    """Write a page and a graph whose only node names a channel file and no layer file."""
    (folder / "index.html").write_text("<html></html>")
    (folder / "L0000.ch.bin").write_bytes(b"\x01")
    node = {"id": "L0000", "name": "fc", "file": None, "channels": "L0000.ch.bin"}
    (folder / "graph.json").write_text(json.dumps({"nodes": [node]}))


@pytest.mark.parametrize("write, message", [
    (write_no_page, "{folder}: no index.html"),
    (write_no_graph, "{folder} has no graph.json and no graph embedded in index.html"),
    (write_no_nodes, "{folder}: the graph lists no nodes"),
    (write_channels_only, "{folder}: no node stores a layer"),
], ids=["no_page", "no_graph", "no_nodes", "channels_only"])
def test_check_bundle_reports_broken_bundle(tmp_path, skill_module, capsys, write, message):
    """check_bundle names the problem of a broken bundle and exits 1."""
    checker = skill_module("check_bundle")
    folder = tmp_path / "bundle"
    folder.mkdir()
    write(folder)
    assert checker.main([str(folder)]) == 1
    assert capsys.readouterr().out.splitlines() == [message.format(folder=folder)]


def capture_bundle(out: Path, inline_mb: str, *flags: str) -> None:
    """Capture the example adapter into out, one page when inline_mb is not 0."""
    assert main(["capture", "--model", "adapter_example:build_model", "--sample",
                 "adapter_example:load_sample", "--out", str(out), "--inline-mb", inline_mb,
                 *flags]) == 0


def remove_layer_file(folder: Path) -> str:
    """Delete one layer file of a multi file bundle and return the expected problem."""
    (folder / "layers" / "L0003.bin").unlink()
    return "node L0003 names layers/L0003.bin, which is missing or empty"


def empty_channel_file(folder: Path) -> str:
    """Truncate one channel file of a multi file bundle and return the expected problem."""
    (folder / "layers" / "L0004.ch.bin").write_bytes(b"")
    return "node L0004 names layers/L0004.ch.bin, which is missing or empty"


def remove_page_script(folder: Path) -> str:
    """Delete the app.js that index.html loads and return the expected problem."""
    (folder / "app.js").unlink()
    return "index.html loads app.js, which is missing or empty"


def make_nodes_a_string(folder: Path) -> str:
    """Replace the graph's node list with a string and return the expected problem."""
    graph = json.loads((folder / "graph.json").read_text())
    (folder / "graph.json").write_text(json.dumps({**graph, "nodes": "L0000"}))
    return "the graph nodes are not a list"


def drop_node_name(folder: Path) -> str:
    """Remove the name of the first node and return the expected problem."""
    graph = json.loads((folder / "graph.json").read_text())
    del graph["nodes"][0]["name"]
    (folder / "graph.json").write_text(json.dumps(graph))
    return "node 0 is not an object with an id and a name"


@pytest.mark.parametrize("breaker", [
    remove_layer_file, empty_channel_file, remove_page_script, make_nodes_a_string, drop_node_name,
], ids=["missing_layer", "empty_channels", "missing_script", "nodes_string", "nameless_node"])
def test_check_bundle_reports_broken_multi_file_bundle(tmp_path, skill_module, capsys, breaker):
    """check_bundle exits 1 and names the one problem broken into a multi file bundle."""
    out = tmp_path / "bundle"
    capture_bundle(out, "0")
    message = breaker(out)
    capsys.readouterr()
    assert skill_module("check_bundle").main([str(out)]) == 1
    assert capsys.readouterr().out.splitlines() == [f"{out}: {message}"]


def test_check_bundle_reports_missing_inline_layer(tmp_path, skill_module, capsys):
    """check_bundle exits 1 when a one page bundle lacks the element of a stored layer."""
    out = tmp_path / "bundle"
    capture_bundle(out, "16")
    page = (out / "index.html").read_text()
    element = re.compile(r'<script [^>]*data-path="layers/L0003\.bin">[^<]*</script>')
    assert len(element.findall(page)) == 1
    (out / "index.html").write_text(element.sub("", page))
    capsys.readouterr()
    assert skill_module("check_bundle").main([str(out)]) == 1
    expected = f"{out}: node L0003 names layers/L0003.bin, which is missing or empty"
    assert capsys.readouterr().out.splitlines() == [expected]


@pytest.mark.parametrize("inline_mb", ["16", "0"])
def test_check_bundle_reports_bundle_with_no_layers(tmp_path, skill_module, capsys, inline_mb):
    """check_bundle exits 1 for a bundle whose include pattern matched no layer."""
    out = tmp_path / "bundle"
    capture_bundle(out, inline_mb, "--include", "nomatch*")
    capsys.readouterr()
    assert skill_module("check_bundle").main([str(out)]) == 1
    assert capsys.readouterr().out.splitlines() == [f"{out}: no node stores a layer"]


def test_check_bundle_reports_missing_side_b_of_a_compare_bundle(tmp_path, skill_module, capsys):
    """check_bundle exits 1 when a comparison bundle lacks the b file of a files pair."""
    out = tmp_path / "bundle"
    assert main(["compare", "--model", "adapter_example:build_model", "--sample",
                 "adapter_example:load_sample", "--against", "random", "--out", str(out),
                 "--inline-mb", "0"]) == 0
    graph = skill_module("check_bundle").read_graph(out)
    node = next(node for node in graph["nodes"] if node["id"] == "L0003")
    assert node["files"] == {"a": "layers/L0003a.bin", "b": "layers/L0003b.bin"}
    (out / "layers" / "L0003b.bin").unlink()
    capsys.readouterr()
    assert skill_module("check_bundle").main([str(out)]) == 1
    expected = f"{out}: node L0003 names layers/L0003b.bin, which is missing or empty"
    assert capsys.readouterr().out.splitlines() == [expected]


def test_summary_counts_only_nodes_with_a_file_or_files_pair(skill_module):
    """summary counts a node as a layer only when it names file or files, not channels alone."""
    nodes = [{"id": "L0", "name": "a", "file": "L0.bin", "channels": "L0.ch.bin"},
             {"id": "L1", "name": "b", "file": None, "files": {"a": "L1a.bin", "b": "L1b.bin"}},
             {"id": "L2", "name": "c", "file": None, "channels": "L2.ch.bin"},
             {"id": "L3", "name": "d", "file": None, "files": None}]
    assert skill_module("check_bundle").summary(nodes) == "ok, 4 nodes, 2 layers"


def test_check_bundle_fails_with_no_folders(skill_module):
    """check_bundle exits 1 when no bundle folder is named."""
    assert skill_module("check_bundle").main([]) == 1


def test_adapter_example_loads_its_own_checkpoint(tmp_path, skill_module):
    """build_model(checkpoint) restores the saved weights in eval mode, as compare needs."""
    adapter = skill_module("adapter_example")
    saved = adapter.build_model()
    with torch.no_grad():
        for tensor in saved.state_dict().values():
            tensor.add_(1)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": saved.state_dict()}, path)
    loaded = adapter.build_model(checkpoint=path)
    restored = {n: torch.equal(loaded.state_dict()[n], t) for n, t in saved.state_dict().items()}
    assert restored == dict.fromkeys(saved.state_dict(), True)
    assert not loaded.training


def test_adapter_example_imports_public_load_state(skill_module):
    """layerviz exposes load_state, and the adapter example imports it from layerviz."""
    import layerviz
    from layerviz.loading import load_state
    assert layerviz.load_state is load_state and "load_state" in layerviz.__all__
    assert skill_module("adapter_example").load_state is load_state
    assert "from layerviz import load_state" in (SKILL / "adapter_example.py").read_text()


def test_adapter_example_sample_is_one_image(skill_module):
    """load_sample returns one deterministic [1, 3, 64, 64] image in [0, 1]."""
    adapter = skill_module("adapter_example")
    sample = adapter.load_sample()
    assert sample.shape == (1, 3, 64, 64)
    assert torch.equal(sample, adapter.load_sample())
    assert sample.min() >= 0 and sample.max() <= 1
