"""
Test that the layerviz skill meets the skill format limits and matches the real command line.

Purpose:

- Check the SKILL.md name, description, body length, and links against the skill format.
- Check that every python -m layerviz command in SKILL.md uses a real subcommand and flag.
- Check that the three eval scenarios and run_evals.sh exist and name Sonnet and Opus.
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
FENCE = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL)
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
                    commands.append(words[start + 3:])
    return commands


def subcommand_flags() -> dict[str, set[str]]:
    """Return each subcommand of the real parser with the option strings it accepts."""
    actions = build_parser()._actions
    choices = next(a.choices for a in actions if isinstance(a, argparse._SubParsersAction))
    return {name: {flag for action in sub._actions for flag in action.option_strings}
            for name, sub in choices.items()}


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


def test_skill_body_is_short_and_covers_each_step():
    """The body stays under 150 lines and names install, adapter, remote serve, and check."""
    body = skill_body()
    assert len(body.strip("\n").splitlines()) < 150
    for text in ("pip install -e", "pip install git+<repo url>", "layerviz_adapter.py",
                 "build_model(checkpoint=None)", "load_sample()", "ssh -N -L",
                 "(adapter_example.py)", "(check_bundle.py)", "joint switch"):
        assert text in body, text


def test_skill_commands_use_real_subcommands_and_flags():
    """Every python -m layerviz command in SKILL.md parses against the real parser."""
    commands = layerviz_commands(skill_body())
    flags = subcommand_flags()
    assert {words[0] for words in commands} == {"capture", "compare", "serve"}
    for words in commands:
        used = {word.split("=")[0] for word in words if word.startswith("--")}
        assert used <= flags[words[0]], f"{words[0]} has no flag {used - flags[words[0]]}"


def test_skill_names_no_date_or_model_version():
    """The skill body and the eval script stay free of dates and model versions."""
    for text in (skill_body(), (SKILL / "evals" / "run_evals.sh").read_text()):
        assert DATED.search(text) is None, DATED.search(text)


def test_evals_hold_three_scenarios_and_run_sonnet_and_opus():
    """The three scenario prompts exist with checklists and the script runs both models."""
    evals = SKILL / "evals"
    for name in SCENARIOS:
        text = (evals / name).read_text()
        assert "## Prompt" in text and "## Expected outcome" in text, name
        assert "- [ ]" in text, name
    script = (evals / "run_evals.sh").read_text()
    assert re.search(r"MODELS=\(sonnet opus\)", script)
    assert "claude -p" in script and '--model "$model"' in script
    assert all(Path(name).stem in script for name in SCENARIOS)
    assert "runs/" in (SKILL / ".gitignore").read_text().splitlines()


@pytest.fixture
def skill_module(monkeypatch):
    """Return an importer of the skill folder's Python files by module name."""
    monkeypatch.syspath_prepend(str(SKILL))
    return importlib.import_module


@pytest.mark.parametrize("inline_mb", ["16", "0"])
def test_adapter_example_captures_and_passes_check_bundle(tmp_path, skill_module, capsys,
                                                          inline_mb):
    """The example adapter captures through the command line and check_bundle passes it."""
    out = tmp_path / "bundle"
    code = main(["capture", "--model", "adapter_example:build_model", "--sample",
                 "adapter_example:load_sample", "--out", str(out), "--inline-mb", inline_mb])
    assert code == 0
    assert capsys.readouterr().out.strip() == str(out / "index.html")
    assert (out / "graph.json").is_file() == (inline_mb == "0")
    checker = skill_module("check_bundle")
    names = {node["name"] for node in checker.read_graph(out)["nodes"]}
    assert {"stem.0", "layer1.conv1", "layer2.conv2", "fc"} <= names
    assert checker.main([str(out)]) == 0
    assert capsys.readouterr().out == f"{out}: ok, {len(checker.read_graph(out)['nodes'])} nodes\n"


def write_no_page(folder: Path) -> None:
    """Leave the bundle folder empty."""


def write_no_graph(folder: Path) -> None:
    """Write a page with no embedded graph and no graph.json."""
    (folder / "index.html").write_text("<html></html>")


def write_no_nodes(folder: Path) -> None:
    """Write a page and a graph.json that lists no nodes."""
    (folder / "index.html").write_text("<html></html>")
    (folder / "graph.json").write_text(json.dumps({"nodes": []}))


@pytest.mark.parametrize("write, message", [
    (write_no_page, "{folder}: no index.html"),
    (write_no_graph, "{folder} has no graph.json and no graph embedded in index.html"),
    (write_no_nodes, "{folder}: the graph lists no nodes"),
], ids=["no_page", "no_graph", "no_nodes"])
def test_check_bundle_reports_broken_bundle(tmp_path, skill_module, capsys, write, message):
    """check_bundle names the problem of a broken bundle and exits 1."""
    checker = skill_module("check_bundle")
    folder = tmp_path / "bundle"
    folder.mkdir()
    write(folder)
    assert checker.main([str(folder)]) == 1
    assert capsys.readouterr().out.splitlines() == [message.format(folder=folder)]


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
    for name, tensor in saved.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    assert not loaded.training


def test_adapter_example_sample_is_one_image(skill_module):
    """load_sample returns one deterministic [1, 3, 64, 64] image in [0, 1]."""
    adapter = skill_module("adapter_example")
    sample = adapter.load_sample()
    assert sample.shape == (1, 3, 64, 64)
    assert torch.equal(sample, adapter.load_sample())
    assert sample.min() >= 0 and sample.max() <= 1
