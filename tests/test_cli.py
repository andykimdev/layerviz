"""
Test the python -m layerviz capture, compare, and serve commands and the package install.

Purpose:

- Check that capture from a factory, checkpoint, and sample matches layerviz.capture.
- Check that compare against random or a checkpoint writes the expected compare bundle.
- Check that serve answers over http and that help and import load neither torch nor spconv.
- Check that the package installs from a path and from git and imports without spconv.

Special Notes:

- The factory and sample loader below are loaded by spec as tests.test_cli:make_model.
- The help and install tests are marked slow since they start subprocesses.
- The install test uses uv when on PATH, else venv and pip.
"""

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
from pathlib import Path

import pytest
import torch
from torch import nn

import layerviz
from layerviz.__main__ import main
from layerviz.comparison import CompareOptions
from layerviz.pipeline import CaptureOptions
from tests.models import image2d_model

REPO = Path(__file__).resolve().parents[1]
MODEL_SPEC = "tests.test_cli:make_model"
SAMPLE_SPEC = "tests.test_cli:make_sample"


def make_model() -> nn.Module:
    """Build the image2d architecture with weights drawn from the global torch seed."""
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, 6, 3, stride=2, padding=1)
    )


def make_sample() -> torch.Tensor:
    """Return the image2d batch and its mirror image as one batch of shape [2, 3, 16, 16]."""
    batch = image2d_model()[1]
    return torch.cat([batch, batch.flip(-1)])


def trained_state() -> dict:
    """Return a state dict whose weights differ from the factory model's."""
    torch.manual_seed(123)
    return {k: torch.randn_like(v) for k, v in make_model().state_dict().items()}


def save_checkpoint(tmp_path: Path, state: dict) -> Path:
    """Save state as a DDP style checkpoint under the model key and return its path."""
    path = tmp_path / "ckpt.pt"
    torch.save({"model": {"module." + k: v for k, v in state.items()}, "step": 7}, path)
    return path


def bundle_files(folder: Path) -> dict:
    """Return graph.json and every layer file of a bundle as relative path to bytes."""
    files = {"graph.json": (folder / "graph.json").read_bytes()}
    for path in sorted((folder / "layers").iterdir()):
        files[f"layers/{path.name}"] = path.read_bytes()
    return files


def trained_module() -> nn.Module:
    """Return the factory model holding the trained_state weights."""
    model = make_model()
    model.load_state_dict(trained_state())
    return model


# 0.003 MB holds 38 positions of 75 bytes, the 2D size, after the one layer file overhead.
CAPTURE_CASES = [
    (["--layer-mb", "0.003", "--exclude", "1", "--seed", "3", "--sample-index", "1"],
     {"layer_mb": 0.003, "exclude": ["1"], "seed": 3, "sample_index": 1}, {"0": 38}),
    (["--total-mb", "0.003", "--include", "1"],
     {"total_mb": 0.003, "include": ["1"]}, {"1": 38}),
]


@pytest.mark.parametrize("flags, options, kept", CAPTURE_CASES)
def test_capture_command_matches_layerviz_capture(tmp_path, flags, options, kept):
    state = trained_state()
    ckpt = save_checkpoint(tmp_path, state)
    out = tmp_path / "cli"
    code = main([
        "capture", "--model", MODEL_SPEC, "--checkpoint", str(ckpt), "--sample", SAMPLE_SPEC,
        "--out", str(out), "--inline-mb", "0", *flags,
    ])
    model = make_model()
    model.load_state_dict(state)
    layerviz.capture(model.eval(), make_sample(), tmp_path / "ref", inline_mb=0, **options)
    graph = json.loads((out / "graph.json").read_text())
    assert code == 0
    assert bundle_files(out) == bundle_files(tmp_path / "ref")
    assert {node["name"]: node["kept"] for node in graph["nodes"] if node["file"]} == kept


def test_compare_command_against_random_writes_a_compare_bundle(tmp_path):
    out = tmp_path / "cmp"
    code = main([
        "compare", "--model", MODEL_SPEC, "--sample", SAMPLE_SPEC, "--against", "random",
        "--labels", "trained", "fresh", "--out", str(out), "--inline-mb", "0",
    ])
    graph = json.loads((out / "graph.json").read_text())
    assert code == 0
    assert graph["meta"]["mode"] == "compare"
    assert graph["meta"]["labels"] == ["trained", "fresh"]
    paired = [node for node in graph["nodes"] if node.get("files")]
    assert [node["name"] for node in paired] == ["0", "1"]
    assert all(node["joint"] and node["cka"] < 1 for node in paired)
    assert all((out / f).is_file() for node in paired for f in node["files"].values())


def assert_matches_trained_compare(out: Path, tmp_path: Path) -> None:
    """Assert that the bundle in out equals layerviz.compare of the seed 0 and trained models."""
    torch.manual_seed(0)
    model_a = make_model().eval()
    options = CompareOptions(("A", "B"), CaptureOptions(), 0)
    ref = layerviz.compare(model_a, trained_module().eval(), make_sample(), tmp_path / "ref", options)
    assert (out / "graph.json").read_bytes() == (ref.parent / "graph.json").read_bytes()
    paired = [node for node in json.loads((out / "graph.json").read_text())["nodes"]
              if node.get("files")]
    assert paired
    assert all((out / f).read_bytes() == (ref.parent / f).read_bytes()
               for node in paired for f in node["files"].values())


def test_compare_command_against_a_checkpoint_matches_layerviz_compare(tmp_path):
    ckpt = save_checkpoint(tmp_path, trained_state())
    out = tmp_path / "cmp"
    code = main([
        "compare", "--model", MODEL_SPEC, "--sample", SAMPLE_SPEC, "--against", str(ckpt),
        "--out", str(out), "--inline-mb", "0",
    ])
    assert code == 0
    assert_matches_trained_compare(out, tmp_path)


def test_compare_command_loads_the_second_checkpoint_from_the_state_key(tmp_path):
    state = trained_state()
    decoy = {k: torch.zeros_like(v) for k, v in state.items()}
    torch.save({"state_dict": decoy, "custom": state}, tmp_path / "other.pt")
    out = tmp_path / "cmp"
    code = main([
        "compare", "--model", MODEL_SPEC, "--sample", SAMPLE_SPEC,
        "--against", str(tmp_path / "other.pt"), "--state-key", "custom",
        "--out", str(out), "--inline-mb", "0",
    ])
    assert code == 0
    assert_matches_trained_compare(out, tmp_path)


def free_server(folder: Path) -> tuple[subprocess.Popen, str]:
    """Start python -m layerviz serve on port 0 and return (process, printed URL)."""
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "layerviz", "serve", str(folder), "--port", "0"],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    return proc, proc.stdout.readline().strip()


def test_serve_command_answers_the_page_and_bundle_files(tmp_path):
    out = tmp_path / "bundle"
    main(["capture", "--model", MODEL_SPEC, "--sample", SAMPLE_SPEC, "--out", str(out),
          "--inline-mb", "0"])
    proc, url = free_server(out)
    try:
        assert url.startswith("http://127.0.0.1:") and url.endswith("/")
        page = urllib.request.urlopen(url + "index.html", timeout=10).read()
        graph = urllib.request.urlopen(url + "graph.json", timeout=10).read()
        layer = urllib.request.urlopen(url + "layers/L0002.bin", timeout=10).read()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    assert page == (out / "index.html").read_bytes()
    assert graph == (out / "graph.json").read_bytes()
    assert layer == (out / "layers" / "L0002.bin").read_bytes()


def test_serve_command_names_a_missing_bundle_folder(tmp_path):
    missing = tmp_path / "no_bundle"
    with pytest.raises(FileNotFoundError, match="bundle folder .*no_bundle"):
        main(["serve", str(missing)])


HELP_CHECK = """
import json, runpy, sys
import layerviz, layerviz.__main__
sys.argv = ["layerviz", "--help"]
try:
    runpy.run_module("layerviz", run_name="__main__")
except SystemExit as exit:
    code = exit.code
loaded = sorted(name for name in ("torch", "spconv") if name in sys.modules)
print(json.dumps({"code": code, "loaded": loaded}))
"""


@pytest.fixture(scope="module")
def help_run() -> tuple[str, dict]:
    """Run import and --help in a fresh interpreter and return (printed text, final report)."""
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    printed = run_checked(sys.executable, "-c", HELP_CHECK, cwd=REPO, env=env)
    return printed, json.loads(printed.strip().splitlines()[-1])


@pytest.mark.slow
@pytest.mark.parametrize("command", ["capture", "compare", "serve"])
def test_help_names_each_subcommand(help_run, command):
    assert command in help_run[0]


@pytest.mark.slow
def test_help_exits_with_code_zero(help_run):
    assert help_run[1]["code"] == 0


@pytest.mark.slow
def test_help_and_import_load_neither_torch_nor_spconv(help_run):
    assert help_run[1]["loaded"] == []


def run_checked(*args: str, **kwargs) -> str:
    """Run a command, raise with its output on failure, and return its stdout."""
    done = subprocess.run(args, capture_output=True, text=True, **kwargs)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def copy_source(tmp_path: Path) -> Path:
    """Copy pyproject.toml and layerviz/ into a temp folder, so builds never touch the repo."""
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(REPO / "pyproject.toml", src)
    shutil.copytree(REPO / "layerviz", src / "layerviz",
                    ignore=shutil.ignore_patterns("__pycache__"))
    return src


def git_source(tmp_path: Path) -> Path:
    """Copy the package source into a fresh git repo and return its folder."""
    src = copy_source(tmp_path)
    git = ["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@t"]
    run_checked(*git, "init", "-q")
    run_checked(*git, "add", ".")
    run_checked(*git, "commit", "-q", "-m", "copy")
    return src


IMPORT_CHECK = (
    "import sys, layerviz, layerviz.__main__; "
    "assert 'spconv' not in sys.modules; print(layerviz.__file__)"
)


def install_commands(venv: Path, target: str) -> list[list[str]]:
    """Return the commands that create venv and install target into it, with uv if present."""
    python = str(venv / "bin" / "python")
    if shutil.which("uv"):
        return [["uv", "venv", "-q", str(venv)],
                ["uv", "pip", "install", "-q", "--no-deps", "--python", python, target]]
    return [[sys.executable, "-m", "venv", str(venv)],
            [python, "-m", "pip", "install", "-q", "--no-deps", target]]


PAGE_SUFFIXES = {".js", ".css", ".html", ".txt"}
PAGE_ASSETS = {
    "": {"index.html"},
    "vendor": {"three.module.min.js", "OrbitControls.js", "LICENSE-three.txt"},
}


def installed_files(web: Path, folder: str) -> set[str]:
    """Return the names of the files directly inside an installed web folder."""
    return {p.name for p in (web / folder).iterdir() if p.is_file()}


@pytest.mark.slow
@pytest.mark.parametrize("source", ["path", "git"])
def test_package_installs_and_imports_without_spconv(tmp_path, source):
    """
    Install from a path or from git, import without spconv, and ship only page assets.

    Notes:
        Needs network access, since uv or pip fetches setuptools to build the package.
    """
    venv = tmp_path / "venv"
    target = str(copy_source(tmp_path)) if source == "path" else f"git+file://{git_source(tmp_path)}"
    for command in install_commands(venv, target):
        run_checked(*command)
    env = {**os.environ, "PYTHONPATH": sysconfig.get_paths()["purelib"]}
    printed = run_checked(str(venv / "bin" / "python"), "-c", IMPORT_CHECK, cwd=tmp_path, env=env)
    assert Path(printed.strip()).is_relative_to(venv)
    web = Path(printed.strip()).parent / "web"
    for folder, required in PAGE_ASSETS.items():
        names = installed_files(web, folder)
        assert required <= names
        assert {Path(n).suffix for n in names} <= PAGE_SUFFIXES
