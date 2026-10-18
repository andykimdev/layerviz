"""
Schedule training snapshots into numbered step folders and link them from one index page.

Purpose:

- Decide which steps are due and name the folder each step writes.
- Write out/index.html with a link to every step folder's own index.html.

Special Notes:

- Pure except write_index. Only rank 0 of a distributed run counts as the main process.
"""

import html
import re
from pathlib import Path

import torch

STEP_RE = re.compile(r"step_\d+")


def due(step: int, every: int) -> bool:
    """
    Tell whether a training step should be captured.

    Args:
        step: The current step, 0 or more.
        every: The capture interval in steps, 1 or more.

    Returns:
        True when step is a multiple of every, step 0 included.

    Raises:
        ValueError: When every is less than 1.
    """
    if every < 1:
        raise ValueError(f"every must be 1 or more to schedule snapshots, got every={every}")
    return step % every == 0


def step_dir(out: Path, step: int) -> Path:
    """
    Name the folder one step writes its bundle into.

    Args:
        out: The snapshot root folder.
        step: The training step.

    Returns:
        out / step_<step padded to six digits>.
    """
    return Path(out) / f"step_{step:06d}"


def is_main_process() -> bool:
    """
    Tell whether this process should capture, which is every process but a nonzero rank.

    Returns:
        False only when torch.distributed is initialized and this rank is above 0.
    """
    dist = torch.distributed
    return not (dist.is_available() and dist.is_initialized() and dist.get_rank() > 0)


def list_steps(out: Path) -> list[Path]:
    """
    List the step folders under out in step order.

    Args:
        out: The snapshot root folder, which may not exist yet.

    Returns:
        The step folders sorted by step number, so step_1000000 follows step_200000.
    """
    root = Path(out)
    if not root.is_dir():
        return []
    steps = [p for p in root.iterdir() if p.is_dir() and STEP_RE.fullmatch(p.name)]
    return sorted(steps, key=lambda p: int(p.name[len("step_"):]))


def index_page(steps: list[str]) -> str:
    """
    Render the snapshot index page that links each step folder.

    Args:
        steps: Step folder names in display order.

    Returns:
        The HTML text of the page.
    """
    items = "".join(
        f'<li><a href="{html.escape(name)}/index.html">{html.escape(name)}</a></li>\n'
        for name in steps
    )
    return (
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"><title>layerviz snapshots"
        f"</title></head>\n<body><h1>layerviz snapshots</h1>\n<ul>\n{items}</ul></body></html>\n"
    )


def write_index(out: Path) -> Path:
    """
    Write out/index.html listing every step folder as a link to its index.html.

    Args:
        out: The snapshot root folder, created when missing.

    Returns:
        Path of the written index page.
    """
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    index = root / "index.html"
    index.write_text(index_page([p.name for p in list_steps(root)]), encoding="utf-8")
    return index
