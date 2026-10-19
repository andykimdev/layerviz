"""
Cross check the browser similarity module against a NumPy reference.

Purpose:

- Build seeded float16 scores and compute reference cosine similarity and percentile ranks.
- Write them as a JSON fixture and run the Node similarity tests against it.

Special Notes:

- The Node script decodes the fixture scores with decodeFloat16 from format.js, so the
  check covers the decoder the page uses.

- Fails when node is not on PATH, since the full check runs node anyway.
- Uses no scipy so ranks follow the module definition.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
JS_TEST = REPO_ROOT / "tests" / "js" / "similarity.test.js"


def reference_cosine(scores: np.ndarray, index: int) -> np.ndarray:
    """
    Compute cosine similarity between one row and every row, with 0 for zero norm rows.

    Args:
        scores: Array of shape (n, k).
        index: Row to compare against.

    Returns:
        Float64 array of length n.
    """
    data = scores.astype(np.float64)
    norms = np.linalg.norm(data, axis=1)
    denom = norms * norms[index]
    dots = data @ data[index]
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)


def reference_percentile_rank(values: np.ndarray) -> np.ndarray:
    """
    Rank each value among all values with average ranks for ties, scaled to 0..1.

    Args:
        values: One dimensional array of length n.

    Returns:
        Float64 array of length n, all zeros when n is 1.
    """
    n = len(values)
    if n < 2:
        return np.zeros(n)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(n)
    start = 0
    while start < n:
        end = start
        while end + 1 < n and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2
        start = end + 1
    return ranks / (n - 1)


def build_fixture(n: int, k: int, index: int, seed: int) -> dict:
    """
    Build seeded float16 scores and their reference similarity and ranks.

    Args:
        n: Number of positions.
        k: Number of score columns.
        index: Clicked position.
        seed: Random seed.

    Returns:
        Dict with the raw uint16 scores, n, k, index, similarity, and ranks.
    """
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((n, k)).astype(np.float16)
    similarity = reference_cosine(scores, index)
    return {
        "scores_uint16": scores.view(np.uint16).ravel().tolist(),
        "n": n,
        "k": k,
        "index": index,
        "similarity": similarity.tolist(),
        "ranks": reference_percentile_rank(similarity).tolist(),
    }


def test_reference_percentile_rank_averages_ties() -> None:
    ranks = reference_percentile_rank(np.array([30.0, 20.0, 10.0, 20.0]))
    np.testing.assert_array_equal(ranks, [1.0, 0.5, 0.0, 0.5])


def test_reference_cosine_gives_zero_for_zero_norm_row() -> None:
    sim = reference_cosine(np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]), 1)
    np.testing.assert_allclose(sim, [0.0, 1.0, 1.0])


def test_browser_similarity_matches_numpy_reference(tmp_path: Path) -> None:
    node = shutil.which("node")
    assert node, "node must be on PATH to run tests/js/similarity.test.js"
    fixture_path = tmp_path / "similarity_fixture.json"
    fixture_path.write_text(json.dumps(build_fixture(n=200, k=32, index=7, seed=0)))
    env = {**os.environ, "LAYERVIZ_FIXTURE": str(fixture_path)}

    result = subprocess.run(
        [node, "--test", str(JS_TEST)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    # A fixture that failed to load would skip silently, so require zero skips.
    assert re.search(r"skipped 0\b", result.stdout), result.stdout
