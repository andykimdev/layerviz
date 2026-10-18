"""
Test the tiny CPU models and the fake sparse tensor shared by every layerviz test.

Purpose:

- Pin the exact output shape of each data shape builder.
- Pin that builders are deterministic and that FakeSparseTensor keeps its structure.
"""

import pytest
import torch

from tests.models import (
    ALL_SHAPES,
    FakeSparseTensor,
    SparseDownsample,
    graph_model,
    sparse_model,
)

EXPECTED_OUTPUT_SHAPES = {
    "image2d": (1, 6, 8, 8),
    "volume3d": (1, 6, 4, 4, 4),
    "tokens": (1, 12, 16),
    "vector": (4, 5),
    "points": (1, 20, 8),
    "graph": (1, 5, 8, 8),
}


SPARSE_FIRST_INPUT_ROWS = [
    11, 17, 0, 6, 9, 23, 8, 25, 3, 26, 18, 7,
    14, 28, 4, 12, 10, 1, 2, 13, 15, 27, 24, 22,
]  # fmt: skip
SPARSE_OUTPUT_INDICES = [
    [0, 0, 0, 3], [0, 0, 1, 2], [0, 0, 2, 2], [0, 0, 2, 3],
    [0, 0, 3, 0], [0, 0, 3, 2], [0, 1, 0, 0], [0, 1, 0, 2],
    [0, 1, 0, 3], [0, 1, 1, 0], [0, 1, 1, 1], [0, 2, 0, 1],
    [0, 2, 0, 3], [0, 2, 1, 0], [0, 2, 1, 2], [0, 2, 1, 3],
    [0, 2, 2, 1], [0, 2, 2, 2], [0, 2, 2, 3], [0, 2, 3, 3],
    [0, 3, 0, 1], [0, 3, 1, 0], [0, 3, 2, 2], [0, 3, 3, 3],
]  # fmt: skip


def run_model(model, batch):
    """
    Run a model on a batch without gradients.

    Args:
        model: Module returned by a builder.
        batch: Input returned by the same builder.

    Returns:
        The model output.
    """
    with torch.no_grad():
        return model(batch)


def tensors_of(output):
    """
    List the tensors that define an output, so two outputs can be compared.

    Args:
        output: A tensor or a FakeSparseTensor.

    Returns:
        A list of tensors.
    """
    if isinstance(output, FakeSparseTensor):
        return [output.features, output.indices]
    return [output]


def make_sparse_tensor():
    """
    Build a two voxel FakeSparseTensor with known indices and features.

    Returns:
        A FakeSparseTensor with spatial_shape [4, 4, 4] and batch_size 1.
    """
    indices = torch.tensor([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=torch.int32)
    features = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return FakeSparseTensor(features, indices, [4, 4, 4], 1)


@pytest.mark.parametrize("name", sorted(EXPECTED_OUTPUT_SHAPES))
def test_builder_output_has_expected_shape(name):
    model, batch = ALL_SHAPES[name]()

    output = run_model(model, batch)

    assert tuple(output.shape) == EXPECTED_OUTPUT_SHAPES[name]


def test_all_shapes_lists_every_builder():
    assert set(ALL_SHAPES) == set(EXPECTED_OUTPUT_SHAPES) | {"sparse"}


def test_sparse_model_output_matches_exact_voxels_and_features():
    model, batch = sparse_model()

    output = run_model(model, batch)

    assert batch.features.shape == (30, 4)
    assert batch.indices.shape == (30, 4)
    assert batch.spatial_shape == [8, 8, 8]
    assert output.spatial_shape == [4, 4, 4]
    assert output.batch_size == 1
    assert output.indices.tolist() == SPARSE_OUTPUT_INDICES
    first_rows = batch.features[SPARSE_FIRST_INPUT_ROWS]
    expected = model[2].linear(model[0].linear(first_rows))
    assert output.features.shape == (24, 6)
    assert torch.allclose(output.features, expected)


def test_downsample_floors_indices_and_keeps_first_row_per_cell():
    rows = [[0, 1, 2, 3], [0, 1, 3, 2], [0, 3, 2, 1]]
    indices = torch.tensor(rows, dtype=torch.int32)
    features = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    sparse = FakeSparseTensor(features, indices, [4, 4, 4], 1)

    output = SparseDownsample()(sparse)

    assert output.indices.tolist() == [[0, 0, 1, 1], [0, 1, 1, 0]]
    assert output.spatial_shape == [2, 2, 2]
    assert output.batch_size == 1
    assert torch.equal(output.features, features[[0, 2]])


def test_graph_model_runs_negative_branch_under_seed_zero():
    model, batch = graph_model()
    calls = []
    model.extra_positive.register_forward_hook(lambda *_: calls.append("positive"))
    model.extra_negative.register_forward_hook(lambda *_: calls.append("negative"))

    run_model(model, batch)

    assert calls == ["negative"]


@pytest.mark.parametrize("name", sorted(ALL_SHAPES))
def test_builder_is_deterministic(name):
    first = run_model(*ALL_SHAPES[name]())
    second = run_model(*ALL_SHAPES[name]())

    first_tensors, second_tensors = tensors_of(first), tensors_of(second)
    assert len(first_tensors) == len(second_tensors)
    for left, right in zip(first_tensors, second_tensors):
        assert torch.equal(left, right)


def test_replace_feature_keeps_structure_and_swaps_features():
    original = make_sparse_tensor()
    new_features = torch.ones(2, 5)

    replaced = original.replace_feature(new_features)

    assert replaced is not original
    assert replaced.features is new_features
    assert torch.equal(replaced.indices, original.indices)
    assert replaced.spatial_shape == [4, 4, 4]
    assert replaced.batch_size == 1
    assert original.features.shape == (2, 3)
