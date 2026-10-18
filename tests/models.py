"""
Build tiny CPU models, matching samples, and a fake sparse tensor for layerviz tests.

Purpose:

- Provide one builder per data shape, each returning (model, batch) deterministically.
- Provide FakeSparseTensor, a stand in for spconv SparseConvTensor without spconv.

Special Notes:

- Every builder seeds torch with 0 first and returns the model in eval mode.
- ALL_SHAPES maps a shape name to its builder for parametrized tests.
"""

from typing import Any

import torch
from torch import nn

SEED = 0
ModelPair = tuple[nn.Module, Any]


class FakeSparseTensor:
    """
    Hold sparse voxel features and indices with the attributes of SparseConvTensor.

    """

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: list[int],
        batch_size: int,
    ) -> None:
        """
        Store the sparse tensor fields.

        Args:
            features: Float tensor [N, C] with one row per active voxel.
            indices: Int tensor [N, 4] with columns (batch, z, y, x).
            spatial_shape: List of 3 ints, the voxel grid size.
            batch_size: Number of samples in the batch.
        """
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size

    def replace_feature(self, features: torch.Tensor) -> "FakeSparseTensor":
        """
        Return a new tensor with other features and the same structure.

        Args:
            features: Float tensor [N, C2] that replaces the features.

        Returns:
            A new FakeSparseTensor sharing indices and spatial_shape with this one.
        """
        return FakeSparseTensor(
            features, self.indices, self.spatial_shape, self.batch_size
        )


class SparseLinear(nn.Module):
    """
    Apply a Linear layer to the features of a FakeSparseTensor.

    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Create the Linear layer.

        Args:
            in_features: Channel count C of the input features.
            out_features: Channel count of the output features.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, sparse: FakeSparseTensor) -> FakeSparseTensor:
        """
        Map the features through the Linear layer.

        Args:
            sparse: FakeSparseTensor with features [N, in_features].

        Returns:
            A FakeSparseTensor with features [N, out_features].
        """
        return sparse.replace_feature(self.linear(sparse.features))


class SparseDownsample(nn.Module):
    """
    Halve a FakeSparseTensor grid by flooring indices by 2 and keeping unique rows.

    """

    def forward(self, sparse: FakeSparseTensor) -> FakeSparseTensor:
        """
        Merge voxels that share a coarse cell, keeping the first voxel's features.

        Args:
            sparse: FakeSparseTensor to downsample.

        Returns:
            A FakeSparseTensor with half the spatial_shape and unique indices.
        """
        coarse = sparse.indices.clone()
        coarse[:, 1:] = torch.div(coarse[:, 1:], 2, rounding_mode="floor")
        unique, inverse = torch.unique(coarse, dim=0, return_inverse=True)
        rows = torch.arange(len(coarse))
        first = torch.full((len(unique),), len(coarse), dtype=torch.long)
        first = first.scatter_reduce(0, inverse, rows, reduce="amin")
        return FakeSparseTensor(
            sparse.features[first],
            unique,
            [size // 2 for size in sparse.spatial_shape],
            sparse.batch_size,
        )


class PointsNet(nn.Module):
    """
    Pass the features of a (coords, feats) point batch through two Linear layers.

    """

    def __init__(self) -> None:
        """
        Create the two Linear layers.
        """
        super().__init__()
        self.first = nn.Linear(4, 16)
        self.second = nn.Linear(16, 8)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Unpack the batch and map feats to 8 channels.

        Args:
            batch: Tuple (coords [1, 20, 3], feats [1, 20, 4]).

        Returns:
            Tensor [1, 20, 8].
        """
        _coords, feats = batch
        return self.second(self.first(feats))


class GraphNet(nn.Module):
    """
    Combine a residual block, a down and up path, a concat, and a data dependent branch.

    """

    def __init__(self) -> None:
        """
        Create the convolution layers.
        """
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.res_conv = nn.Conv2d(4, 4, 3, padding=1)
        self.down = nn.Conv2d(4, 4, 3, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(4, 4, 2, stride=2)
        self.head = nn.Conv2d(8, 5, 1)
        self.extra_positive = nn.Conv2d(5, 5, 1)
        self.extra_negative = nn.Conv2d(5, 5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the branching graph.

        Args:
            x: Tensor [1, 3, 8, 8].

        Returns:
            Tensor [1, 5, 8, 8].
        """
        stem_out = self.stem(x)
        res_out = stem_out + self.res_conv(stem_out)
        up_out = self.up(self.down(res_out))
        out = self.head(torch.cat([stem_out, up_out], dim=1))
        if out.mean() > 0:
            return self.extra_positive(out)
        return self.extra_negative(out)


def as_eval_pair(model: nn.Module, batch: Any) -> ModelPair:
    """
    Put a model in eval mode and pair it with its batch.

    Args:
        model: Freshly built module.
        batch: Input matching the model.

    Returns:
        The tuple (model, batch).
    """
    return model.eval(), batch


def image2d_model() -> ModelPair:
    """
    Build a two layer Conv2d model with a stride 2 second layer.

    Returns:
        (model, batch) with batch [1, 3, 16, 16] and output [1, 6, 8, 8].
    """
    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, 6, 3, stride=2, padding=1)
    )
    return as_eval_pair(model, torch.randn(1, 3, 16, 16))


def volume3d_model() -> ModelPair:
    """
    Build a Conv3d model followed by a stride 2 Conv3d.

    Returns:
        (model, batch) with batch [1, 2, 8, 8, 8] and output [1, 6, 4, 4, 4].
    """
    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Conv3d(2, 4, 3, padding=1), nn.Conv3d(4, 6, 3, stride=2, padding=1)
    )
    return as_eval_pair(model, torch.randn(1, 2, 8, 8, 8))


def wide_volume3d_model() -> ModelPair:
    """
    Build one stride 2 Conv3d with 32 channels, so its layer stores all 32 scores.

    Returns:
        (model, batch) with batch [1, 2, 8, 8, 8] and output [1, 32, 4, 4, 4].
    """
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Conv3d(2, 32, 3, stride=2, padding=1))
    return as_eval_pair(model, torch.randn(1, 2, 8, 8, 8))


def tokens_model() -> ModelPair:
    """
    Build a Linear layer followed by a TransformerEncoderLayer.

    Returns:
        (model, batch) with batch [1, 12, 8] and output [1, 12, 16].
    """
    torch.manual_seed(SEED)
    encoder = nn.TransformerEncoderLayer(
        d_model=16, nhead=2, dim_feedforward=32, dropout=0.0, batch_first=True
    )
    return as_eval_pair(nn.Sequential(nn.Linear(8, 16), encoder), torch.randn(1, 12, 8))


def vector_model() -> ModelPair:
    """
    Build two Linear layers for a batch of vectors.

    Returns:
        (model, batch) with batch [4, 10] and output [4, 5].
    """
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(10, 16), nn.Linear(16, 5))
    return as_eval_pair(model, torch.randn(4, 10))


def points_model() -> ModelPair:
    """
    Build a point cloud model that reads the feats of a (coords, feats) tuple.

    Returns:
        (model, batch) with batch (coords [1, 20, 3], feats [1, 20, 4]).
    """
    torch.manual_seed(SEED)
    model = PointsNet()
    return as_eval_pair(model, (torch.randn(1, 20, 3), torch.randn(1, 20, 4)))


def make_voxel_indices(count: int, grid: int) -> torch.Tensor:
    """
    Pick distinct voxels of a cubic grid for batch 0.

    Args:
        count: Number of voxels to pick.
        grid: Side length of the cubic grid.

    Returns:
        Int tensor [count, 4] with columns (batch, z, y, x).
    """
    flat = torch.randperm(grid**3)[:count]
    z = torch.div(flat, grid * grid, rounding_mode="floor")
    y = torch.div(flat, grid, rounding_mode="floor") % grid
    x = flat % grid
    batch_column = torch.zeros_like(flat)
    return torch.stack([batch_column, z, y, x], dim=1).to(torch.int32)


def sparse_model() -> ModelPair:
    """
    Build a sparse model with two SparseLinear layers around a SparseDownsample.

    Returns:
        (model, batch) with batch a FakeSparseTensor of 30 voxels, 4 channels, and
        spatial_shape [8, 8, 8]. The output has spatial_shape [4, 4, 4], 6 channels.
    """
    torch.manual_seed(SEED)
    model = nn.Sequential(SparseLinear(4, 8), SparseDownsample(), SparseLinear(8, 6))
    batch = FakeSparseTensor(
        torch.randn(30, 4), make_voxel_indices(30, 8), [8, 8, 8], 1
    )
    return as_eval_pair(model, batch)


def graph_model() -> ModelPair:
    """
    Build a branching convolution graph with a residual, concat, and Python if.

    Returns:
        (model, batch) with batch [1, 3, 8, 8] and output [1, 5, 8, 8].
    """
    torch.manual_seed(SEED)
    return as_eval_pair(GraphNet(), torch.randn(1, 3, 8, 8))


ALL_SHAPES = {
    "image2d": image2d_model,
    "volume3d": volume3d_model,
    "tokens": tokens_model,
    "vector": vector_model,
    "points": points_model,
    "sparse": sparse_model,
    "graph": graph_model,
}
