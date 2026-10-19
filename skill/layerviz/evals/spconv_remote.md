# Scenario: spconv repo on a remote server

A sparse 3D detection backbone that only runs on a remote GPU server where spconv is
installed. spconv is not installed locally, and the scenario must not need it.

## Setup

```bash
mkdir -p det3d
touch det3d/__init__.py
cat > det3d/backbone.py <<'PY'
"""Build the sparse voxel backbone of the detector."""
import spconv.pytorch as spconv
from torch import nn


class VoxelBackbone(nn.Module):
    """Encode a SparseConvTensor of voxels with submanifold and strided sparse convs."""

    def __init__(self, in_channels=4):
        super().__init__()
        self.conv_input = spconv.SubMConv3d(in_channels, 16, 3, padding=1, indice_key="subm1")
        self.conv1 = spconv.SparseConv3d(16, 32, 3, stride=2, padding=1)
        self.conv2 = spconv.SubMConv3d(32, 32, 3, padding=1, indice_key="subm2")

    def forward(self, x):
        return self.conv2(self.conv1(self.conv_input(x)))
PY
cat > det3d/data.py <<'PY'
"""Load lidar sweeps and voxelize them into features and integer voxel indices."""
import numpy as np


def load_sweep(path):
    """Return the [N, 4] float32 points (x, y, z, intensity) of one .bin sweep."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def voxelize(points, voxel_size=0.1, grid=(41, 1600, 1408)):
    """Return mean features [M, 4] and (z, y, x) voxel indices [M, 3] inside grid."""
    coords = np.floor(points[:, [2, 1, 0]] / voxel_size).astype(np.int32)
    keep = np.all((coords >= 0) & (coords < np.array(grid)), axis=1)
    coords, points = coords[keep], points[keep]
    uniq, inverse = np.unique(coords, axis=0, return_inverse=True)
    feats = np.zeros((len(uniq), 4), np.float32)
    np.add.at(feats, inverse, points)
    return feats / np.bincount(inverse)[:, None], uniq
PY
```

## Prompt

My lidar backbone in det3d/ only runs on my GPU server, reachable as ssh gpu-box, where the
repo lives at ~/det3d and the conda env has spconv and CUDA. spconv is not installed on this
laptop and I do not want it here. Sweeps are at ~/data/sweeps/*.bin on the server and the
trained weights at ~/det3d/ckpt/backbone.pth. I want to look at what the sparse conv layers
encode, in a browser on my laptop. The layerviz source is at {{LAYERVIZ_REPO}} on this
laptop, and the server can pip install from git. Write what is needed here and give me the
exact commands to run on the server and on the laptop. Do not ssh anywhere yourself.

## Expected outcome

- [ ] Does not install or import spconv locally and does not run capture locally.
- [ ] Writes layerviz_adapter.py with build_model(checkpoint=None) building VoxelBackbone on cuda.
- [ ] load_sample voxelizes one sweep with det3d.data and returns a spconv SparseConvTensor on cuda.
- [ ] Gives the server install as python -m pip install git+https://github.com/andykimdev/layerviz.git without asking for the URL.
- [ ] Gives python -m layerviz capture on the server with --checkpoint ~/det3d/ckpt/backbone.pth.
- [ ] Gives python -m layerviz serve on the server with a fixed --port and runs it in the background.
- [ ] Gives ssh -L forwarding that port to 127.0.0.1 on the laptop and the URL to open.
- [ ] Tells the user to check the bundle with check_bundle.py on the server.
