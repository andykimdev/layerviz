# Scenario: 2D CNN repo

A small image classifier repo with a factory function and a trained checkpoint. Claude has
never seen it and must capture it and compare the checkpoint against random weights.

## Setup

```bash
mkdir -p netlib checkpoints
touch netlib/__init__.py
cat > netlib/model.py <<'PY'
"""Build the VGG style classifier used by train.py."""
from torch import nn


def make_net(width=32, num_classes=10):
    """Return a three stage conv classifier for 3 x 64 x 64 images."""
    def stage(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                             nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(stage(3, width), stage(width, 2 * width), stage(2 * width, 4 * width),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * width, num_classes))
PY
python -c "import torch, netlib.model as m; torch.manual_seed(3); torch.save({'model': m.make_net().state_dict(), 'epoch': 30}, 'checkpoints/last.pt')"
```

## Prompt

I have a small image classifier in netlib/model.py and a trained checkpoint at
checkpoints/last.pt. I want to see what each conv stage encodes on one image, and how the
trained network differs from a randomly initialized one. The layerviz source is at
{{LAYERVIZ_REPO}} and python on PATH already has torch. Set it up, run it, and tell me how to
open the result.

## Expected outcome

- [ ] Installs layerviz with pip install -e and the local path, not from an invented URL.
- [ ] Writes layerviz_adapter.py at the project root with build_model(checkpoint=None) and load_sample().
- [ ] build_model calls netlib.model.make_net with the default width, so the checkpoint keys fit.
- [ ] load_sample returns a [1, 3, 64, 64] tensor.
- [ ] Runs python -m layerviz capture with --checkpoint checkpoints/last.pt and the adapter specs.
- [ ] Runs python -m layerviz compare with --against random.
- [ ] Checks each bundle with check_bundle.py, which reports index.html and a node count.
- [ ] Gives a python -m layerviz serve command for each bundle and the URL to open.
