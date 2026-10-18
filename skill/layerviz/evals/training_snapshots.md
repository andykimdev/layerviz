# Scenario: training loop that needs snapshots

A short training script. Claude must add periodic snapshots with one capture call, run the
training, and serve the snapshot root.

## Setup

```bash
cat > train.py <<'PY'
"""Train a tiny conv classifier on synthetic stripes for a few hundred steps."""
import torch
from torch import nn


def make_batch(size=32):
    """Return images whose label is the stripe orientation."""
    labels = torch.randint(0, 2, (size,))
    axis = torch.arange(32).float()
    rows = torch.sin(axis)[None, :, None].expand(size, 32, 32)
    images = torch.where(labels[:, None, None] == 1, rows, rows.transpose(1, 2))
    return images[:, None] + 0.3 * torch.randn(size, 1, 32, 32), labels


def main():
    """Train for 600 steps and print the loss every 100 steps."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 16, 3, padding=1),
                          nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(600):
        images, labels = make_batch()
        loss = nn.functional.cross_entropy(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(step, loss.item())


if __name__ == "__main__":
    main()
PY
```

## Prompt

train.py trains a small conv net. I want to watch how the conv layers' features change over
training, with a snapshot every 200 steps. The layerviz source is at {{LAYERVIZ_REPO}} and
python on PATH already has torch. Add it to the training, run train.py once, and tell me how
to look at the snapshots.

## Expected outcome

- [ ] Installs layerviz with pip install -e and the local path.
- [ ] Picks one fixed batch before the loop, rather than capturing each new random batch.
- [ ] Adds one layerviz.capture call after optimizer.step with step=step and every=200.
- [ ] Leaves the rest of the loop unchanged.
- [ ] Runs train.py and finds step_000000, step_000200, and step_000400 folders under one snapshot root.
- [ ] Checks one step folder with check_bundle.py.
- [ ] Serves the snapshot root with python -m layerviz serve and gives the URL to open.
