---
name: layerviz
description: Shows what each layer of a PyTorch model encodes by installing layerviz, writing a small adapter with a model factory and a sample loader, capturing or comparing layer outputs as PCA color maps, and serving the bundle in a browser, also from a remote server or as snapshots during training. Use when the user asks to see what a layer encodes, visualize features or activations, compare checkpoints or a trained model against random weights, or watch features change while training.
---

# layerviz

layerviz reduces every layer output of one forward pass to PCA colors in input coordinates
and writes a bundle folder that a browser page opens. **Install** → **Adapter** →
**Capture or compare** → **Serve** → **Check**.

## Install

Install into the environment that imports the project's model (Python 3.10 or newer, torch
2.1 or newer). Use the local checkout when one exists, otherwise ask for the repo URL.

```bash
pip install -e /path/to/layerviz
pip install git+<repo url>
```

## Adapter

Write `<project>/layerviz_adapter.py` at the project root with two functions. Read
[adapter_example.py](adapter_example.py) for a complete adapter of a torchvision style CNN.

- **`build_model(checkpoint=None)`:** Builds the model with the project's own class and
  config, loads the checkpoint when given, and returns it in eval mode.
- **`load_sample()`:** Returns one batch exactly as the model's forward takes it, with a
  batch dimension. Prefer the first evaluation item after the project's own transform.

The command line calls `build_model()` with no arguments and loads `--checkpoint` itself.
It accepts a state dict, a dict holding one under `state_dict` or `model` (or the key given
by `--state-key`), or a saved nn.Module, which is loaded into the factory model through its
state_dict. A `module.` prefix is stripped. Missing or unexpected keys raise ValueError, so
fix the factory config to match the checkpoint rather than loading loosely.

When the model needs a GPU, move it to the device inside `build_model` and the sample inside
`load_sample`. `--sample` also takes a `.pt`, `.npy`, or `.npz` file instead of a spec.

## Shape overrides

Each output is matched to a shape plugin by its type and rank.

| Plugin | Output it handles |
| --- | --- |
| image2d | [B, C, H, W] tensors |
| volume3d | [B, C, D, H, W] tensors |
| tokens | [B, N, C] tensors |
| vector | [B, C] tensors |
| sparse3d | objects with features, indices, spatial_shape, batch_size (spconv) |
| points | [B, N, C] tensors when load_sample returns a (coords, feats) pair as the sample |

When an output is ambiguous, such as image patches flattened to [B, N, C], force a plugin
through the Python API with `shapes`, a dict from module name patterns (fnmatch) to plugin
names. The command line has no shapes flag. When forward does not take the batch as one
positional argument, pass `forward`.

```python
import layerviz
from layerviz_adapter import build_model, load_sample
layerviz.capture(build_model(), load_sample(), "layerviz_runs/capture",
                 shapes={"encoder.blocks.*": "tokens"},
                 forward=lambda model, batch: model(**batch))
```

## Capture or compare

Run from the project root, so `layerviz_adapter` imports. Each command prints the path of
the written index.html.

```bash
python -m layerviz capture --model layerviz_adapter:build_model --checkpoint best.pt \
  --sample layerviz_adapter:load_sample --out layerviz_runs/capture
python -m layerviz compare --model layerviz_adapter:build_model --checkpoint epoch10.pt \
  --against epoch50.pt --labels epoch10 epoch50 \
  --sample layerviz_adapter:load_sample --out layerviz_runs/compare
python -m layerviz compare --model layerviz_adapter:build_model --checkpoint best.pt \
  --against random --sample layerviz_adapter:load_sample --out layerviz_runs/vs_random
```

- **`--against`:** Takes a second checkpoint path, or the word random for a fresh copy
  seeded with `--seed` plus 1.
- **`--include` and `--exclude`:** Take module name patterns, such as `layer1.*`, to keep
  or drop layers in large models.
- **Size flags:** `--layer-mb` (16) and `--total-mb` (1024) cap the bundle. `--inline-mb`
  (16) is the largest index.html written as one self contained page.
- **Other flags:** `--sample-index` (0) picks the item in the batch, `--seed` (0) seeds
  the factory and the PCA.

From Python, `compare(model_a, model_b, batch, out, options)` takes a frozen CompareOptions.

```python
from layerviz import compare
from layerviz.comparison import CompareOptions
from layerviz.pipeline import CaptureOptions
from layerviz_adapter import build_model, load_sample
options = CompareOptions(labels=("init", "trained"), capture=CaptureOptions(include=["layer*"]))
compare(build_model(), build_model("best.pt"), load_sample(), "layerviz_runs/compare", options)
```

## Snapshots during training

Pick one fixed batch before the loop, so every snapshot sees the same input. Then add one
line after the optimizer step.

```python
import layerviz
viz_batch = next(iter(val_loader))[0][:1].to(device)  # before the loop
layerviz.capture(model, viz_batch, "layerviz_runs/train", step=step, every=500)
```

Steps that are not multiples of `every` return at once and only rank 0 writes, so the line
is safe under DistributedDataParallel and torch.compile, and the model keeps its training
mode. Each capture writes a six digit padded folder such as `step_000200/` and a root
index.html linking them. Serve the root.

## Serve

```bash
python -m layerviz serve layerviz_runs/capture --port 8000
```

Serve runs until interrupted, so start it in the background and read the URL it prints.
Always serve rather than opening index.html from disk, since bundles past `--inline-mb` load
their layer files over http. `--port 0` picks a free port.

On a remote server, install, capture, and serve there, then forward the port from the
user's machine and open http://127.0.0.1:8000/ on it. Serve binds 127.0.0.1 only. Spconv
models capture where spconv runs. layerviz never imports spconv, so never install it locally.

```bash
ssh -N -L 8000:127.0.0.1:8000 user@server
```

## Check

After every capture, confirm the bundle folder holds index.html and its graph lists nodes
with [check_bundle.py](check_bundle.py) in this skill's folder. It reads graph.json, or the
graph inside a one page index.html. For snapshots, check one step folder such as `step_000200/`.

```bash
python /path/to/skills/layerviz/check_bundle.py layerviz_runs/capture
```

It prints the node count. If it exits 1 or a layer is missing, widen `--include` or fix the
adapter, then capture and check again.

## Expected page behavior

In a comparison, a layer pair whose joint flag is false has no joint colors, so the page
disables the joint switch with a note. That is expected, not a bug.
