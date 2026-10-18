# layerviz

layerviz shows what each layer of a PyTorch model encodes. It runs one forward pass, reduces
every layer output to PCA colors in input coordinates, and writes a folder that a browser
page opens. It handles dense 2D layers and sparse 3D layers such as spconv.

## Install

layerviz needs Python 3.10 or newer and torch 2.1 or newer.

```bash
pip install git+https://github.com/andykimdev/layerviz.git
```

For development, clone the repo and run `pip install -e ".[dev]"`.

## Quick start

Capture one model from Python:

```python
import layerviz

layerviz.capture(model, batch, "bundle")
```

Then serve the bundle and open the printed address in a browser:

```bash
python -m layerviz serve bundle
```

The page opens at http://127.0.0.1:8000 by default. Use `--port` to change it.

## Command line

The command line takes a model factory such as `pkg.mod:build` and a sample file.

```bash
python -m layerviz capture --model pkg.mod:build --sample sample.pt --out bundle
```

- **capture:** Writes a bundle for one model.
- **compare:** Writes one bundle that shows two models side by side. Pass `--against` with a
  second checkpoint, or random to compare against random weights.
- **serve:** Serves a bundle folder over http.

Useful flags for capture and compare:

- **`--checkpoint`:** Loads a checkpoint file into the model.
- **`--include` and `--exclude`:** Keep or drop layers by module name pattern.
- **`--layer-mb` and `--total-mb`:** Cap the size of each layer file and of the whole bundle.
- **`--inline-mb`:** Sets the largest page written as one self contained index.html.

## Training snapshots

Pass a step to capture a bundle during training and watch features change over time.

```python
layerviz.capture(model, batch, "runs", step=step, every=1000)
```

## Compare two models

```python
layerviz.compare(model_a, model_b, batch, Path("bundle"))
```

The page follows the layers of model A and shows model B beside it.

## Custom sparse layers

Register a shape plugin with `layerviz.register_shape` so layerviz can read a custom tensor
type. The built-in plugins cover dense tensors and spconv.

## Claude skill

The folder skill/layerviz holds a skill that guides an assistant through installing layerviz,
writing a model adapter, capturing and serving a bundle. To use it, link the folder:

```bash
ln -s "$(pwd)/skill/layerviz" ~/.claude/skills/layerviz
```

## Tests

```bash
python -m pytest -q
node --test 'tests/js/**/*.test.js'
```

The tests in tests/test_spconv_real.py skip unless spconv is installed.

## License

MIT. The bundled three.js files in layerviz/web/vendor keep their own MIT license.
