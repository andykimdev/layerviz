"""
Test the model loading helpers that resolve specs, read checkpoints, and read sample files.

Purpose:

- Check that parse_spec resolves a module:function spec and names a bad one.
- Check that load_state finds the state dict and strips a wrapper prefix only from every key.
- Check that load_model loads checkpoint weights and names missing or unexpected keys.
- Check that load_sample reads a spec, a .pt, .npy, or .npz file, and rejects other suffixes.
- Check that .pt files load safely first, falling back with one warning for pickled objects.
- Check that load_model leaves the global torch random state as it found it.

Special Notes:

- The factories below are loaded by spec as tests.test_loading:make_model.
"""

import os
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from layerviz.loading import load_model, load_sample, load_state, parse_spec
from tests.models import image2d_model

MODEL_SPEC = "tests.test_loading:make_model"
SAMPLE_SPEC = "tests.test_loading:make_sample"


def make_model() -> nn.Module:
    """Build the image2d architecture with weights drawn from the global torch seed."""
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, 6, 3, stride=2, padding=1)
    )


def make_sample() -> torch.Tensor:
    """Return the image2d batch and its mirror image as one batch of shape [2, 3, 16, 16]."""
    batch = image2d_model()[1]
    return torch.cat([batch, batch.flip(-1)])


def unseeded_model() -> nn.Module:
    """Build a linear layer whose weights depend only on the global torch seed."""
    return nn.Linear(3, 2)


def model_with_a_module_child() -> nn.Module:
    """Build a model whose first child is really named module, beside a head child."""
    return nn.Sequential(OrderedDict(module=nn.Linear(3, 2), head=nn.Linear(2, 1)))


def trained_state() -> dict:
    """Return a state dict whose weights differ from the factory model's."""
    torch.manual_seed(123)
    return {k: torch.randn_like(v) for k, v in make_model().state_dict().items()}


def save_checkpoint(tmp_path: Path, state: dict) -> Path:
    """Save state as a DDP style checkpoint under the model key and return its path."""
    path = tmp_path / "ckpt.pt"
    torch.save({"model": {"module." + k: v for k, v in state.items()}, "step": 7}, path)
    return path


def trained_module() -> nn.Module:
    """Return the factory model holding the trained_state weights."""
    model = make_model()
    model.load_state_dict(trained_state())
    return model


def assert_same_weights(model: nn.Module, state: dict) -> None:
    """Assert that model holds exactly the keys and tensors of state."""
    assert list(model.state_dict()) == list(state)
    assert all(torch.equal(model.state_dict()[k], state[k]) for k in state)


def assert_same_state(loaded: dict, state: dict) -> None:
    """Assert that loaded holds exactly the keys of state, in order, with equal tensors."""
    assert list(loaded) == list(state)
    assert all(torch.equal(loaded[k], state[k]) for k in state)


@pytest.mark.parametrize("spec, expected", [
    ("os.path:join", os.path.join), ("pathlib:Path.cwd", Path.cwd),
])
def test_parse_spec_resolves_a_dotted_attribute_path(spec, expected):
    assert parse_spec(spec) == expected


@pytest.mark.parametrize("spec", ["no_colon", "os.path:missing_name"])
def test_parse_spec_rejects_a_bad_spec_with_its_name(spec):
    with pytest.raises(ValueError, match=spec.split(":")[-1]):
        parse_spec(spec)


@pytest.mark.parametrize("key", ["state_dict", "model", "weights", None])
def test_load_state_finds_the_tensor_dict_and_strips_module(tmp_path, key):
    state = trained_state()
    wrapped = {"module." + k: v for k, v in state.items()}
    payload = wrapped if key is None else {key: wrapped, "epoch": 3}
    torch.save(payload, tmp_path / "c.pt")
    loaded = load_state(tmp_path / "c.pt", "weights" if key == "weights" else None)
    assert_same_state(loaded, state)


@pytest.mark.parametrize("prefix", ["_orig_mod.", "module._orig_mod.", "_orig_mod.module."])
def test_load_state_strips_every_wrapper_prefix_that_all_keys_share(tmp_path, prefix):
    state = trained_state()
    torch.save({prefix + k: v for k, v in state.items()}, tmp_path / "c.pt")
    assert_same_state(load_state(tmp_path / "c.pt"), state)


def test_load_state_keeps_a_real_module_child_when_not_every_key_has_the_prefix(tmp_path):
    state = model_with_a_module_child().state_dict()
    torch.save(state, tmp_path / "c.pt")
    assert_same_state(load_state(tmp_path / "c.pt"), state)


def test_load_model_loads_a_compiled_checkpoint_into_the_plain_model(tmp_path):
    state = trained_state()
    torch.save({"_orig_mod." + k: v for k, v in state.items()}, tmp_path / "c.pt")
    assert_same_weights(load_model(MODEL_SPEC, tmp_path / "c.pt"), state)


def test_load_model_loads_a_model_with_a_real_module_child(tmp_path):
    torch.manual_seed(5)
    state = model_with_a_module_child().state_dict()
    torch.save({"state_dict": state}, tmp_path / "c.pt")
    model = load_model("tests.test_loading:model_with_a_module_child", tmp_path / "c.pt")
    assert_same_weights(model, state)


def test_load_state_prefers_the_state_key_over_state_dict(tmp_path):
    state = trained_state()
    decoy = {k: torch.zeros_like(v) for k, v in state.items()}
    torch.save({"state_dict": decoy, "custom": state}, tmp_path / "c.pt")
    assert_same_state(load_state(tmp_path / "c.pt", "custom"), state)


SAFE_LOADERS = {
    "load_state": lambda path: load_state(path),
    "load_sample": lambda path: load_sample(str(path)),
}


@pytest.mark.parametrize("loader", list(SAFE_LOADERS))
def test_safe_load_reads_a_tensor_file_with_no_warning(tmp_path, loader):
    state = trained_state()
    torch.save(state, tmp_path / "c.pt")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = SAFE_LOADERS[loader](tmp_path / "c.pt")
    assert_same_state(loaded, state)


@pytest.mark.parametrize("loader", list(SAFE_LOADERS))
def test_safe_load_falls_back_for_a_pickled_module_with_one_warning(tmp_path, loader):
    torch.save(trained_module(), tmp_path / "m.pt")
    with pytest.warns(UserWarning) as record:
        loaded = SAFE_LOADERS[loader](tmp_path / "m.pt")
    messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
    assert len(messages) == 1 and str(tmp_path / "m.pt") in messages[0]
    assert isinstance(loaded, nn.Sequential)
    assert_same_weights(loaded, trained_state())


def test_load_model_leaves_the_global_rng_state_as_it_found_it():
    torch.manual_seed(42)
    expected = torch.rand(1)
    torch.manual_seed(42)
    load_model("tests.test_loading:unseeded_model", seed=1)
    assert torch.equal(torch.rand(1), expected)


def test_load_state_returns_a_saved_module_as_is(tmp_path):
    torch.save(trained_module(), tmp_path / "m.pt")
    with pytest.warns(UserWarning, match="m.pt"):
        loaded = load_state(tmp_path / "m.pt")
    assert isinstance(loaded, nn.Sequential)
    assert_same_weights(loaded, trained_state())


def test_load_state_without_a_tensor_dict_names_the_keys(tmp_path):
    torch.save({"epoch": 3}, tmp_path / "c.pt")
    with pytest.raises(ValueError, match="epoch"):
        load_state(tmp_path / "c.pt")


def test_load_model_loads_the_checkpoint_weights(tmp_path):
    state = trained_state()
    model = load_model(MODEL_SPEC, save_checkpoint(tmp_path, state))
    assert_same_weights(model, state)
    assert not model.training


def test_load_model_copies_a_saved_module_into_the_factory_model(tmp_path):
    torch.save(trained_module(), tmp_path / "m.pt")
    with pytest.warns(UserWarning, match="m.pt"):
        model = load_model(MODEL_SPEC, tmp_path / "m.pt")
    assert_same_weights(model, trained_state())
    assert not model.training


def test_load_model_same_seed_gives_the_same_weights():
    a = load_model("tests.test_loading:unseeded_model", seed=1)
    b = load_model("tests.test_loading:unseeded_model", seed=1)
    assert torch.equal(a.weight, b.weight)


def test_load_model_different_seeds_give_different_weights():
    a = load_model("tests.test_loading:unseeded_model", seed=1)
    c = load_model("tests.test_loading:unseeded_model", seed=2)
    assert not torch.equal(a.weight, c.weight)


def test_load_model_names_missing_and_unexpected_keys(tmp_path):
    state = trained_state()
    state["extra.weight"] = state.pop("0.weight")
    with pytest.raises(ValueError, match=r"0\.weight.*extra\.weight"):
        load_model(MODEL_SPEC, save_checkpoint(tmp_path, state))


def test_load_model_names_the_missing_keys_of_a_subset_checkpoint(tmp_path):
    state = trained_state()
    del state["1.weight"], state["1.bias"]
    with pytest.raises(ValueError, match=r"missing keys \['1\.weight', '1\.bias'\], "
                                         r"unexpected keys \[\]"):
        load_model(MODEL_SPEC, save_checkpoint(tmp_path, state))


def test_load_model_names_the_unexpected_keys_of_a_superset_checkpoint(tmp_path):
    state = trained_state()
    state["extra.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match=r"missing keys \[\], "
                                         r"unexpected keys \['extra\.weight'\]"):
        load_model(MODEL_SPEC, save_checkpoint(tmp_path, state))


SAMPLE_WRITERS = {
    "b.pt": lambda path, batch: torch.save(batch, path),
    "b.npy": lambda path, batch: np.save(path, batch.numpy()),
    "one.npz": lambda path, batch: np.savez(path, x=batch.numpy()),
}


def test_load_sample_calls_a_spec():
    assert torch.equal(load_sample(SAMPLE_SPEC), make_sample())


@pytest.mark.parametrize("name", list(SAMPLE_WRITERS))
def test_load_sample_reads_a_single_tensor_file(tmp_path, name):
    batch = make_sample()
    SAMPLE_WRITERS[name](tmp_path / name, batch)
    assert torch.equal(load_sample(str(tmp_path / name)), batch)


def test_load_sample_reads_a_several_array_npz_as_a_dict(tmp_path):
    batch = make_sample()
    np.savez(tmp_path / "two.npz", x=batch.numpy(), y=np.arange(3))
    both = load_sample(str(tmp_path / "two.npz"))
    assert set(both) == {"x", "y"}
    assert torch.equal(both["x"], batch) and torch.equal(both["y"], torch.arange(3))


def test_load_sample_rejects_an_unknown_suffix(tmp_path):
    (tmp_path / "b.csv").write_text("1")
    with pytest.raises(ValueError, match="b.csv"):
        load_sample(str(tmp_path / "b.csv"))
