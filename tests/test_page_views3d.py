"""
Test that the 3D point view draws known colors, picks positions, and fits its pane.

Purpose:
- Check which side of the canvas a voxel lands on and what the slab controls hide.
- Check a corner click picks the nearest position in 3D.
- Check color mode and layer switches keep one WebGL context alive without warnings.
- Check the canvas fits the window and follows the pane through resizes.

Special Notes:
- The layers come from the known bundle in tests.views_known.
- A WebGL canvas is read in the animation frame that follows a render, since the page keeps
  no copy of its drawing buffer.
"""

import pytest

from tests.harness import browser, page  # noqa: F401
from tests.views_known import (  # noqa: F401
    GREEN, LEGENDS, RED, click_text, click_view_center, click_view_corner, known_bundle,
    legend_text, open_known, select_known)

MODES = ["pca", "similarity", "channel"]
SWITCHES = 40
GAP_LIMIT_PX = 8


def webgl_blob(page, color):
    """Return the pixel count and mean x and y share of the WebGL pixels near a pure color."""
    script = """(color) => new Promise((resolve) => requestAnimationFrame(() => {
      const source = document.querySelector('#view canvas');
      const copy = document.createElement('canvas');
      [copy.width, copy.height] = [source.width, source.height];
      const context = copy.getContext('2d');
      context.drawImage(source, 0, 0);
      const data = context.getImageData(0, 0, copy.width, copy.height).data;
      let [count, sumX, sumY] = [0, 0, 0];
      for (let i = 0; i < data.length; i += 4) {
        if (color.every((value, a) => Math.abs(data[i + a] - value) <= 60)) {
          count += 1;
          sumX += (i / 4) % copy.width;
          sumY += Math.floor(i / 4 / copy.width);
        }
      }
      resolve({ count, x: sumX / count / copy.width, y: sumY / count / copy.height });
    }))"""
    return page.evaluate(script, color)


def axis_titles(page):
    """Return the title of the slab axis slider at each of its three values."""
    axis = page.locator(".layerviz-slab-controls input").first
    titles = []
    for value in ("0", "1", "2"):
        axis.fill(value)
        titles.append(axis.get_attribute("title"))
    return titles


def console_warnings(page):
    """Return a list that keeps receiving the text of every console warning of the page."""
    warnings = []
    # Software WebGL reports a performance hint on each readback, which is not a page warning.
    quiet = "GL Driver Message"
    page.on("console", lambda msg: warnings.append(msg.text)
            if msg.type == "warning" and quiet not in msg.text else None)
    return warnings


def is_same_canvas(page, canvas):
    """Return whether the handle is still the connected first canvas of the view."""
    script = "c => c.isConnected && c === document.querySelector('#view canvas')"
    return page.evaluate(script, canvas)


def view_metrics(page):
    """Return the vertical overflow of the view and document, the canvas height, and the gap."""
    script = """() => {
      const view = document.querySelector('#view');
      const root = document.documentElement;
      const padding = parseFloat(getComputedStyle(view).paddingBottom);
      return {
        viewOverflow: view.scrollHeight - view.clientHeight,
        pageOverflow: root.scrollHeight - innerHeight,
        canvasHeight: view.querySelector('canvas').getBoundingClientRect().height,
        gap: view.getBoundingClientRect().bottom - padding
          - view.lastElementChild.getBoundingClientRect().bottom,
      };
    }"""
    return page.evaluate(script)


def next_frames(page, count=3):
    """Wait for a few animation frames, so a resize observer and its fit have run."""
    script = """(count) => new Promise((resolve) => {
      const step = (left) => requestAnimationFrame(() => (left ? step(left - 1) : resolve()));
      step(count);
    })"""
    page.evaluate(script, count)


@pytest.mark.parametrize(
    ("right", "bottom", "expected"),
    [(False, False, "0, 10, 0"), (True, True, "10, 0, 0")],
)
def test_a_corner_click_in_3d_picks_the_position_nearest_that_corner(
        page, known_bundle, right, bottom, expected):
    open_known(page, known_bundle)
    select_known(page, "square")
    click_view_corner(page, right, bottom)
    assert click_text(page) == expected


def test_a_voxel_at_a_large_stored_column_2_draws_on_the_right_half(page, known_bundle):
    open_known(page, known_bundle)
    select_known(page, "cloud")
    assert webgl_blob(page, RED)["x"] > 0.5
    assert webgl_blob(page, GREEN)["x"] < 0.5


def test_the_slab_axis_control_lists_the_stored_columns_as_z_y_x(page, known_bundle):
    open_known(page, known_bundle)
    select_known(page, "cloud")
    assert axis_titles(page) == ["z", "y", "x"]


def test_a_slab_on_the_axis_named_z_hides_the_voxel_outside_stored_column_0(page, known_bundle):
    open_known(page, known_bundle)
    select_known(page, "slab")
    assert webgl_blob(page, GREEN)["count"] > 0
    controls = page.locator(".layerviz-slab-controls input")
    controls.nth(0).fill(str(axis_titles(page).index("z")))
    controls.nth(1).fill("0")
    controls.nth(2).fill("0")
    assert webgl_blob(page, GREEN)["count"] == 0
    assert webgl_blob(page, RED)["count"] > 0


def test_switching_the_color_mode_40_times_keeps_the_canvas_and_logs_no_warning(
        page, known_bundle):
    errors = open_known(page, known_bundle)
    warnings = console_warnings(page)
    select_known(page, "square")
    click_view_center(page)
    canvas = page.evaluate_handle("document.querySelector('#view canvas')")
    page.select_option("#channel", "0")
    modes = [MODES[step % 3] for step in range(SWITCHES)]
    seen = []
    for mode in modes:
        page.select_option("#mode", mode)
        page.wait_for_selector(f'#view[data-mode="{mode}"] canvas')
        seen.append((legend_text(page), is_same_canvas(page, canvas)))
    assert seen == [(LEGENDS[mode], True) for mode in modes]
    assert warnings == []
    assert errors == []


def test_switching_between_two_3d_layers_40_times_logs_no_warning(page, known_bundle):
    errors = open_known(page, known_bundle)
    warnings = console_warnings(page)
    for step in range(SWITCHES):
        select_known(page, ["cloud", "square"][step % 2])
    assert warnings == []
    assert errors == []


def test_a_3d_canvas_fits_the_window_and_follows_the_pane_through_resizes(page, known_bundle):
    page.set_viewport_size({"width": 1400, "height": 900})
    open_known(page, known_bundle)
    select_known(page, "cloud")
    metrics = [view_metrics(page)]
    for width, height in [(900, 600), (1600, 1100)]:
        page.set_viewport_size({"width": width, "height": height})
        next_frames(page)
        metrics.append(view_metrics(page))
    assert [(m["viewOverflow"], m["pageOverflow"]) for m in metrics] == [(0, 0)] * 3
    assert all(0 <= m["gap"] <= GAP_LIMIT_PX for m in metrics)
    assert metrics[1]["canvasHeight"] < metrics[0]["canvasHeight"] < metrics[2]["canvasHeight"]
