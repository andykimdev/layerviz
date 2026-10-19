/**
 * Test the geometry3d.js helpers behind the 3D point view and the caption update of views3d.js.
 *
 * Purpose:
 * - Check the slab mask, the bounding box and sphere, and ray picking on hand built points.
 * - Check the stored column to screen axis map, the pane fitted canvas size, and the caption.
 *
 * Special Notes:
 * - Node has no WebGL, so renderPoints is covered by the Playwright tasks instead.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  slabMask, boundsOf, nearestOnRay, rgbaColors, clampSlab, screenAxes, toScreen, canvasSize,
  DEFAULT_SIDE_PX,
} from "../../layerviz/web/geometry3d.js";
import { showCaption } from "../../layerviz/web/views3d.js";

const Z_LINE = new Float32Array([
  0, 0, 0,
  0, 0, 1,
  0, 0, 2,
  0, 0, 3,
  0, 0, 4,
]);

test("slabMask keeps the 3 points along z within half the thickness of the center", () => {
  const mask = slabMask(Z_LINE, 5, 2, 2, 2);
  assert.ok(mask instanceof Uint8Array);
  assert.deepEqual([...mask], [0, 1, 1, 1, 0]);
});

test("slabMask measures along the named axis only", () => {
  const mask = slabMask(Z_LINE, 5, 0, 0, 0.5);
  assert.deepEqual([...mask], [1, 1, 1, 1, 1]);
});

test("boundsOf a unit cube gives its corners, center, and half diagonal", () => {
  const cube = new Float32Array([
    0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
    0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
  ]);
  const bounds = boundsOf(cube, 8);
  assert.deepEqual(bounds.min, [0, 0, 0]);
  assert.deepEqual(bounds.max, [1, 1, 1]);
  assert.deepEqual(bounds.center, [0.5, 0.5, 0.5]);
  assert.ok(Math.abs(bounds.radius - Math.sqrt(3) / 2) < 1e-9);
});

test("boundsOf rejects an empty point set", () => {
  assert.throws(() => boundsOf(new Float32Array(0), 0), /at least one position/);
});

test("nearestOnRay picks the point on the ray over one beside it", () => {
  const points = new Float32Array([
    0.5, 0, 5,
    0, 0, 9,
    0.2, 0.1, 1,
  ]);
  const index = nearestOnRay(points, 3, [0, 0, -10], [0, 0, 2]);
  assert.equal(index, 1);
});

test("nearestOnRay skips points the mask hides and returns -1 when none remain", () => {
  const points = new Float32Array([0, 0, 1, 3, 0, 1]);
  assert.equal(nearestOnRay(points, 2, [0, 0, 0], [0, 0, 1], new Uint8Array([0, 1])), 1);
  assert.equal(nearestOnRay(points, 2, [0, 0, 0], [0, 0, 1], new Uint8Array([0, 0])), -1);
});

test("nearestOnRay skips points behind the ray origin", () => {
  const points = new Float32Array([0, 0, -1, 0.5, 0, 3]);
  assert.equal(nearestOnRay(points, 2, [0, 0, 0], [0, 0, 1]), 1);
  assert.equal(nearestOnRay(points, 1, [0, 0, 0], [0, 0, 1]), -1);
});

const BOX = { min: [0, 0, 0], max: [4, 2, 10], center: [2, 1, 5] };

test("clampSlab fills a missing or invalid slab with the whole z extent", () => {
  const whole = { axis: 2, center: 5, thickness: 10 };
  assert.deepEqual(clampSlab(undefined, BOX), whole);
  assert.deepEqual(clampSlab({ axis: 3, center: 1, thickness: 1 }, BOX), whole);
  assert.deepEqual(clampSlab({ axis: 0, center: NaN, thickness: 1 }, BOX), whole);
});

test("clampSlab pulls a slab from another layer back inside the new bounds", () => {
  assert.deepEqual(clampSlab({ axis: 0, center: 50, thickness: 30 }, BOX),
    { axis: 0, center: 4, thickness: 4 });
  assert.deepEqual(clampSlab({ axis: 1, center: -3, thickness: -1 }, BOX),
    { axis: 1, center: 0, thickness: 0 });
});

test("clampSlab keeps a slab that already fits the bounds", () => {
  assert.deepEqual(clampSlab({ axis: 1, center: 1.5, thickness: 0.5 }, BOX),
    { axis: 1, center: 1.5, thickness: 0.5 });
});

test("rgbaColors copies each rgb and sets alpha 255 inside the slab and 0 outside", () => {
  const colors = new Uint8Array([10, 20, 30, 40, 50, 60]);
  const rgba = rgbaColors(colors, new Uint8Array([1, 0]), 2);
  assert.ok(rgba instanceof Uint8Array);
  assert.deepEqual([...rgba], [10, 20, 30, 255, 40, 50, 60, 0]);
});

test("rgbaColors rejects colors that are not 3 bytes per position", () => {
  assert.throws(() => rgbaColors(new Uint8Array(5), new Uint8Array(2), 2), /needs 6 color bytes/);
});

test("screenAxes draws sparse3d stored column 2 on x and column 0, named z, on z", () => {
  assert.deepEqual(screenAxes("sparse3d"), { columns: [2, 1, 0], names: ["x", "y", "z"] });
});

test("screenAxes keeps the stored order of a points layer", () => {
  assert.deepEqual(screenAxes("points"), { columns: [0, 1, 2], names: ["x", "y", "z"] });
});

test("screenAxes rejects a kind the 3D view does not draw", () => {
  assert.throws(() => screenAxes("image2d"), /no 3D axis order for layer kind image2d/);
});

test("toScreen draws a sparse3d voxel at a large stored column 2 far out on x", () => {
  const stored = new Float32Array([0, 0, 0, 1, 2, 50]);
  const screen = toScreen(stored, 2, screenAxes("sparse3d").columns);
  assert.ok(screen instanceof Float32Array);
  assert.deepEqual([...screen], [0, 0, 0, 50, 2, 1]);
});

test("toScreen leaves a points layer in its stored order", () => {
  const stored = new Float32Array([1, 2, 3, 4, 5, 6]);
  assert.deepEqual([...toScreen(stored, 2, screenAxes("points").columns)], [1, 2, 3, 4, 5, 6]);
});

test("a slab on the screen z axis of a sparse3d layer filters stored column 0", () => {
  const stored = new Float32Array([0, 0, 9, 5, 0, 0]);
  const screen = toScreen(stored, 2, screenAxes("sparse3d").columns);
  assert.deepEqual([...slabMask(screen, 2, 2, 5, 1)], [0, 1]);
});

const PANE = { width: 800, height: 600, padding: [12, 12, 12, 12], reserved: 50, overflowing: false };

test("canvasSize fills the pane content box less padding and the other pane content", () => {
  assert.deepEqual(canvasSize(PANE), { width: 776, height: 526 });
});

test("canvasSize floors fractional room to whole pixels", () => {
  assert.deepEqual(canvasSize({ ...PANE, width: 800.7, reserved: 50.4 }), { width: 776, height: 525 });
});

test("canvasSize falls back to the default side when the pane has no room yet", () => {
  const empty = { width: 0, height: 74, padding: [12, 12, 12, 12], reserved: 50, overflowing: false };
  assert.deepEqual(canvasSize(empty), { width: DEFAULT_SIDE_PX, height: DEFAULT_SIDE_PX });
});

test("canvasSize falls back to the default height when the room is too short to read", () => {
  assert.deepEqual(canvasSize({ ...PANE, height: 150 }), { width: 776, height: DEFAULT_SIDE_PX });
});

test("canvasSize keeps the height of a pane sized by its content that does not overflow", () => {
  assert.deepEqual(canvasSize(PANE, 540), { width: 776, height: 540 });
});

test("canvasSize shrinks an overflowing pane to its room and grows into new room", () => {
  assert.deepEqual(canvasSize({ ...PANE, overflowing: true }, 700), { width: 776, height: 526 });
  assert.deepEqual(canvasSize(PANE, 400), { width: 776, height: 526 });
});

/**
 * Build a fake view root that holds element stubs and finds them by class name.
 *
 * @param {object[]} children Element stubs already inside the root.
 * @returns {object} A root with children, querySelector, and prepend.
 */
function fakeRoot(children) {
  return {
    children,
    querySelector: (selector) => children.find((child) => `.${child.className}` === selector)
      ?? null,
    prepend(child) { children.unshift(child); },
  };
}

test("showCaption rewrites the legend line already in the root", () => {
  const line = { className: "layerviz-legend", textContent: "pca" };
  const root = fakeRoot([line, { className: "canvas" }]);
  showCaption(root, "similarity to the clicked position");
  assert.equal(root.children.length, 2);
  assert.equal(root.children[0], line);
  assert.equal(line.textContent, "similarity to the clicked position");
});

test("showCaption adds a legend line first when the root has none", (t) => {
  const previousDocument = globalThis.document;
  t.after(() => { globalThis.document = previousDocument; });
  globalThis.document = { createElement: (tag) => ({ tag, className: "", textContent: "" }) };
  const root = fakeRoot([{ className: "canvas" }]);
  showCaption(root, "norm");
  assert.deepEqual(root.children[0], { tag: "div", className: "layerviz-legend", textContent: "norm" });
  assert.equal(root.children.length, 2);
});
