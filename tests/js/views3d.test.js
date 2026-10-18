/**
 * Test the pure geometry helpers behind the 3D point view in views3d.js.
 *
 * Purpose:
 * - Check the slab mask, the bounding box and sphere, and ray picking on hand built points.
 *
 * Special Notes:
 * - Node has no WebGL, so renderPoints is covered by the Playwright tasks instead.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  slabMask, boundsOf, nearestOnRay, rgbaColors, clampSlab,
} from "../../layerviz/web/views3d.js";

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
