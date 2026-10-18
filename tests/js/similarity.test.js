/**
 * Test the pure similarity helpers that turn a click into per position similarity.
 *
 * Purpose:
 * - Check float16 decoding, cosine similarity, percentile ranks, and nearest position.
 * - Cross check against a NumPy fixture when LAYERVIZ_FIXTURE names one.
 *
 * Special Notes:
 * - tests/test_similarity_reference.py writes the fixture and runs this file.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  float16ToFloat32,
  cosineSimilarity,
  percentileRank,
  nearestIndex,
} from "../../layerviz/web/similarity.js";

/**
 * Assert two numeric sequences have equal length and match within a tolerance.
 *
 * @param {ArrayLike<number>} actual Values under test.
 * @param {ArrayLike<number>} expected Reference values.
 * @param {number} tolerance Largest allowed absolute difference.
 */
function assertClose(actual, expected, tolerance) {
  assert.equal(actual.length, expected.length, "length differs");
  for (let i = 0; i < expected.length; i += 1) {
    const diff = Math.abs(actual[i] - expected[i]);
    assert.ok(diff <= tolerance, `index ${i}: ${actual[i]} vs ${expected[i]}`);
  }
}

/**
 * Build a flat array of grid points with integer coordinates.
 *
 * @param {number[]} sizes Number of points along each axis.
 * @returns {Float32Array} Points in row major order, sizes.length values each.
 */
function gridPositions(sizes) {
  const points = [[]];
  for (const size of sizes) {
    const next = [];
    for (const prefix of points) {
      for (let v = 0; v < size; v += 1) next.push([...prefix, v]);
    }
    points.splice(0, points.length, ...next);
  }
  return Float32Array.from(points.flat());
}

test("float16ToFloat32 decodes zero, signed zero, one, max, subnormal, and infinity", () => {
  const bits = Uint16Array.from([0x0000, 0x8000, 0x3c00, 0x7bff, 0x0001, 0x7c00, 0xfc00]);
  const out = float16ToFloat32(bits);
  assert.ok(out instanceof Float32Array);
  assert.deepEqual(Array.from(out), [0, -0, 1, 65504, Math.fround(2 ** -24), Infinity, -Infinity]);
  assert.ok(Object.is(out[1], -0), "negative zero lost its sign");
  assert.ok(Math.abs(out[4] - 6e-8) < 1e-9, "smallest subnormal is about 6e-8");
});

test("float16ToFloat32 decodes NaN and negative normals", () => {
  const out = float16ToFloat32(Uint16Array.from([0x7e00, 0xc000, 0x3555]));
  assert.ok(Number.isNaN(out[0]));
  assert.equal(out[1], -2);
  assert.equal(out[2], Math.fround(0.333251953125));
});

test("cosineSimilarity matches a hand computed 3 by 2 matrix", () => {
  const scores = Float32Array.from([1, 0, 0, 2, 3, 4]);
  const sim = cosineSimilarity(scores, 3, 2, 2);
  assert.ok(sim instanceof Float32Array);
  assertClose(sim, [0.6, 0.8, 1], 1e-6);
});

test("cosineSimilarity gives 0 for a zero norm row", () => {
  const scores = Float32Array.from([0, 0, 1, 1]);
  assert.deepEqual(Array.from(cosineSimilarity(scores, 2, 2, 0)), [0, 0]);
  assertClose(cosineSimilarity(scores, 2, 2, 1), [0, 1], 1e-6);
});

test("cosineSimilarity throws a RangeError naming the index and n for an index out of range", () => {
  const scores = Float32Array.from([1, 0, 0, 2, 3, 4]);
  const expected = { name: "RangeError", message: /index 3 .*n = 3/ };
  assert.throws(() => cosineSimilarity(scores, 3, 2, 3), expected);
  assert.throws(() => cosineSimilarity(scores, 3, 2, -1), { name: "RangeError", message: /index -1 .*n = 3/ });
});

test("percentileRank gives tied values the average rank", () => {
  const ranks = percentileRank(Float32Array.from([30, 20, 10, 20]));
  assert.ok(ranks instanceof Float32Array);
  assert.deepEqual(Array.from(ranks), [1, 0.5, 0, 0.5]);
});

test("percentileRank gives 0 for a single value and nothing for no values", () => {
  assert.deepEqual(Array.from(percentileRank(Float32Array.from([5]))), [0]);
  assert.deepEqual(Array.from(percentileRank(new Float32Array(0))), []);
});

test("nearestIndex finds the closest point on a 2D grid", () => {
  const positions = gridPositions([3, 3]);
  assert.equal(nearestIndex(positions, 2, [1.9, 0.2]), 6);
  assert.equal(nearestIndex(positions, 2, [-5, 9]), 2);
});

test("nearestIndex picks the nearer of two positions for a point between them", () => {
  const positions = gridPositions([3, 3]);
  assert.equal(nearestIndex(positions, 2, [1, 1.4]), 4);
  assert.equal(nearestIndex(positions, 2, [1, 1.6]), 5);
});

/**
 * A point at the exact midpoint of two positions resolves to the lower index, so a click
 * on a boundary always picks the same position.
 */
test("nearestIndex returns the lower index for a point at an exact midpoint", () => {
  const positions = gridPositions([3, 3]);
  assert.equal(nearestIndex(positions, 2, [1, 1.5]), 4);
});

test("nearestIndex returns -1 for empty positions", () => {
  assert.equal(nearestIndex(new Float32Array(0), 3, [0, 0, 0]), -1);
});

test("nearestIndex finds the closest point on a 3D grid", () => {
  const positions = gridPositions([2, 2, 2]);
  assert.equal(nearestIndex(positions, 3, [0.9, 0.1, 0.8]), 5);
  assert.equal(nearestIndex(positions, 3, [0.4, 0.6, 0.45]), 2);
});

test("browser similarity matches the NumPy reference fixture", (t) => {
  const path = process.env.LAYERVIZ_FIXTURE;
  if (!path) {
    t.skip("LAYERVIZ_FIXTURE is not set, run tests/test_similarity_reference.py");
    return;
  }
  const fixture = JSON.parse(readFileSync(path, "utf8"));
  const scores = float16ToFloat32(Uint16Array.from(fixture.scores_uint16));
  const sim = cosineSimilarity(scores, fixture.n, fixture.k, fixture.index);
  assertClose(sim, fixture.similarity, 1e-3);
  assertClose(percentileRank(sim), fixture.ranks, 1e-6);
});
