/**
 * Test the grid cell, token wrap, and scatter helpers in grid2d.js on hand built records.
 *
 * Purpose:
 * - Check that scatter scaling stays finite on layer sized inputs.
 * - Pin the grid edge cases and the per record token cell cache.
 *
 * Special Notes:
 * - Records mirror the shape parseLayer returns, built inline with typed arrays.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  imagePixels, scatterPoints, tokenCells, tokenGrid, tokenIndex,
} from "../../layerviz/web/grid2d.js";

const LARGE_N = 200000;

/**
 * Build a scatter record with n positions of k scores each, score i*k+c equal to i+c.
 *
 * @param {number} n Number of positions.
 * @param {number} k Number of scores per position.
 * @returns {object} A record with header.k and scores.
 */
function scatterRecord(n, k) {
  const scores = new Float32Array(n * k);
  for (let i = 0; i < n; i += 1) {
    for (let c = 0; c < k; c += 1) {
      scores[i * k + c] = i + c;
    }
  }
  return { header: { k }, scores };
}

test("scatterPoints over 200000 positions scales both axes into [0, 1]", () => {
  const points = scatterPoints(scatterRecord(LARGE_N, 2));
  assert.equal(points.length, LARGE_N * 2);
  assert.ok(points.every(Number.isFinite));
  assert.deepEqual([points[0], points[1]], [0, 0]);
  assert.deepEqual([points.at(-2), points.at(-1)], [1, 1]);
});

test("scatterPoints with k = 1 puts every point at 0.5 on the second axis", () => {
  const points = scatterPoints(scatterRecord(3, 1));
  assert.deepEqual(Array.from(points), [0, 0.5, 0.5, 0.5, 1, 0.5]);
});

test("imagePixels maps a position past the grid edge to the last cell", () => {
  const record = {
    header: { grid: [2, 2], scale: null },
    positions: new Float32Array([5, 9]),
  };
  const image = imagePixels(record, new Uint8Array([10, 20, 30]));
  const expected = new Uint8ClampedArray(16);
  expected.set([10, 20, 30, 255], 12);
  assert.deepEqual(image, { width: 2, height: 2, data: expected });
});

test("tokenGrid of 0 tokens is a 0 by 0 grid", () => {
  assert.deepEqual(tokenGrid(0), { cols: 0, rows: 0 });
});

test("tokenCells returns the same cells object on a second call for one record", () => {
  const record = { header: { grid: null, dim: 1 }, positions: new Float32Array([0, 2, 3]) };
  const first = tokenCells(record);
  assert.deepEqual(first, { tokens: new Int32Array([0, 2, 2]), total: 3 });
  assert.equal(tokenCells(record), first);
});

test("tokenIndex answers from the cached cells on repeated calls", () => {
  const record = { header: { grid: [4], scale: null }, positions: new Float32Array([0, 3]) };
  assert.deepEqual(
    [tokenIndex(record, [0, 0]), tokenIndex(record, [1, 1]), tokenIndex(record, [0, 1])],
    [0, 1, -1],
  );
  assert.equal(tokenCells(record).tokens, tokenCells(record).tokens);
});
