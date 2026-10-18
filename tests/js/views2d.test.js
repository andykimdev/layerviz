/**
 * Test the pure pixel helpers of grid2d.js and the colormaps of colors.js.
 *
 * Purpose:
 * - Pin how positions land on image and volume grids, and how tokens wrap.
 * - Check the scatter normalization, the click to position mapping, and every color mode.
 *
 * Special Notes:
 * - Records are built by hand in the shape format.js returns, so no file decoding runs here.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  imagePixels,
  slicePixels,
  tokenGrid,
  scatterPoints,
  nearestCell,
  displaySize,
  scatterDot,
  scatterTarget,
  tokenIndex,
  tokenPixels,
} from "../../layerviz/web/grid2d.js";
import {
  VIRIDIS,
  similarityColor,
  channelColor,
  colorsFrom,
  legend,
} from "../../layerviz/web/colors.js";

/**
 * Build a decoded layer record whose positions are the row major cells of a grid.
 *
 * @param {number[]} grid Spatial shape, one size per axis.
 * @param {object} extra Header fields and record fields to override.
 * @returns {object} A record with header, positions, rgb, scores and rgbJoint.
 */
function gridRecord(grid, extra = {}) {
  const n = grid.reduce((a, b) => a * b, 1);
  const positions = [];
  for (let i = 0; i < n; i += 1) {
    let rest = i;
    const cell = [];
    for (let axis = grid.length - 1; axis >= 0; axis -= 1) {
      cell.unshift(rest % grid[axis]);
      rest = Math.floor(rest / grid[axis]);
    }
    positions.push(...cell);
  }
  return {
    header: { n, dim: grid.length, k: 2, grid, scale: null, ...extra.header },
    positions: new Float32Array(extra.positions ?? positions),
    rgb: new Uint8Array(n * 3),
    scores: new Float32Array(n * 2),
    rgbJoint: null,
  };
}

/**
 * Make N*3 colors where position i has the color [i, 10 + i, 20 + i].
 *
 * @param {number} n Number of positions.
 * @returns {Uint8Array} The colors.
 */
function indexColors(n) {
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i += 1) {
    out.set([i, 10 + i, 20 + i], i * 3);
  }
  return out;
}

/**
 * Expand position indices into the RGBA bytes indexColors gives them, -1 for an empty cell.
 *
 * @param {number[]} order Position index per pixel in row major order, or -1.
 * @returns {number[]} The expected RGBA bytes.
 */
function rgbaOf(order) {
  return order.flatMap((i) => (i < 0 ? [0, 0, 0, 0] : [i, 10 + i, 20 + i, 255]));
}

test("imagePixels lays a 2 by 3 grid out row major with alpha 255", () => {
  const record = gridRecord([2, 3]);
  const image = imagePixels(record, indexColors(6));
  assert.equal(image.width, 3);
  assert.equal(image.height, 2);
  assert.ok(image.data instanceof Uint8ClampedArray);
  assert.deepEqual(Array.from(image.data), rgbaOf([0, 1, 2, 3, 4, 5]));
});

test("imagePixels maps input coordinates through scale and leaves gaps clear", () => {
  // Grid 2 by 2 with stride 4: cell centers sit at 1.5 and 5.5 on each axis.
  const record = gridRecord([2, 2], {
    header: { n: 2, scale: [4, 4] },
    positions: [5.5, 1.5, 1.5, 5.5],
  });
  const image = imagePixels(record, indexColors(2));
  assert.equal(image.width, 2);
  assert.equal(image.height, 2);
  assert.deepEqual(Array.from(image.data), rgbaOf([-1, 1, 0, -1]));
});

test("displaySize multiplies the grid by the scale", () => {
  const record = gridRecord([2, 3], { header: { scale: [4, 2] } });
  assert.deepEqual(displaySize(record), [8, 6]);
});

test("displaySize equals the grid when scale is null", () => {
  assert.deepEqual(displaySize(gridRecord([2, 3])), [2, 3]);
});

// In the 2 by 2 by 2 volume below, the position index is d * 4 + h * 2 + w.
test("slicePixels cuts a 2 by 2 by 2 volume along axis 0", () => {
  const image = slicePixels(gridRecord([2, 2, 2]), indexColors(8), 0, 1);
  assert.deepEqual([image.height, image.width], [2, 2]);
  assert.deepEqual(Array.from(image.data), rgbaOf([4, 5, 6, 7]));
});

test("slicePixels cuts a 2 by 2 by 2 volume along axis 1", () => {
  const image = slicePixels(gridRecord([2, 2, 2]), indexColors(8), 1, 0);
  assert.deepEqual([image.height, image.width], [2, 2]);
  assert.deepEqual(Array.from(image.data), rgbaOf([0, 1, 4, 5]));
});

test("slicePixels cuts a 2 by 2 by 2 volume along axis 2", () => {
  const image = slicePixels(gridRecord([2, 2, 2]), indexColors(8), 2, 1);
  assert.deepEqual([image.height, image.width], [2, 2]);
  assert.deepEqual(Array.from(image.data), rgbaOf([1, 3, 5, 7]));
});

test("slicePixels gives an axis 0 slice of a 1 by 2 by 3 volume 2 rows and 3 columns", () => {
  const image = slicePixels(gridRecord([1, 2, 3]), indexColors(6), 0, 0);
  assert.deepEqual([image.height, image.width], [2, 3]);
  assert.deepEqual(Array.from(image.data), rgbaOf([0, 1, 2, 3, 4, 5]));
});

test("slicePixels gives an axis 1 slice of a 1 by 2 by 3 volume 1 row and 3 columns", () => {
  const image = slicePixels(gridRecord([1, 2, 3]), indexColors(6), 1, 1);
  assert.deepEqual([image.height, image.width], [1, 3]);
  assert.deepEqual(Array.from(image.data), rgbaOf([3, 4, 5]));
});

test("slicePixels gives an axis 2 slice of a 1 by 2 by 3 volume 1 row and 2 columns", () => {
  const image = slicePixels(gridRecord([1, 2, 3]), indexColors(6), 2, 2);
  assert.deepEqual([image.height, image.width], [1, 2]);
  assert.deepEqual(Array.from(image.data), rgbaOf([2, 5]));
});

test("tokenGrid wraps 10 tokens into 4 columns and 3 rows", () => {
  assert.deepEqual(tokenGrid(10), { cols: 4, rows: 3 });
  assert.deepEqual(tokenGrid(9), { cols: 3, rows: 3 });
  assert.deepEqual(tokenGrid(1), { cols: 1, rows: 1 });
});

test("scatterPoints normalizes PC1 and PC2 to span 0 to 1", () => {
  const record = gridRecord([3]);
  record.header.k = 3;
  record.scores = new Float32Array([-2, 10, 99, 0, 20, 99, 2, 30, 99]);
  assert.deepEqual(Array.from(scatterPoints(record)), [0, 0, 0.5, 0.5, 1, 1]);
});

test("scatterPoints centers an axis with no spread", () => {
  const record = gridRecord([2]);
  record.scores = new Float32Array([4, 1, 4, 3]);
  assert.deepEqual(Array.from(scatterPoints(record)), [0.5, 0, 0.5, 1]);
});

test("scatterTarget maps the center pixel of a drawn dot back to its point", () => {
  for (const point of [[0, 0], [0.5, 0.25], [1, 1]]) {
    const [left, top] = scatterDot(point);
    assert.deepEqual(scatterTarget(top + 2, left + 2), point, `point ${point}`);
  }
});

test("scatterDot puts PC1 0 at the left edge and PC2 1 at the top edge", () => {
  assert.deepEqual(scatterDot([0, 1]), [0, 0]);
});

test("nearestCell returns the position closest to a grid cell", () => {
  const record = gridRecord([2, 2], {
    header: { n: 2, scale: [4, 4] },
    positions: [5.5, 1.5, 1.5, 5.5],
  });
  assert.equal(nearestCell(record, [1, 0]), 0);
  assert.equal(nearestCell(record, [0, 1]), 1);
  assert.equal(nearestCell(record, [0.4, 0]), 0);
});

/**
 * Build a tokens record of 5 tokens that keeps only tokens 0, 2 and 4.
 *
 * @returns {object} A record with grid [5], scale [1], and positions 0, 2, 4.
 */
function sparseTokens() {
  return gridRecord([5], { header: { n: 3, scale: [1] }, positions: [0, 2, 4] });
}

test("tokenPixels wraps token colors into rows with the tail left clear", () => {
  const image = tokenPixels(gridRecord([3]), indexColors(3));
  assert.deepEqual([image.width, image.height], [2, 2]);
  assert.deepEqual(Array.from(image.data), rgbaOf([0, 1, 2, -1]));
});

test("tokenPixels places subsampled tokens at their positions", () => {
  const image = tokenPixels(sparseTokens(), indexColors(3));
  assert.deepEqual([image.width, image.height], [3, 2]);
  assert.deepEqual(Array.from(image.data), rgbaOf([0, -1, 1, -1, 2, -1]));
});

test("tokenPixels treats a null grid as one token per position", () => {
  const record = gridRecord([4]);
  record.header.grid = null;
  const image = tokenPixels(record, indexColors(4));
  assert.deepEqual([image.width, image.height], [2, 2]);
  assert.deepEqual(Array.from(image.data), rgbaOf([0, 1, 2, 3]));
});

test("tokenIndex maps a wrapped cell back to its position index", () => {
  const record = gridRecord([10]);
  assert.equal(tokenIndex(record, [0, 0]), 0);
  assert.equal(tokenIndex(record, [1, 2]), 6);
  assert.equal(tokenIndex(record, [2, 1]), 9);
  assert.equal(tokenIndex(record, [2, 2]), -1);
});

test("tokenIndex rejects a column past the wrap width", () => {
  assert.equal(tokenIndex(gridRecord([10]), [0, 4]), -1);
});

test("tokenIndex finds subsampled tokens and rejects dropped ones", () => {
  const record = sparseTokens();
  assert.equal(tokenIndex(record, [0, 2]), 1);
  assert.equal(tokenIndex(record, [1, 1]), 2);
  assert.equal(tokenIndex(record, [0, 1]), -1);
  assert.equal(tokenIndex(record, [1, 2]), -1);
});

test("similarityColor returns the first table entry at 0", () => {
  assert.equal(VIRIDIS.length, 16);
  assert.deepEqual(similarityColor(0), VIRIDIS[0]);
});

test("similarityColor returns the last table entry at 1", () => {
  assert.deepEqual(similarityColor(1), VIRIDIS[15]);
});

test("similarityColor blends entries 7 and 8 at 0.5", () => {
  // Entry 7 is [35, 136, 142] and entry 8 is [31, 152, 139].
  assert.deepEqual(similarityColor(0.5), [33, 144, 141]);
});

test("similarityColor clamps ranks outside 0 to 1", () => {
  assert.deepEqual(similarityColor(-1), VIRIDIS[0]);
  assert.deepEqual(similarityColor(2), VIRIDIS[15]);
});

test("channelColor maps a value to an equal gray", () => {
  assert.deepEqual(channelColor(0), [0, 0, 0]);
  assert.deepEqual(channelColor(200), [200, 200, 200]);
});

test("colorsFrom returns rgb for pca and rgbJoint for joint", () => {
  const record = gridRecord([2]);
  record.rgb = new Uint8Array([1, 2, 3, 4, 5, 6]);
  record.rgbJoint = new Uint8Array([7, 8, 9, 10, 11, 12]);
  assert.deepEqual(Array.from(colorsFrom(record, "pca")), [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(Array.from(colorsFrom(record, "joint")), [7, 8, 9, 10, 11, 12]);
});

test("colorsFrom maps ranks through viridis and channel values to gray", () => {
  const record = gridRecord([2]);
  const similarity = colorsFrom(record, "similarity", new Float32Array([0, 1]));
  assert.deepEqual(Array.from(similarity), [...VIRIDIS[0], ...VIRIDIS[15]]);
  const channel = colorsFrom(record, "channel", new Uint8Array([5, 250]));
  assert.deepEqual(Array.from(channel), [5, 5, 5, 250, 250, 250]);
});

test("colorsFrom names the problem for a missing joint fit or an unknown mode", () => {
  const record = gridRecord([2]);
  assert.throws(() => colorsFrom(record, "joint"), /rgbJoint/);
  assert.throws(() => colorsFrom(record, "hsv"), /hsv/);
});

test("colorsFrom fails fast when similarity has no ranks", () => {
  assert.throws(() => colorsFrom(gridRecord([2]), "similarity"), /similarity mode needs 2 ranks/);
});

test("colorsFrom fails fast when channel has no column", () => {
  assert.throws(() => colorsFrom(gridRecord([2]), "channel"), /channel mode needs 2 values/);
});

test("colorsFrom fails fast when the ranks do not match the position count", () => {
  const record = gridRecord([2]);
  assert.throws(() => colorsFrom(record, "similarity", [0.5]), /needs 2 ranks, got 1/);
});

test("legend labels each mode", () => {
  assert.equal(legend("pca"), "PCA RGB of PC1 to PC3");
  assert.equal(legend("joint"), "PCA RGB, joint fit");
  assert.equal(legend("similarity"), "Similarity percentile, low to high");
  assert.equal(legend("channel"), "Channel value, low to high");
});
