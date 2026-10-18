/**
 * Map decoded layer positions onto grid cells, pixel buffers, and scatter coordinates.
 *
 * Purpose:
 * - Lay per position colors out as pixels on a layer grid, a volume slice, or a token wrap.
 * - Map canvas pixels and scatter clicks back to a position index.
 *
 * Special Notes:
 * - Pure ES module with no DOM access, so Node tests it directly.
 * - Positions are input coordinates. A position lands in grid cell floor(p / scale) on each
 *   axis, with scale from header.scale, or a stride of 1 when that is null.
 */

import { nearestIndex } from "./similarity.js";

export const SCATTER_PX = 320;
export const SCATTER_DOT_PX = 4;

/**
 * Return the stride per axis between grid cells, in input coordinates.
 *
 * @param {object} header Layer header with grid and scale, a list or null.
 * @returns {number[]} header.scale, or 1 on every axis when it is null.
 */
function strideOf(header) {
  return header.scale ?? header.grid.map(() => 1);
}

/**
 * Return the input size a layer covers, used as the CSS size of its canvas.
 *
 * @param {object} record Decoded layer record with header.grid and header.scale.
 * @returns {number[]} grid times scale on each axis.
 */
export function displaySize(record) {
  const stride = strideOf(record.header);
  return record.header.grid.map((size, a) => size * stride[a]);
}

/**
 * Return the grid cell of every position, as N*dim integer cell indices.
 *
 * @param {object} record Decoded layer record with header.grid and positions.
 * @returns {Int32Array} Cell index per axis per position, clamped inside the grid.
 */
function cellsOf(record) {
  const { grid } = record.header;
  const stride = strideOf(record.header);
  const dim = grid.length;
  const n = record.positions.length / dim;
  const cells = new Int32Array(n * dim);
  for (let i = 0; i < n; i += 1) {
    for (let a = 0; a < dim; a += 1) {
      const cell = Math.floor(record.positions[i * dim + a] / stride[a]);
      cells[i * dim + a] = Math.min(Math.max(cell, 0), grid[a] - 1);
    }
  }
  return cells;
}

/**
 * Paint positions into an RGBA buffer, leaving unpainted pixels fully transparent.
 *
 * @param {number} width Image width in pixels.
 * @param {number} height Image height in pixels.
 * @param {Uint8Array} colors N*3 colors.
 * @param {function(number): number} pixelOf Maps a position to its pixel index, or -1.
 * @returns {{width: number, height: number, data: Uint8ClampedArray}} The image.
 */
function paint(width, height, colors, pixelOf) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < colors.length / 3; i += 1) {
    const pixel = pixelOf(i);
    if (pixel >= 0) {
      data.set([colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2], 255], pixel * 4);
    }
  }
  return { width, height, data };
}

/**
 * Lay an image2d layer out on its grid as RGBA pixels in row major order.
 *
 * @param {object} record Decoded layer record with a 2 axis header.grid of [height, width].
 * @param {Uint8Array} colors N*3 colors to draw.
 * @returns {{width: number, height: number, data: Uint8ClampedArray}} The image at grid
 *   resolution. Cells with no position stay transparent.
 */
export function imagePixels(record, colors) {
  const [height, width] = record.header.grid;
  const cells = cellsOf(record);
  return paint(width, height, colors, (i) => cells[i * 2] * width + cells[i * 2 + 1]);
}

/**
 * Lay one orthogonal slice of a volume3d layer out as RGBA pixels.
 *
 * @param {object} record Decoded layer record with a 3 axis header.grid.
 * @param {Uint8Array} colors N*3 colors to draw.
 * @param {number} axis Axis the slice is orthogonal to, 0, 1 or 2.
 * @param {number} index Cell index of the slice along that axis.
 * @returns {{width: number, height: number, data: Uint8ClampedArray}} The slice, with the
 *   lower remaining axis as height and the higher one as width.
 */
export function slicePixels(record, colors, axis, index) {
  const { grid } = record.header;
  const [rowAxis, colAxis] = [0, 1, 2].filter((a) => a !== axis);
  const width = grid[colAxis];
  const cells = cellsOf(record);
  return paint(width, grid[rowAxis], colors, (i) =>
    cells[i * 3 + axis] === index ? cells[i * 3 + rowAxis] * width + cells[i * 3 + colAxis] : -1,
  );
}

/**
 * Choose the wrap of a token sequence into a near square grid.
 *
 * @param {number} n Number of tokens.
 * @returns {{cols: number, rows: number}} cols is ceil(sqrt(n)), rows holds the rest.
 */
export function tokenGrid(n) {
  const cols = Math.ceil(Math.sqrt(n));
  return { cols, rows: cols ? Math.ceil(n / cols) : 0 };
}

/**
 * Return a tokens record whose header always has a one axis grid.
 *
 * @param {object} record Decoded tokens layer record, header.grid may be null.
 * @returns {object} The record, or a copy with grid [N] when the grid is null.
 */
function withTokenGrid(record) {
  if (record.header.grid) {
    return record;
  }
  const n = record.positions.length / (record.header.dim ?? 1);
  return { ...record, header: { ...record.header, grid: [n] } };
}

/**
 * Return the sequence index of every kept token, from its position.
 *
 * @param {object} record Decoded tokens layer record.
 * @returns {{tokens: Int32Array, total: number}} Token index per position, and the
 *   sequence length the grid wraps.
 */
function tokenCells(record) {
  const gridded = withTokenGrid(record);
  return { tokens: cellsOf(gridded), total: gridded.header.grid[0] };
}

/**
 * Lay token colors out on their wrapped grid as RGBA pixels, each at its own token.
 *
 * @param {object} record Decoded tokens layer record, possibly subsampled.
 * @param {Uint8Array} colors N*3 colors, one per position.
 * @returns {{width: number, height: number, data: Uint8ClampedArray}} One pixel per token
 *   of the full sequence. Dropped tokens stay transparent.
 */
export function tokenPixels(record, colors) {
  const { tokens, total } = tokenCells(record);
  const { cols, rows } = tokenGrid(total);
  return paint(cols, rows, colors, (i) => tokens[i]);
}

/**
 * Map a cell of the wrapped token grid back to the position holding that token.
 *
 * @param {object} record Decoded tokens layer record, possibly subsampled.
 * @param {number[]} cell The [row, col] cell.
 * @returns {number} The position index, or -1 when the token was dropped or lies past
 *   the end of the sequence.
 */
export function tokenIndex(record, [row, col]) {
  const { tokens, total } = tokenCells(record);
  const { cols } = tokenGrid(total);
  if (col < 0 || col >= cols) {
    return -1;
  }
  return tokens.indexOf(row * cols + col);
}

/**
 * Rescale values to 0..1, mapping an axis with no spread to 0.5.
 *
 * @param {number[]} values Values on one axis.
 * @returns {number[]} The rescaled values.
 */
function unitScale(values) {
  const low = Math.min(...values);
  const span = Math.max(...values) - low;
  return values.map((v) => (span > 0 ? (v - low) / span : 0.5));
}

/**
 * Return PC1 and PC2 of every position, each rescaled to span 0..1.
 *
 * @param {object} record Decoded layer record with scores of N*k and header.k.
 * @returns {Float32Array} N*2 normalized [PC1, PC2] pairs. PC2 is 0.5 when k is 1.
 */
export function scatterPoints(record) {
  const { k } = record.header;
  const n = record.scores.length / k;
  const pc1 = [];
  const pc2 = [];
  for (let i = 0; i < n; i += 1) {
    pc1.push(record.scores[i * k]);
    pc2.push(k > 1 ? record.scores[i * k + 1] : 0);
  }
  const [x, y] = [unitScale(pc1), unitScale(pc2)];
  return new Float32Array(x.flatMap((value, i) => [value, y[i]]));
}

/**
 * Return the top left canvas pixel of the dot drawn for a scatter point.
 *
 * @param {number[]} point Normalized [PC1, PC2], each in 0..1.
 * @returns {number[]} The [left, top] pixel. PC2 grows upward, so the row is flipped.
 */
export function scatterDot([x, y]) {
  const reach = SCATTER_PX - SCATTER_DOT_PX;
  return [x * reach, (1 - y) * reach];
}

/**
 * Map a clicked canvas pixel to normalized scatter coordinates, centered on the dots.
 *
 * @param {number} row Canvas row of the click.
 * @param {number} col Canvas column of the click.
 * @returns {number[]} The [PC1, PC2] whose dot is centered on that pixel.
 */
export function scatterTarget(row, col) {
  const reach = SCATTER_PX - SCATTER_DOT_PX;
  const half = SCATTER_DOT_PX / 2;
  return [(col - half) / reach, 1 - (row - half) / reach];
}

/**
 * Return the position whose grid cell is closest to a target cell.
 *
 * @param {object} record Decoded layer record with header.grid and positions.
 * @param {number[]} cell Target cell, one index per grid axis.
 * @returns {number} The closest position index, so clicks on empty cells still pick one.
 */
export function nearestCell(record, cell) {
  return nearestIndex(cellsOf(record), record.header.grid.length, cell);
}
