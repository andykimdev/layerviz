/**
 * Compute click to similarity for the viewer in pure JavaScript with no DOM access.
 *
 * Purpose:
 * - Compute cosine similarity of decoded scores against a clicked row.
 * - Turn similarity into percentile ranks and map a 3D point to its nearest position.
 *
 * Special Notes:
 * - Sums accumulate in doubles and only results are stored as Float32Array.
 * - Scores arrive already decoded by decodeFloat16 in format.js.
 */

/**
 * Compute the Euclidean norm of every row of a row major matrix.
 *
 * @param {ArrayLike<number>} scores Flat array of n rows by k columns.
 * @param {number} n Number of rows.
 * @param {number} k Number of columns.
 * @returns {Float64Array} Norm of each row.
 */
function rowNorms(scores, n, k) {
  const norms = new Float64Array(n);
  for (let row = 0; row < n; row += 1) {
    let sum = 0;
    for (let col = row * k; col < row * k + k; col += 1) sum += scores[col] * scores[col];
    norms[row] = Math.sqrt(sum);
  }
  return norms;
}

/**
 * Compute the dot product of two rows of a row major matrix.
 *
 * @param {ArrayLike<number>} scores Flat array of rows with k columns.
 * @param {number} k Number of columns.
 * @param {number} a First row index.
 * @param {number} b Second row index.
 * @returns {number} Dot product of rows a and b.
 */
function rowDot(scores, k, a, b) {
  let sum = 0;
  for (let col = 0; col < k; col += 1) sum += scores[a * k + col] * scores[b * k + col];
  return sum;
}

/**
 * Compute cosine similarity between one row and every row over k score columns.
 *
 * @param {ArrayLike<number>} scores Flat array of n rows by k columns.
 * @param {number} n Number of rows.
 * @param {number} k Number of columns.
 * @param {number} index Row to compare against, the clicked position.
 * @returns {Float32Array} Similarity per row in -1..1, 0 where either row has zero norm
 *   or holds Infinity or NaN, so percentileRank always sees finite values.
 * @throws {RangeError} When index is not an integer in 0..n-1.
 */
export function cosineSimilarity(scores, n, k, index) {
  if (!Number.isInteger(index) || index < 0 || index >= n) {
    throw new RangeError(`cosineSimilarity: index ${index} is outside 0..n-1 for n = ${n}`);
  }
  const norms = rowNorms(scores, n, k);
  const out = new Float32Array(n);
  for (let row = 0; row < n; row += 1) {
    const denom = norms[row] * norms[index];
    out[row] = denom > 0 && Number.isFinite(denom) ? rowDot(scores, k, row, index) / denom : 0;
  }
  return out;
}

/**
 * Rank each value among all values, with ties sharing their average rank.
 *
 * @param {ArrayLike<number>} values Values to rank.
 * @returns {Float32Array} Rank divided by n - 1, in 0..1, all zeros when n is 1.
 */
export function percentileRank(values) {
  const n = values.length;
  const out = new Float32Array(n);
  if (n < 2) return out;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => values[a] - values[b]);
  let start = 0;
  while (start < n) {
    let end = start;
    while (end + 1 < n && values[order[end + 1]] === values[order[start]]) end += 1;
    const rank = (start + end) / 2 / (n - 1);
    for (let i = start; i <= end; i += 1) out[order[i]] = rank;
    start = end + 1;
  }
  return out;
}

/**
 * Find the position closest to a point in squared Euclidean distance.
 *
 * @param {ArrayLike<number>} positions Flat array of positions with dim values each.
 * @param {number} dim Number of coordinates per position.
 * @param {ArrayLike<number>} point Query point with dim coordinates.
 * @returns {number} Index of the nearest position, the lowest index on a tie, -1 when empty.
 */
export function nearestIndex(positions, dim, point) {
  let best = -1;
  let bestDistance = Infinity;
  for (let i = 0; i * dim < positions.length; i += 1) {
    let distance = 0;
    for (let d = 0; d < dim; d += 1) distance += (positions[i * dim + d] - point[d]) ** 2;
    if (distance < bestDistance) {
      best = i;
      bestDistance = distance;
    }
  }
  return best;
}
