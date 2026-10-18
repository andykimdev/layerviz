/**
 * Turn a layer record and a color mode into the per position colors every view draws.
 *
 * Purpose:
 * - Map similarity ranks through a viridis table and channel values to gray.
 * - Pick the N*3 colors for PCA, joint PCA, similarity, or channel mode, with a legend label.
 *
 * Special Notes:
 * - Pure ES module with no DOM access, so Node tests it directly.
 */

/** Viridis sampled at 16 evenly spaced points from 0 to 1, as [r, g, b] bytes. */
export const VIRIDIS = [
  [68, 1, 84], [72, 26, 108], [71, 47, 125], [65, 68, 135],
  [57, 86, 140], [49, 104, 142], [42, 120, 142], [35, 136, 142],
  [31, 152, 139], [34, 168, 132], [53, 183, 121], [84, 197, 104],
  [122, 209, 81], [165, 219, 54], [210, 226, 27], [253, 231, 37],
];

const LEGENDS = {
  pca: "PCA RGB of PC1 to PC3",
  joint: "PCA RGB, joint fit",
  similarity: "Similarity percentile, low to high",
  channel: "Channel value, low to high",
};

/**
 * Map a similarity percentile rank to a viridis color by linear interpolation.
 *
 * @param {number} rank Percentile rank in 0..1. Values outside are clamped.
 * @returns {number[]} The [r, g, b] bytes.
 */
export function similarityColor(rank) {
  const t = Math.min(Math.max(rank, 0), 1) * (VIRIDIS.length - 1);
  const low = Math.floor(t);
  const high = Math.min(low + 1, VIRIDIS.length - 1);
  const frac = t - low;
  return VIRIDIS[low].map((value, c) => Math.round(value + (VIRIDIS[high][c] - value) * frac));
}

/**
 * Map a quantized channel value to a gray color.
 *
 * @param {number} value Channel byte in 0..255.
 * @returns {number[]} The [r, g, b] bytes, all equal to value.
 */
export function channelColor(value) {
  return [value, value, value];
}

/**
 * Fill N*3 colors by applying a color function to each of N values.
 *
 * @param {ArrayLike<number>} values One value per position.
 * @param {function(number): number[]} colorOf Maps a value to [r, g, b].
 * @returns {Uint8Array} The colors.
 */
function mapColors(values, colorOf) {
  const out = new Uint8Array(values.length * 3);
  for (let i = 0; i < values.length; i += 1) {
    out.set(colorOf(values[i]), i * 3);
  }
  return out;
}

/**
 * Check that per position values exist and hold one value per position.
 *
 * @param {object} record Decoded layer record whose header.n is the position count.
 * @param {ArrayLike<number>|undefined} values The values a color mode needs.
 * @param {string} mode Color mode name for the error message.
 * @param {string} noun What the values are, for the error message.
 * @returns {ArrayLike<number>} The values, unchanged.
 * @throws {Error} When values are missing or their length is not header.n.
 */
function requireValues(record, values, mode, noun) {
  const { n } = record.header;
  if (values?.length !== n) {
    throw new Error(`${mode} needs ${n} ${noun}, got ${values?.length ?? "none"}`);
  }
  return values;
}

/**
 * Return the N*3 colors a view should draw for one layer in one color mode.
 *
 * @param {object} record Decoded layer record from format.js parseLayer.
 * @param {string} mode One of pca, joint, similarity, or channel.
 * @param {ArrayLike<number>} [extra] Percentile ranks for similarity, or the Uint8Array
 *   channel column for channel. Unused by pca and joint.
 * @returns {Uint8Array} N*3 colors.
 * @throws {Error} When mode is joint and the record has no rgbJoint, similarity or channel
 *   values are missing or not one per position, or mode is unknown.
 */
export function colorsFrom(record, mode, extra) {
  if (mode === "pca") {
    return record.rgb;
  }
  if (mode === "joint") {
    if (!record.rgbJoint) {
      throw new Error("joint mode needs record.rgbJoint, this layer has no joint fit");
    }
    return record.rgbJoint;
  }
  if (mode === "similarity") {
    return mapColors(requireValues(record, extra, "similarity mode", "ranks"), similarityColor);
  }
  if (mode === "channel") {
    return mapColors(requireValues(record, extra, "channel mode", "values"), channelColor);
  }
  throw new Error(`unknown color mode ${mode}, expected pca, joint, similarity or channel`);
}

/**
 * Return the legend label for a color mode.
 *
 * @param {string} mode One of pca, joint, similarity, or channel.
 * @returns {string} The label, or an empty string for an unknown mode.
 */
export function legend(mode) {
  return LEGENDS[mode] ?? "";
}
