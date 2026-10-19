/**
 * Compute the pure geometry behind the 3D point view, with no DOM or WebGL access.
 *
 * Purpose:
 * - Map stored position columns to screen axes per layer kind, and size the canvas to its pane.
 * - Hold the slab mask, bounds, ray picking, and RGBA colors that views3d.js draws with.
 *
 * Special Notes:
 * - Every function takes plain arrays and numbers, so Node tests them directly.
 */

export const DEFAULT_SIDE_PX = 480;
const MIN_SIDE_PX = 160;
const SCREEN_NAMES = ["x", "y", "z"];
// A sparse3d layer stores voxel indices as (z, y, x), while a points layer stores (x, y, z).
const STORED_NAMES = { sparse3d: ["z", "y", "x"], points: ["x", "y", "z"] };

/**
 * Return which stored column each screen axis draws for a layer kind, and the axis names.
 *
 * @param {string} kind Layer kind, sparse3d or points.
 * @returns {{columns: number[], names: string[]}} columns[a] is the stored column drawn on
 *   screen axis a, and names[a] is that column's name.
 * @throws {Error} When the kind has no 3D axis order.
 */
export function screenAxes(kind) {
  const stored = STORED_NAMES[kind];
  if (!stored) {
    throw new Error(`no 3D axis order for layer kind ${kind}, expected sparse3d or points`);
  }
  const columns = SCREEN_NAMES.map((name) => stored.indexOf(name));
  return { columns, names: columns.map((column) => stored[column]) };
}

/**
 * Reorder N*3 stored coordinates into screen axis order.
 *
 * @param {ArrayLike<number>} positions N*3 stored coordinates.
 * @param {number} n Position count.
 * @param {number[]} columns Stored column per screen axis, from screenAxes.
 * @returns {Float32Array} N*3 coordinates whose axis a holds stored column columns[a].
 */
export function toScreen(positions, n, columns) {
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i += 1) {
    for (let a = 0; a < 3; a += 1) {
      out[i * 3 + a] = positions[i * 3 + columns[a]];
    }
  }
  return out;
}

/**
 * Return the canvas size that fills a pane content box beside the rest of the pane content.
 *
 * @param {{width: number, height: number, padding: number[], reserved: number,
 *   overflowing: boolean}} pane Client size, padding as [top, right, bottom, left], the outer
 *   height of everything in the pane except the canvas box, and whether the pane scrolls.
 * @param {number} [current] Canvas height now, or undefined before the first fit.
 * @returns {{width: number, height: number}} Whole pixels. A side without room for a
 *   readable canvas falls back to DEFAULT_SIDE_PX.
 */
export function canvasSize(pane, current) {
  const [top, right, bottom, left] = pane.padding;
  const width = Math.floor(pane.width - left - right);
  const room = Math.floor(pane.height - top - bottom - pane.reserved);
  let height = room >= MIN_SIDE_PX ? room : DEFAULT_SIDE_PX;
  // A pane sized by its content reports room just under the canvas, so shrink only on overflow.
  if (current !== undefined && height < current && !pane.overflowing) {
    height = current;
  }
  return { width: width > 0 ? width : DEFAULT_SIDE_PX, height };
}

/**
 * Mark the positions whose coordinate on one axis lies inside a slab.
 *
 * @param {ArrayLike<number>} positions N*3 coordinates.
 * @param {number} n Position count.
 * @param {number} axis Axis index 0, 1, or 2 the slab is measured along.
 * @param {number} center Slab center on that axis.
 * @param {number} thickness Full slab thickness. Points within half of it are inside.
 * @returns {Uint8Array} One byte per position, 1 inside the slab and 0 outside.
 */
export function slabMask(positions, n, axis, center, thickness) {
  const half = thickness / 2;
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i += 1) {
    mask[i] = Math.abs(positions[i * 3 + axis] - center) <= half ? 1 : 0;
  }
  return mask;
}

/**
 * Return the axis aligned bounds of a point set and the sphere around them.
 *
 * @param {ArrayLike<number>} positions N*3 coordinates.
 * @param {number} n Position count, at least 1.
 * @returns {{min: number[], max: number[], center: number[], radius: number}} Corners, the
 *   box center, and half the box diagonal.
 * @throws {Error} When n is 0, since an empty set has no bounds.
 */
export function boundsOf(positions, n) {
  if (n < 1) {
    throw new Error("boundsOf needs at least one position, got 0");
  }
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i += 1) {
    for (let a = 0; a < 3; a += 1) {
      min[a] = Math.min(min[a], positions[i * 3 + a]);
      max[a] = Math.max(max[a], positions[i * 3 + a]);
    }
  }
  const center = min.map((low, a) => (low + max[a]) / 2);
  const radius = Math.hypot(...min.map((low, a) => max[a] - low)) / 2;
  return { min, max, center, radius };
}

/**
 * Return the visible position closest to a ray, measured perpendicular to the ray line.
 *
 * Positions behind the ray origin are skipped, so the camera never picks what it cannot see.
 *
 * @param {ArrayLike<number>} positions N*3 coordinates.
 * @param {number} n Position count.
 * @param {number[]} origin Ray origin [x, y, z].
 * @param {number[]} direction Ray direction [x, y, z], any nonzero length.
 * @param {Uint8Array} [mask] Optional visibility per position. Zeros are never picked.
 * @returns {number} The position index, or -1 when no position is visible ahead of origin.
 */
export function nearestOnRay(positions, n, origin, direction, mask) {
  const length = Math.hypot(...direction);
  const unit = direction.map((value) => value / length);
  let best = -1;
  let bestDistance = Infinity;
  for (let i = 0; i < n; i += 1) {
    if (mask && !mask[i]) {
      continue;
    }
    const offset = [0, 1, 2].map((a) => positions[i * 3 + a] - origin[a]);
    const along = offset[0] * unit[0] + offset[1] * unit[1] + offset[2] * unit[2];
    if (along < 0) {
      continue;
    }
    const distance = Math.hypot(...offset.map((value, a) => value - along * unit[a]));
    if (distance < bestDistance) {
      best = i;
      bestDistance = distance;
    }
  }
  return best;
}

/**
 * Join N*3 colors and a slab mask into N*4 colors whose alpha hides points outside.
 *
 * @param {ArrayLike<number>} colors N*3 color bytes from colors.js colorsFrom.
 * @param {Uint8Array} mask One byte per position, nonzero when visible.
 * @param {number} n Position count.
 * @returns {Uint8Array} N*4 bytes, alpha 255 for visible points and 0 for hidden ones.
 * @throws {Error} When colors does not hold 3 bytes per position.
 */
export function rgbaColors(colors, mask, n) {
  if (colors?.length !== n * 3) {
    throw new Error(`rgbaColors needs ${n * 3} color bytes, got ${colors?.length ?? "none"}`);
  }
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i += 1) {
    out[i * 4] = colors[i * 3];
    out[i * 4 + 1] = colors[i * 3 + 1];
    out[i * 4 + 2] = colors[i * 3 + 2];
    out[i * 4 + 3] = mask[i] ? 255 : 0;
  }
  return out;
}

/**
 * Return a slab that fits the bounds, so one kept from another layer cannot hide every point.
 *
 * @param {{axis: number, center: number, thickness: number}} [slab] Slab to fit, if any.
 * @param {{min: number[], max: number[], center: number[]}} bounds Bounds of the points.
 * @returns {{axis: number, center: number, thickness: number}} A new slab whose center lies
 *   inside the bounds on its axis and whose thickness is at most that extent. A missing or
 *   invalid slab becomes the whole z extent.
 */
export function clampSlab(slab, bounds) {
  const valid = slab && [0, 1, 2].includes(slab.axis)
    && Number.isFinite(slab.center) && Number.isFinite(slab.thickness);
  if (!valid) {
    return { axis: 2, center: bounds.center[2], thickness: bounds.max[2] - bounds.min[2] };
  }
  const { axis } = slab;
  const low = bounds.min[axis];
  const high = bounds.max[axis];
  return {
    axis,
    center: Math.min(Math.max(slab.center, low), high),
    thickness: Math.min(Math.max(slab.thickness, 0), high - low),
  };
}
