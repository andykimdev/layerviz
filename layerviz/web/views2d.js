/**
 * Draw image2d, volume3d, tokens, and vector layers on canvases from a decoded layer record.
 *
 * Purpose:
 * - Render each view kind into a root element with a legend line for the color mode.
 * - Map canvas clicks back to a position through the pure helpers in grid2d.js.
 *
 * Special Notes:
 * - Only this module touches the DOM. grid2d.js holds the pixel and cell math Node tests.
 */

import { legend } from "./colors.js";
import { nearestIndex } from "./similarity.js";
import {
  SCATTER_PX,
  SCATTER_DOT_PX,
  displaySize,
  imagePixels,
  slicePixels,
  tokenPixels,
  tokenIndex,
  scatterPoints,
  scatterDot,
  scatterTarget,
  nearestCell,
} from "./grid2d.js";

const TOKEN_CELL_PX = 24;

/**
 * Empty a root element and give it a legend line for the current colors.
 *
 * @param {HTMLElement} root Element the view renders into.
 * @param {object} state View state, whose optional mode names the color mode.
 */
function resetRoot(root, state) {
  root.replaceChildren();
  if (state.mode) {
    const caption = document.createElement("div");
    caption.className = "layerviz-legend";
    caption.textContent = legend(state.mode);
    root.append(caption);
  }
}

/**
 * Create a canvas showing an image, scaled up by CSS with crisp pixels.
 *
 * @param {{width: number, height: number, data: Uint8ClampedArray}} image Pixels to draw.
 * @param {number} cssWidth Displayed width in CSS pixels.
 * @param {number} cssHeight Displayed height in CSS pixels.
 * @returns {HTMLCanvasElement} The canvas.
 */
function imageCanvas(image, cssWidth, cssHeight) {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.style.imageRendering = "pixelated";
  drawImage(canvas, image);
  return canvas;
}

/**
 * Put an image onto a canvas of the same size.
 *
 * @param {HTMLCanvasElement} canvas Target canvas.
 * @param {{width: number, height: number, data: Uint8ClampedArray}} image Pixels to draw.
 */
function drawImage(canvas, image) {
  if (image.width && image.height) {
    const pixels = new ImageData(image.data, image.width, image.height);
    canvas.getContext("2d").putImageData(pixels, 0, 0);
  }
}

/**
 * Return the canvas pixel under a pointer event, as [row, col] at canvas resolution.
 *
 * @param {MouseEvent} event Pointer event on the canvas.
 * @param {HTMLCanvasElement} canvas The canvas the event hit.
 * @returns {number[]} The [row, col] pixel.
 */
function pixelUnder(event, canvas) {
  const box = canvas.getBoundingClientRect();
  const col = Math.floor(((event.clientX - box.left) * canvas.width) / box.width);
  const row = Math.floor(((event.clientY - box.top) * canvas.height) / box.height);
  return [row, col];
}

/**
 * Call state.onPick with the position a click maps to, when both exist.
 *
 * @param {HTMLCanvasElement} canvas Canvas to listen on.
 * @param {object} state View state with an optional onPick(positionIndex).
 * @param {function(number[]): number} positionOf Maps a [row, col] pixel to a position.
 */
function onCanvasPick(canvas, state, positionOf) {
  canvas.addEventListener("click", (event) => {
    const position = positionOf(pixelUnder(event, canvas));
    if (position >= 0 && state.onPick) {
      state.onPick(position);
    }
  });
}

/**
 * Render an image2d layer as one canvas at grid resolution, shown at input size.
 *
 * @param {HTMLElement} root Element to render into, cleared first.
 * @param {object} record Decoded layer record with a 2 axis grid.
 * @param {Uint8Array} colors N*3 colors to draw.
 * @param {object} [state] View state with optional mode and onPick.
 */
export function renderImage2D(root, record, colors, state = {}) {
  resetRoot(root, state);
  const [inputHeight, inputWidth] = displaySize(record);
  const canvas = imageCanvas(imagePixels(record, colors), inputWidth, inputHeight);
  onCanvasPick(canvas, state, (cell) => nearestCell(record, cell));
  root.append(canvas);
}

/**
 * Render one slice canvas and its range input for a volume3d layer.
 *
 * @param {object} record Decoded layer record with a 3 axis grid.
 * @param {Uint8Array} colors N*3 colors to draw.
 * @param {object} state View state whose slice array holds the index per axis.
 * @param {number} axis Axis the slice is orthogonal to.
 * @returns {HTMLElement} A wrapper holding the canvas and the slider.
 */
function sliceControl(record, colors, state, axis) {
  const { grid } = record.header;
  const input = displaySize(record);
  const [rowAxis, colAxis] = [0, 1, 2].filter((a) => a !== axis);
  const canvas = imageCanvas(
    slicePixels(record, colors, axis, state.slice[axis]),
    input[colAxis],
    input[rowAxis],
  );
  const slider = document.createElement("input");
  Object.assign(slider, { type: "range", min: 0, max: grid[axis] - 1, step: 1 });
  slider.value = String(state.slice[axis]);
  slider.addEventListener("input", () => {
    state.slice[axis] = Number(slider.value);
    drawImage(canvas, slicePixels(record, colors, axis, state.slice[axis]));
  });
  onCanvasPick(canvas, state, ([row, col]) => {
    const cell = [0, 0, 0];
    [cell[axis], cell[rowAxis], cell[colAxis]] = [state.slice[axis], row, col];
    return nearestCell(record, cell);
  });
  const wrapper = document.createElement("div");
  wrapper.className = "layerviz-slice";
  wrapper.append(canvas, slider);
  return wrapper;
}

/**
 * Render a volume3d layer as three orthogonal slice canvases with one slider each.
 *
 * @param {HTMLElement} root Element to render into, cleared first.
 * @param {object} record Decoded layer record with a 3 axis grid.
 * @param {Uint8Array} colors N*3 colors to draw.
 * @param {object} state View state. state.slice is set to the grid centers when missing
 *   and rewritten as sliders move. Optional mode and onPick.
 */
export function renderVolume3D(root, record, colors, state) {
  resetRoot(root, state);
  const { grid } = record.header;
  if (!Array.isArray(state.slice) || state.slice.length !== 3) {
    state.slice = grid.map((size) => Math.floor(size / 2));
  }
  state.slice = state.slice.map((index, a) => Math.min(Math.max(index, 0), grid[a] - 1));
  for (const axis of [0, 1, 2]) {
    root.append(sliceControl(record, colors, state, axis));
  }
}

/**
 * Render a tokens layer as a wrapped grid with each token's label shown on hover.
 *
 * @param {HTMLElement} root Element to render into, cleared first.
 * @param {object} record Decoded layer record, with optional header.labels per position.
 * @param {Uint8Array} colors N*3 colors to draw.
 * @param {object} [state] View state with optional mode and onPick.
 */
export function renderTokens(root, record, colors, state = {}) {
  resetRoot(root, state);
  const image = tokenPixels(record, colors);
  const canvas = imageCanvas(image, image.width * TOKEN_CELL_PX, image.height * TOKEN_CELL_PX);
  const labels = record.header.labels ?? [];
  canvas.addEventListener("mousemove", (event) => {
    const index = tokenIndex(record, pixelUnder(event, canvas));
    canvas.title = index >= 0 ? String(labels[index] ?? index) : "";
  });
  onCanvasPick(canvas, state, (cell) => tokenIndex(record, cell));
  root.append(canvas);
}

/**
 * Render a vector layer as a scatter of the batch by PC1 and PC2.
 *
 * @param {HTMLElement} root Element to render into, cleared first.
 * @param {object} record Decoded layer record with scores.
 * @param {Uint8Array} colors N*3 colors to draw.
 * @param {object} [state] View state with optional mode and onPick.
 */
export function renderVector(root, record, colors, state = {}) {
  resetRoot(root, state);
  const points = scatterPoints(record);
  const canvas = document.createElement("canvas");
  canvas.width = SCATTER_PX;
  canvas.height = SCATTER_PX;
  const context = canvas.getContext("2d");
  for (let i = 0; i < points.length / 2; i += 1) {
    context.fillStyle = `rgb(${colors[i * 3]}, ${colors[i * 3 + 1]}, ${colors[i * 3 + 2]})`;
    const [left, top] = scatterDot([points[i * 2], points[i * 2 + 1]]);
    context.fillRect(left, top, SCATTER_DOT_PX, SCATTER_DOT_PX);
  }
  onCanvasPick(canvas, state, ([row, col]) => nearestIndex(points, 2, scatterTarget(row, col)));
  root.append(canvas);
}
