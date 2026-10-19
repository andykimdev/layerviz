/**
 * Draw sparse3d and points layers as a rotatable three.js point cloud with a slab control.
 *
 * Purpose:
 * - Render positions as WebGL points colored by any color mode, with picking by click.
 * - Show only the points inside a slab whose axis, center, and thickness are range inputs.
 *
 * Special Notes:
 * - three.js and OrbitControls load from ./vendor/ so bundles work offline.
 * - slabMask, boundsOf, nearestOnRay, rgbaColors, and clampSlab are pure, so Node tests them.
 */

import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { legend } from "./colors.js";

const DEFAULT_SIDE_PX = 480;
const CLICK_SLOP_PX = 4;
const SLAB_STEPS = 100;
const AXIS_NAMES = ["x", "y", "z"];

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

/**
 * Create a camera looking at the point cloud from outside its bounding sphere.
 *
 * @param {{center: number[], radius: number}} bounds Bounds of the points.
 * @returns {THREE.PerspectiveCamera} The camera.
 */
function makeCamera(bounds) {
  const radius = Math.max(bounds.radius, 1e-6);
  const camera = new THREE.PerspectiveCamera(50, 1, radius / 100, radius * 100);
  const [x, y, z] = bounds.center;
  camera.position.set(x, y, z + radius * 3);
  return camera;
}

/**
 * Create the Points object with a position attribute and an RGBA color attribute.
 *
 * @param {Float32Array} positions N*3 coordinates.
 * @param {Uint8Array} rgba N*4 colors from rgbaColors.
 * @param {number} radius Bounding sphere radius, which sets the point size.
 * @returns {THREE.Points} The points, whose alphaTest discards hidden ones.
 */
function makePoints(positions, rgba, radius) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(rgba, 4, true));
  const material = new THREE.PointsMaterial({
    size: Math.max(radius / 60, 1e-6),
    vertexColors: true,
    transparent: true,
    alphaTest: 0.5,
  });
  return new THREE.Points(geometry, material);
}

/**
 * Create a labelled range input that calls onInput with its numeric value.
 *
 * @param {string} label Text shown before the slider.
 * @param {{min: number, max: number, step: number, value: number}} range Slider settings.
 * @param {function(number): void} onInput Called on every input event.
 * @returns {{wrapper: HTMLElement, input: HTMLInputElement}} The label and its slider.
 */
function rangeControl(label, range, onInput) {
  const input = document.createElement("input");
  Object.assign(input, { type: "range", min: range.min, max: range.max, step: range.step });
  input.value = String(range.value);
  input.addEventListener("input", () => onInput(Number(input.value)));
  const wrapper = document.createElement("label");
  wrapper.className = "layerviz-slab";
  wrapper.append(`${label} `, input);
  return { wrapper, input };
}

/**
 * Build the axis, center, and thickness sliders that rewrite state.slab.
 *
 * @param {object} state View state with a valid slab.
 * @param {{min: number[], max: number[], center: number[]}} bounds Bounds of the points.
 * @param {function(): void} onChange Called after state.slab changes.
 * @returns {HTMLElement} A wrapper holding the three sliders.
 */
function slabControls(state, bounds, onChange) {
  const extent = (axis) => Math.max(bounds.max[axis] - bounds.min[axis], 1e-6);
  const { axis } = state.slab;
  const center = rangeControl("center", {
    min: bounds.min[axis], max: bounds.max[axis], step: extent(axis) / SLAB_STEPS,
    value: state.slab.center,
  }, (value) => { state.slab.center = value; onChange(); });
  const thickness = rangeControl("thickness", {
    min: 0, max: extent(axis), step: extent(axis) / SLAB_STEPS, value: state.slab.thickness,
  }, (value) => { state.slab.thickness = value; onChange(); });
  const axisInput = rangeControl("axis", { min: 0, max: 2, step: 1, value: axis }, (value) => {
    state.slab = { axis: value, center: bounds.center[value], thickness: extent(value) };
    Object.assign(center.input, { min: bounds.min[value], max: bounds.max[value] });
    center.input.step = String(extent(value) / SLAB_STEPS);
    center.input.value = String(state.slab.center);
    Object.assign(thickness.input, { max: extent(value), step: extent(value) / SLAB_STEPS });
    thickness.input.value = String(state.slab.thickness);
    axisInput.input.title = AXIS_NAMES[value];
    onChange();
  });
  axisInput.input.title = AXIS_NAMES[axis];
  const wrapper = document.createElement("div");
  wrapper.className = "layerviz-slab-controls";
  wrapper.append(axisInput.wrapper, center.wrapper, thickness.wrapper);
  return wrapper;
}

/**
 * Call onClick with a pointer event only when the pointer did not drag between down and up.
 *
 * @param {HTMLCanvasElement} canvas Canvas to listen on.
 * @param {function(PointerEvent): void} onClick Called for clicks that are not orbit drags.
 */
function onStillClick(canvas, onClick) {
  let down = null;
  canvas.addEventListener("pointerdown", (event) => { down = [event.clientX, event.clientY]; });
  canvas.addEventListener("pointerup", (event) => {
    if (down && Math.hypot(event.clientX - down[0], event.clientY - down[1]) <= CLICK_SLOP_PX) {
      onClick(event);
    }
    down = null;
  });
}

/**
 * Return the world space ray under a pointer event, as origin and direction arrays.
 *
 * @param {PointerEvent} event Pointer event on the canvas.
 * @param {HTMLCanvasElement} canvas The canvas the event hit.
 * @param {THREE.Camera} camera Camera the canvas renders through.
 * @returns {{origin: number[], direction: number[]}} The ray.
 */
function rayUnder(event, canvas, camera) {
  const box = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((event.clientX - box.left) / box.width) * 2 - 1,
    -((event.clientY - box.top) / box.height) * 2 + 1,
  );
  const caster = new THREE.Raycaster();
  caster.setFromCamera(ndc, camera);
  return { origin: caster.ray.origin.toArray(), direction: caster.ray.direction.toArray() };
}

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
 * Create a renderer sized to root, orbit controls on the camera, and a frame loop drawing scene.
 *
 * @param {HTMLElement} root Element whose size sets the canvas size.
 * @param {THREE.Scene} scene Scene to draw every frame.
 * @param {THREE.PerspectiveCamera} camera Camera to orbit, possibly shared with other panes.
 * @param {{center: number[]}} bounds Bounds of the points, whose center the controls orbit.
 * @returns {{renderer: THREE.WebGLRenderer, controls: OrbitControls}} The running renderer
 *   and its controls, both for the caller to dispose.
 */
function startRenderer(root, scene, camera, bounds) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  if (typeof devicePixelRatio === "number") {
    renderer.setPixelRatio(devicePixelRatio);
  }
  // A compare pane holds only its caption at this point, so its height is no canvas size.
  const height = Math.max(root.clientHeight, DEFAULT_SIDE_PX);
  let width = root.clientWidth || DEFAULT_SIDE_PX;
  renderer.setSize(width, height);
  const canvas = renderer.domElement;
  // The first compare pane is measured before the second exists, so the width follows the pane.
  Object.assign(canvas.style, { width: "100%", boxSizing: "border-box" });
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(...bounds.center);
  controls.update();
  // A frame loop keeps panes that share one camera in step when another pane orbits it.
  renderer.setAnimationLoop(() => {
    if (canvas.clientWidth && canvas.clientWidth !== width) {
      width = canvas.clientWidth;
      renderer.setSize(width, height, false);
    }
    camera.aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1) || 1;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  });
  return { renderer, controls };
}

/**
 * Render a sparse3d or points layer as a rotatable point cloud with slab sliders.
 *
 * @param {HTMLElement} root Element to render into, cleared first.
 * @param {object} record Decoded layer record whose positions are N*3.
 * @param {Uint8Array} colors N*3 colors to draw.
 * @param {object} state View state. state.camera is created when missing and shared by
 *   panes given the same state.camera. state.slab is filled and rewritten by the sliders.
 *   Optional mode and onPick(positionIndex).
 * @returns {{updateColors: function(Uint8Array): void, dispose: function(): void}} A handle
 *   that recolors without rebuilding the scene, and one that stops rendering.
 * @throws {Error} When the record positions are not 3 dimensional or the layer is empty.
 */
export function renderPoints(root, record, colors, state) {
  const { n, dim } = record.header;
  if (dim !== 3) {
    throw new Error(`renderPoints needs 3 dimensional positions, this layer has dim ${dim}`);
  }
  resetRoot(root, state);
  const bounds = boundsOf(record.positions, n);
  state.slab = clampSlab(state.slab, bounds);
  state.camera ??= makeCamera(bounds);
  let current = colors;
  const maskNow = () => slabMask(record.positions, n, state.slab.axis, state.slab.center,
    state.slab.thickness);
  let mask = maskNow();
  const points = makePoints(record.positions, rgbaColors(current, mask, n), bounds.radius);
  const scene = new THREE.Scene();
  scene.add(points);
  const { renderer, controls } = startRenderer(root, scene, state.camera, bounds);
  const recolor = () => {
    points.geometry.getAttribute("color").array.set(rgbaColors(current, mask, n));
    points.geometry.getAttribute("color").needsUpdate = true;
  };
  const sliders = slabControls(state, bounds, () => { mask = maskNow(); recolor(); });
  onStillClick(renderer.domElement, (event) => {
    const { origin, direction } = rayUnder(event, renderer.domElement, state.camera);
    const index = nearestOnRay(record.positions, n, origin, direction, mask);
    if (index >= 0 && state.onPick) {
      state.onPick(index);
    }
  });
  root.append(renderer.domElement, sliders);
  return {
    updateColors(next) {
      current = next;
      recolor();
    },
    dispose() {
      renderer.setAnimationLoop(null);
      controls.dispose();
      points.geometry.dispose();
      points.material.dispose();
      renderer.dispose();
    },
  };
}
